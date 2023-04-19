import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from PIL import Image
import seaborn as sns
import time
import datetime

from dataset import MaskBaseDataset, MaskDataset, GenderDataset, AgeDataset # dataset.py
from dataset import TestDataset
from loss import create_criterion # loss.py
from f1score import get_F1_Score # f1score.py
from submission import submission # submission.py
from inference import inference, mask_inference, gender_inference, age_inference # inference.py
import wandb


def seed_everything(seed):
    '''
    PyTorch와 Numpy에서 사용되는 랜덤 시드를 설정하는 함수입니다.
    시드 값을 설정하면, 해당 값에 대한 동일한 seed 값으로 다시 실행해도 동일한 결과가 나오도록 보장할 수 있습니다
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    '''
    현재 optimizer의 학습률 (learning rate)을 반환하는 함수
    optimizer의 param_groups 속성을 사용하여 parameter groups의 list를 얻고,
    각 group의 첫 번째 parameter의 학습률 값을 반환합니다.
    이 코드에서는 하나의 parameter group만 사용되고 있으므로, 학습률 값은 하나만 반환됩니다.
    '''
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    '''
    입력으로 받은 이미지들을 그리드 형태로 시각화하는 기능을 수행
    주어진 배치 크기에서 n개의 이미지를 랜덤으로 선택하고,
    해당 이미지들의 ground truth 및 예측값, 그리고 마스크, 성별, 나이의 세 가지 정보를 함께 시각화합니다.
    선택된 이미지들을 n_grid x n_grid 크기의 그리드 형태로 배열하여 반환합니다.
    시각화된 이미지와 함께 각 이미지의 ground truth와 예측값이 제목에 표시됩니다.
    반환된 figure 객체를 이용하여 이미지를 출력할 수 있습니다.
    '''
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).

    경로명을 자동으로 증가시켜주는 함수입니다.
    입력받은 경로가 "runs/exp"이라면 이미 존재하는 경우(즉, exist_ok=True) 그대로 반환하고,
    없는 경우에는 바로 해당 경로를 반환합니다. 
    만약 exist_ok가 False인 경우, 해당 경로가 이미 존재하는 경우 경로명을 증가시켜주어 새로운 경로를 반환합니다. 예를 들어 "runs/exp0"이 이미 존재한다면, "runs/exp1"로 경로를 증가시켜주어 반환합니다.
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def wandb_config(args):
    config_dict  = {'seed'         : args.seed,
                    'epochs'       : args.epochs,
                    'dataset'      : args.dataset,
                    'augmentation' : args.augmentation,
                    'resize'       : args.resize,
                    'batch_size'   : args.batch_size,
                    'valid_batch_size' : args.valid_batch_size,
                    'model'            : args.model,
                    'optimizer'        : args.optimizer,
                    'lr'               : args.lr,
                    'val_ratio'        : args.val_ratio,
                    'criterion'        : args.criterion,
                    'lr_decay_step'    : args.lr_decay_step,
                    'log_interval'     : args.log_interval,
                    'name'             : args.name,
                    'model_dir'        : args.model_dir,
                    'patience_limit'   : args.patience_limit}
    return config_dict

def train(data_dir, model_dir, args):
    '''
    data_dir : 데이터 경로
    model_dir : 모델 경로
    args : 인자

    seed_everything(args.seed) 함수를 호출하여 학습시 고정된 시드를 사용하도록 설정
    save_dir : increment_path 함수를 사용하여 모델이 저장될 경로를 생성
    dataset_module : args.dataset이라는 인자를 통해 사용할 데이터셋을 설정하고 불러옴
    transform_module : 데이터셋에 적용할 데이터 augmentation 기법을 설정
    train_loader : train 데이터 loader
    val_loader   : vaild 데이터 loader
    criterion : create_criterion() 함수를 호출하여 손실 함수(criterion)를 생성
    optimizer : import_module() 함수를 사용하여 torch.optim 모듈에서 사용자가 지정한 최적화 함수(optimizer)를 가져옴
                lr은 학습률(learning rate)을 나타내며, weight_decay는 L2 정규화(regularization)의 강도를 조절합니다.
    scheduler : StepLR은 일정한 스텝(step)마다 학습률을 감소시키는 스케줄러
                args.lr_decay_step은 학습률 감소 스텝의 크기를 나타내며,
                gamma는 감소 비율을 나타냄


    '''
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskPreprocessDataset
    dataset = dataset_module(
        data_dir=data_dir,
        outlier_remove=args.outlier_remove
    )
    
    num_classes = dataset.num_classes # mask : 3, gender : 2, age : 3

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
#         pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=4,
#         num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
#         pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(num_classes=num_classes).to(device)
    model = torch.nn.DataParallel(model)
    
#     # -- freeze
#     freeze = args.freeze
#     if freeze :
#         for param in model.parameters():
#             param.requires_grad = False

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
    optimizer = opt_module(
#         filter(lambda p: p.requires_grad, model.parameters()),
        model.parameters(),
        lr=args.lr,
#         weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    
    
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    
    wandb.init(project = "Mask_Classification", config = wandb_config(args))
    wandb.run.name = args.exp_name
    
    ## ---- starting train ----
    best_val_acc = 0
    best_val_loss = np.inf
    
    # early stop init
    patience_limits = args.patience_limit
    best_loss = 10 ** 9 # 매우 큰 값으로 초기값 가정
    patience_limit = patience_limits # 몇 번의 epoch까지 지켜볼지를 결정
    patience_check = 0 # 현재 몇 epoch 연속으로 loss 개선이 안되는지를 기록
    
    # time
    start_time = time.time()
    for epoch in range(args.epochs):
        midel_time = time.time()
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        valid_f1_score = get_F1_Score()
        train_f1_score = get_F1_Score()
        
        for idx, (inputs,labels) in enumerate(train_loader):
            inputs, labels = inputs.cuda(),labels.cuda()

            optimizer.zero_grad()

            outs = model(inputs) # batch_size, label
            preds = torch.argmax(outs, dim=-1)
            if args.criterion == 'f1' or args.criterion == 'label_smoothing':
                criterion.classes = num_classes

            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                train_f1_score.update(preds, labels)
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch+1}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || train_f1_score {train_f1_score.get_score :4.2} || lr {current_lr}"
                )
                wandb.log({"train acc": train_acc, "train loss": train_loss, 'train_f1_score' : train_f1_score.get_score}, step = epoch)
                loss_value = 0
                matches = 0

#         scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            
            for inputs,labels in val_loader:
                inputs, labels = inputs.cuda(),labels.cuda()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)                
                valid_f1_score.update(preds, labels)
                
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)

#                 if figure is None:
#                     inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
#                     inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
#                     figure = grid_image(
#                         inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
            
            # early stop
            if val_loss > best_loss: # loss가 개선되지 않은 경우
                patience_check += 1
                if patience_check >= patience_limit:
                    print("Early stopping")
                    break
            else: # loss가 개선된 경우 계속 진행
                best_loss = val_loss
                patience_check = 0
                
            ## 최고 val acc 모델 갱신
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            
            # time
            sec = time.time()-midel_time # 종료 - 시작 (걸린 시간)
            times = str(datetime.timedelta(seconds=sec)) # 걸린시간 보기좋게 바꾸기
            short = times.split(".")[0] # 초 단위 까지만
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2} || "
                f"f1 score : {valid_f1_score.get_score :4.2} || epoch time {short}"
            )

            wandb.log({"valid acc": val_acc, "valid loss": val_loss, 'valid_f1_score' : valid_f1_score.get_score},step = epoch)
            
            # early stop
            if val_loss > best_loss: # loss가 개선되지 않은 경우
                patience_check += 1
                if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
                    print("Early stopping")
                    break
                    
            else: # loss가 개선된 경우 계속 진행
                best_loss = val_loss
                patience_check = 0
                best_cm=valid_f1_score.get_cm
            print('early stopping patience', patience_check)
            print()
    # total time
    sec = time.time()- start_time # 종료 - 시작 (걸린 시간)
    times = str(datetime.timedelta(seconds=sec)) # 걸린시간 보기좋게 바꾸기
    short = times.split(".")[0] # 초 단위 까지만
    print(f'Total time : {short}')
    print(best_cm)
    wandb.finish()
    
    model_type = args.model_type
    # ---- making submission or inference ----
    if args.inference_make:
        test_dir = '/opt/ml/input/data/eval'
        
        if model_type == 'MaskBase':
            inference(test_dir, save_dir, save_dir, args) # model_dir -> load_model(saved_model 
        elif model_type == 'Mask':
            mask_inference(test_dir, save_dir, save_dir, args) # model_dir -> load_model(saved_model 
        elif model_type == 'Gender':
            gender_inference(test_dir, save_dir, save_dir, args) # model_dir -> load_model(saved_model 
        elif model_type == 'Age':
            age_inference(test_dir, save_dir, save_dir, args) # model_dir -> load_model(saved_model 
        else:
            print('inference 파일 생성 에러')
        
#     if args.submission_make:
#         submission(model, save_dir=save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=111, help='random seed (default: 111)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskPreprocessDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: YoonpyoAugmentation)')
    parser.add_argument("--resize", nargs="+", type=tuple, default=(224,224), help='resize size for image when training (default 224, 224)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--valid_batch_size', type=int, default=32, help='input batch size for validing (default: 32)')
    parser.add_argument('--model', type=str, default='EfficientNetB3', help='model type (default: EfficientNetB3)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 1e-5)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
#     parser.add_argument('--freeze', type=bool, default=False, help='model freeze (default: False)')
    parser.add_argument('--patience_limit', type=int, default=3, help='early stopping patience_limit (default: 3)')
    parser.add_argument('--exp_name', type=str, default='mask', help='wandb exp name (default: exp)')
    parser.add_argument('--inference_make', type=bool, default=True, help='inference make info (default : False)')
    parser.add_argument('--outlier_remove', type=bool, default=False, help='remove outlier (default : False)')
    parser.add_argument('--model_type', type=str, default='MaskBase', help = 'Mask or Gender or Age or MaskBase')
    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)