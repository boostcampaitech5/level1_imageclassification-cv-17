import argparse
import multiprocessing
import os
from importlib import import_module
import time
import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

from dataset import TestDataset, MaskBaseDataset, MaskDataset, GenderDataset, AgeDataset, MaskGenderDataset


def load_model(saved_model, num_classes, device, import_model):
    '''
    저장된 모델 파일을 로드하여 PyTorch 모델 객체를 반환하는 함수인 load_model()입니다.
    load_model() :  세 개의 매개변수 saved_model, num_classes, device를 받습니다.
    saved_model  : 로드할 모델이 저장된 경로를 나타냅니다.
    num_classes  :  모델의 클래스 수를 나타내며, 이 매개변수는 모델 객체를 생성할 때 사용됩니다. 
    device       :  모델이 실행될 디바이스(CPU 또는 GPU)를 나타냅니다.
    - 변수 설명
    model_cls    : 모듈 이름이 model인 모듈에서 args.model로 지정된 클래스를 가져옵니다. 이 코드에서는 getattr 함수를 사용하여 동적으로 모듈에서 클래스를 가져옵니다.
    model        : model_cls에서 생성된 모델 객체입니다. num_classes를 인자로 전달하여 생성자에서 모델의 클래스 수를 설정합니다.
    model_path   : saved_model 경로와 best.pth 파일 이름을 결합하여 모델 파일의 전체 경로를 지정합니다. 
                   이후 torch.load 함수를 사용하여 모델 파일을 읽어들이고, map_location 매개변수를 사용하여 모델이 실행될 디바이스를 설정합니다. 
                   마지막으로, 함수는 로드된 모델 객체를 반환합니다.
    '''
    model_cls = getattr(import_module("model"), import_model)
    model = model_cls(
        num_classes=num_classes
    )

    '''
    주석 처리된 코드는 .tar.gz 파일을 압축 해제하는 코드입니다.
    압축 파일이 아닌 경우 주석 처리하고 사용하지 않아도 됩니다.
    '''
    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    - 클래스 설명
    모델을 사용하여 이미지를 추론하고, 추론 결과를 저장하는 함수
    @torch.no_grad() : PyTorch의 기능 중 하나인 gradient 계산을 하지 않도록 하는 함수 데코레이터입니다. 
                       이것은 모델 추론 시 메모리 사용량을 줄이고 추론 속도를 높이는 데 도움이 됩니다.
    
    - 클래스 인자
    data_dir   : 테스트 데이터의 디렉토리 경로
    model_dir  : 학습된 모델의 디렉토리 경로
    output_dir :  추론 결과를 저장할 디렉토리 경로
    args       :  다른 인자들을 받아들이기 위한 argparse.ArgumentParser() 인스턴스를 이용한 변수들임
    - 함수 변수 및 내부 구현 설명
    use_cuda : GPU를 사용할 수 있는지를 확인
    device :  사용 가능하다면 GPU를 사용하고, 그렇지 않으면 CPU를 사용합니다.
    num_classes :  MaskBaseDataset 클래스의 클래스 변수인 num_classes의 값을 가져와서 정의합니다. 
    model : load_model 함수를 사용하여 모델을 불러옵니다. 모델을 불러온 후 model.eval() 함수를 사용하여 모델을 평가 모드로 전환합니다.
    img_root :  data_dir와 'images'를 결합하여 이미지가 저장된 디렉토리 경로를 설정합니다.
    info_path : data_dir와 'info.csv'를 결합하여 이미지 정보가 저장된 csv 파일의 경로를 설정합니다. 
    info : pd.read_csv 함수를 사용하여 info_path(csv 파일)에서 정보를 읽어옵니다.
    img_paths : img_root와 각 이미지 파일의 이름을 결합하여 이미지 파일들의 경로 리스트를 만듭니다. 
    dataset : img_paths를 이용하여 TestDataset 클래스의 인스턴스를 생성합니다. 
    loader :  torch.utils.data.DataLoader 함수를 사용하여 데이터를 로드합니다
    preds : 추론 결과를 저장할 리스트를 생성합니다. 
    torch.no_grad() 블록을 사용하여 gradient 계산을 하지 않도록 설정합니다. 
    loader에서 데이터를 읽어와 모델을 이용하여 추론을 수행합니다. 이후 추론 결과를 저장합니다.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18 or 3 or 2
    import_model = args.model
    model = load_model(model_dir, num_classes, device, import_model).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
#         num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")



'''
- inference.py 실행 부분
스크립트 파일을 실행할 때, argparse 모듈을 사용하여 명령행 인수를 구문 분석하고,
생성할 디렉토리를 만든 다음 inference() 함수를 호출합니다.
    - 구현
    argparse 모듈을 사용하여 명령행 인수를 구문 분석합니다
    data_dir, model_dir, output_dir 변수를 선언하고,
    이 변수들에는 각각 args.data_dir, args.model_dir, args.output_dir 값을 할당합니다.
    그리고나서 os.makedirs() 함수를 사용하여 output_dir에 해당하는 디렉토리를 생성합니다.
    생성할 디렉토리가 이미 존재하면 새로 생성하지 않고 그대로 유지합니다.(os.makedirs(output_dir, exist_ok=True))
    마지막으로 inference() 함수를 호출하고, data_dir, model_dir, output_dir, args를 인자로 전달합니다.
'''


@torch.no_grad()
def mask_inference(data_dir, model_dir, output_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskDataset.num_classes  # 3
    import_model = args.model
    model = load_model(model_dir, num_classes, device, import_model).to(device) # load_model(saved_model, num_classes, device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
#         num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(tqdm(loader)):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Mask Inference Done! Mask Inference result saved at {save_path}")
    
def gender_inference(data_dir, model_dir, output_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = GenderDataset.num_classes  # 2
    import_model = args.model
    model = load_model(model_dir, num_classes, device, import_model).to(device) # load_model(saved_model, num_classes, device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
#         num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(tqdm(loader)):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Gender Inference Done! Gender Inference result saved at {save_path}")
    
def age_inference(data_dir, model_dir, output_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = AgeDataset.num_classes  # 3
    import_model = args.model
    model = load_model(model_dir, num_classes, device, import_model).to(device) # load_model(saved_model, num_classes, device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
#         num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in tqdm(enumerate(loader)):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Age Inference Done! Age Inference result saved at {save_path}")
    
def maskgender_inference(data_dir, model_dir, output_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    num_classes = MaskGenderDataset.num_classes  # 6
    import_model = args.model
    model = load_model(model_dir, num_classes, device, import_model).to(device) # load_model(saved_model, num_classes, device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
#         num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(tqdm(loader)):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Mask&Gender Inference Done! Mask Inference result saved at {save_path}")

    
def combine_inference(data_dir, output_dir): # Mask, Gender, Age inference make
    data_dir = '../input/data/eval'
    info_path = pd.read_csv(os.path.join(data_dir, 'info.csv'))
    info = pd.read_csv(info_path)

    # 비교할 두 개의 csv 파일 경로 설정
    csv_file1 = '' #비교할 csv
    csv_file2 = '' 
    csv_file2 = ''

    ans=0
    all_predictions=[]
    with open(csv_file1, 'r') as file1, open(csv_file2, 'r') as file2, open(csv_file3, 'r') as file3:
        reader1 = csv.reader(file1)
        reader2 = csv.reader(file2)
        reader3 = csv.reader(file3)

        for row1, row2,row3 in zip(reader1, reader2,reader3):
            if row1[1]!='ans' and row2[1]!='ans' and row3[1]!='ans':
                ans=6*int(row1[1])+3*int(row2[1])+int(row3[1])
                all_predictions.append(ans)

    info['ans'] = all_predictions
    info.to_csv(os.path.join(output_dir, 'multi_output.csv'), index=False)
    print('test munti inference is done!')

def combine_inference_2(data_dir, output_dir): # Mask+Gender, Age inference make
    data_dir = '../input/data/eval'
    info_path = pd.read_csv(os.path.join(data_dir, 'info.csv'))
    info = pd.read_csv(info_path)

    # 비교할 두 개의 csv 파일 경로 설정
    csv_file1 = '' #MaskGender
    csv_file2 = '' 

    ans=0
    all_predictions=[]
    with open(csv_file1, 'r') as file1, open(csv_file2, 'r') as file2:
        reader1 = csv.reader(file1)
        reader2 = csv.reader(file2)

        for row1, row2 in zip(reader1, reader2):
            if row1[1]!='ans' and row2[1]!='ans':
                ans=int(row1[1])+int(row2[1])
                all_predictions.append(ans)

    info['ans'] = all_predictions
    info.to_csv(os.path.join(output_dir, 'multi_output.csv'), index=False)
    print('test multi inference is done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories 
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', nargs="+", type=tuple, default=(512, 384), help='resize size for image when you trained (default: (512, 384))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    parser.add_argument('--model_type', type=str, default='MaskBase', help = 'Mask or Gender or Age or MaskBase')

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir
    model_type = args.model_type

    os.makedirs(output_dir, exist_ok=True)

    if model_type == 'MaskBase':
        inference(data_dir, model_dir, output_dir, args) # model_dir -> load_model(saved_model 
    elif model_type == 'Mask':
        mask_inference(data_dir, model_dir, output_dir, args) # model_dir -> load_model(saved_model 
    elif model_type == 'Gender':
        gender_inference(data_dir, model_dir, output_dir, args) # model_dir -> load_model(saved_model 
    elif model_type == 'Age':
        age_inference(data_dir, model_dir, output_dir, args) # model_dir -> load_model(saved_model 
    elif model_type == 'MaskGender':
        maskgender_inference(data_dir, model_dir, output_dir, args) # model_dir -> load_model(saved_model 

    else:
        print('inference 파일 생성 에러')
