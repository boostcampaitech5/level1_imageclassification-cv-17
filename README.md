# pstage_01_image_classification

## Getting Started    
### Dependencies
- torch==1.7.1
- torchvision==0.8.2                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN={YOUR_TRAIN_IMG_DIR} SM_MODEL_DIR={YOUR_MODEL_SAVING_DIR} python train.py`

### Inference
- `SM_CHANNEL_EVAL={YOUR_EVAL_DIR} SM_CHANNEL_MODEL={YOUR_TRAINED_MODEL_DIR} SM_OUTPUT_DATA_DIR={YOUR_INFERENCE_OUTPUT_DIR} python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR={YOUR_GT_DIR} SM_OUTPUT_DATA_DIR={YOUR_INFERENCE_OUTPUT_DIR} python evaluation.py`


## 파일별 세부 설명
### dataset.py : 데이터셋 함수 및 클래스 정의
- function
    is_image_file(filename) : 이미지 파일 확인하는 함수

- class
    BaseAugmentation   : Resize, ToTensor, Normalize
    AddGaussianNoise   : 이미지에 노이즈 추가
    CustomAugmentation : CenterCrop((320, 256))
                         Resize()
                         ColorJitter(0.1, 0.1, 0.1, 0.1)
                         ToTensor()
                         Normalize()
                         AddGaussianNoise()
    MaskLabels         : 마스크 착용 여부 label
    GenderLabels       : 성별 label
    AgeLabels          : 나이 label
    MaskBaseDataset    : 마스크를 쓴 사람의 얼굴 이미지를 다루는 데이터셋을 구성
    MaskSplitByProfileDataset : MaskBaseDataset 클래스를 상속받은 클래스로,
                                이미지 데이터셋을 프로필(person)을 기준으로 train과 validation으로 나누는 기능을 구현
    TestDataset : Test 데이터셋 구성, transform: Resize, ToTensor(),Normalize


### inference.py : 
dataset.py의 함수를 import 해서 사용함
- function
    load_model : 저장된 모델 파일을 로드하여 PyTorch 모델 객체를 반환하는 함수 
    inference  : 모델을 사용하여 이미지를 추론하고, 추론 결과를 저장하는 함수

### loss.py
- class
    Focal Loss         : 이진 분류와 다중 클래스 분류에서 불균형한 데이터셋에서 사용할 수 있는 손실 함수 클래스
    LabelSmoothingLoss : Label Smoothing Loss를 구현한 pytorch 모듈
                         정답 라벨에 대한 확신을 줄이고, 모델이 더욱 일반화된 결정을 내릴 수 있도록 하는 클래스
    F1Loss :  F1 score에 기반한 loss function을 구현한 클래스

- function
    criterion_entrypoint : 주어진 손실 함수 이름에 해당하는 생성 함수를 반환
    is_criterion : 주어진 이름이 유효한 손실 함수 이름인지 확인
    create_criterion : nn.CrossEntropyLoss, FocalLoss, LabelSmoothingLossm F1Loss로 손실 함수를 생성

### model.py
- class
    BaseModel : 기본 제공 baseline 모델
                conv1-relu
                conv2-relu-maxpool
                conv3-relu-maxpool-dropout
                avgpool
                fc
    MyModel : 커스텀 모델

### train.py
dataset.py, loss.py의 함수를 import 해서 사용함
- function
    seed_everything : 랜덤 시드 설정, 해당 값에 대한 동일한 seed 값으로 다시 실행해도 동일한 결과가 나오도록 보장함
    get_lr : 현재 optimizer의 학습률(learning rate)을 반환하는 함수
    grid_image : 입력으로 받은 이미지들을 그리드 형태로 시각화하는 기능을 수행합니다
    increment_path : 경로명을 자동으로 증가시켜주는 함수
    train : data_dir(데이터 경로), model_dir(모델 경로), args(인자)를 받아와서 모델을 학습하는 함수