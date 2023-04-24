# Pstage_01_image_classification

## <span style='color:black;background-color:#fff5b1'>Getting Started</spam>    
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

<br/>

## <span style='color:black;background-color:#fff5b1'>파일별 세부 설명</spam>
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
    EfficientNetB3 : from efficientnet_pytorch, no freeze
    EfficientNetB4 : from efficientnet_pytorch, no freeze
    EfficientNetB5 : from efficientnet_pytorch, no freeze
    ResNet34       : from torchvision.models, no freeze
    ResNext50      : from torchvision.models, no freeze
    ResNext101     : from torchvision.models, no freeze
    Vgg19          : from torchvision.models, no freeze
    

### train.py
dataset.py, loss.py의 함수를 import 해서 사용함
- function
    seed_everything : 랜덤 시드 설정, 해당 값에 대한 동일한 seed 값으로 다시 실행해도 동일한 결과가 나오도록 보장함
    get_lr : 현재 optimizer의 학습률(learning rate)을 반환하는 함수
    grid_image : 입력으로 받은 이미지들을 그리드 형태로 시각화하는 기능을 수행합니다
    increment_path : 경로명을 자동으로 증가시켜주는 함수
    train : data_dir(데이터 경로), model_dir(모델 경로), args(인자)를 받아와서 모델을 학습하는 함수  

<br/>

## <span style='color:black;background-color:#fff5b1'>프로젝트 진행 과정</spam>
- ### **Time-Line**
    <img src='https://ifh.cc/g/4f99yJ.png' width="600" height="400"/>
----
    협업 방식
    - [1] 각 날짜마다 어떤 실험을 했고 어떤 결과가 있었는지 정리하여 Notion 페이지에 작성
    - [2] 실험을 할 때 Wandb에 누가했는지 기록하도록 하여 서로의 실험 결과를 쉽게 비교 
    - [3] 각자 아침마다 자신이 진행한 실험과 결과에 대해서 이야기해보는 시간을 가짐
    - [4] 공동 Github에 메인 branch 이외 팀원 별 하나의 branch를 생성하고 각자 실험해본 뒤, 팀원들과 상의 후 Merge
    - [5] 소스코드와 일정을 관리하는 PM을 정하여 팀원들이 원활하게 소통할 수 있도록 도움

<br/>

## <span style='color:black;background-color:#fff5b1'>최종 모델의 아키텍처 및 하이퍼파라미터</spam>

- 2개(mask+gender , age)로 나누어 학습을 진행해주었다.

  |병렬 모델|model|dataset|augmentation|optimizer|lr|criterion|epochs|
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  |Mask+Gender|EfficientNetB3_init|MaskGenderDataset|bestAugmentation|AdamW|le-5|cross_entropy|30(early stop cnt=3)|
  |Age|EfficientNetB3|AgeDataset|bestAugmentation|Adam|le-5|cross_entropy|20(early stop cnt=3)|
<br/>

## <span style='color:black;background-color:#fff5b1'>최종 모델</spam>
- ### **모델 구조**
    <img src="https://ifh.cc/g/3ZDJBs.png">
<br/>

- ### **모델 성능**
    <img src="https://ifh.cc/g/2jHcpt.png">

