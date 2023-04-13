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


### dataset.py : 데이터셋 함수 및 클래스 정의
- function
    is_image_file(filename) : 이미지 파일 확인하는 함수

- class
    BaseAugmentation : Resize, ToTensor, Normalize
    AddGaussianNoise : 이미지에 노이즈 추가
    CustomAugmentation : CenterCrop((320, 256))
                         Resize()
                         ColorJitter(0.1, 0.1, 0.1, 0.1)
                         ToTensor()
                         Normalize()
                         AddGaussianNoise()
    MaskLabels : 마스크 착용 여부 label
    GenderLabels : 성별 label
    AgeLabels : 나이 label
    MaskBaseDataset : 마스크를 쓴 사람의 얼굴 이미지를 다루는 데이터셋을 구성
    MaskSplitByProfileDataset : MaskBaseDataset 클래스를 상속받은 클래스로,
                                 이미지 데이터셋을 프로필(person)을 기준으로 train과 validation으로 나누는 기능을 구현
    TestDataset : Test 데이터셋 구성, transform: Resize, ToTensor(),Normalize


### inference.py : 
- function
    load_model : 저장된 모델 파일을 로드하여 PyTorch 모델 객체를 반환하는 함수 inference :  모델을 사용하여 이미지를 추론하고, 추론 결과를 저장하는 함수


    