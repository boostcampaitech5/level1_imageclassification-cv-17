import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnext50_32x4d, resnext101_32x8d, vgg19
from efficientnet_pytorch import EfficientNet
import torch.nn.init as init

def initialize_weights(model):
    """
    Xavier uniform 분포로 모든 weight 를 초기화합니다.
    더 많은 weight 초기화 방법은 다음 문서에서 참고해주세요. https://pytorch.org/docs/stable/nn.init.html
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)

class EfficientNetB3(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(EfficientNetB3, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
        self.num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(self.num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
class EfficientNetB4(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(EfficientNetB4, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        self.num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(self.num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
class EfficientNetB5(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(EfficientNetB5, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5')
        self.num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(self.num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
class ResNet34(nn.Module):
    '''
    생성자 : 김용우
    수정자 : 박승희
    pretrain된 resnet34를 가져와 
    '''
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        self.model = resnet34(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, num_classes) # 18
        self.model.fc = initialize_weights(self.model.fc)

    def forward(self, x):
        x = self.model(x)
        return x
    
class ResNext50(nn.Module):
    '''
    생성자 : 이다현
    수정자 : 박승희
    '''
    def __init__(self, num_classes):
        super(ResNext50, self).__init__()
        self.model = resnext50_32x4d(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, num_classes) # 18
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class ResNext101_8d(nn.Module):
    '''
    생성자 : 박승희
    수정자 : 박승희
    '''
    def __init__(self, num_classes):
        super(ResNext101_8d, self).__init__()
        self.model = resnext101_32x8d(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, num_classes) # 18
        
    def forward(self, x):
        x = self.model(x)
        return x

class Vgg19(nn.Module):
    '''
    생성자 : 이상민
    수정자 : 박승희
    '''
    def __init__(self, num_classes):
        super(Vgg19, self).__init__()
        self.model = vgg19(pretrained=True)
        self.num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(self.num_ftrs, num_classes) # 18
        
    def forward(self, x):
        x = self.model(x)
        return x