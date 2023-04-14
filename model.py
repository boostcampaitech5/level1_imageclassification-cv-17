import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnext50_32x4d, resnext101_32x8d, resnext101_64x4d, vgg19
from efficientnet_pytorch import EfficientNet

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
        self.model.fc = nn.Linear(num_ftrs, num_classes) # 18

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
    
class ResNext101_4d(nn.Module):
    '''
    생성자 : 박승희
    수정자 : 박승희
    '''
    def __init__(self, num_classes):
        super(ResNext101_4d, self).__init__()
        self.model = resnext101_64x4d(pretrained=True)
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
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, num_classes) # 18
        
    def forward(self, x):
        x = self.model(x)
        return x