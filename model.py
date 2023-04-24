import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# https://pytorch.org/vision/0.10/models.html
from torchvision.models import vgg16, vgg16_bn, vgg19, vgg19_bn
from torchvision.models import resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
from torchvision.models import densenet121, densenet169, densenet161, densenet201
from efficientnet_pytorch import EfficientNet
import timm

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

### ResNet ###
class ResNet34(nn.Module):
    '''
    마지막 수정자 : 김용우
    '''
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        self.model = resnet34(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, num_classes) # 18

    def forward(self, x):
        x = self.model(x)
        return x
    
class ResNet34_init(nn.Module):
    '''
    마지막 수정자 : 김용우
    '''
    def __init__(self, num_classes):
        super(ResNet34_init, self).__init__()
        self.model = resnet34(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, num_classes) # 18
        initialize_weights(self.model.fc)

    def forward(self, x):
        x = self.model(x)
        return x
    
class ResNet50(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.model = resnet50(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, num_classes) # 18

    def forward(self, x):
        x = self.model(x)
        return x
    

    
class ResNet101(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(ResNet101, self).__init__()
        self.model = resnet101(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, num_classes) # 18

    def forward(self, x):
        x = self.model(x)
        return x
    
class ResNet152(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(ResNet152, self).__init__()
        self.model = resnet152(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, num_classes) # 18

    def forward(self, x):
        x = self.model(x)
        return x

### ResNext ###
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

### wide_resnet ###
class WideResNet50(nn.Module):
    '''
    생성자 : 박승희
    수정자 : 박승희
    '''
    def __init__(self, num_classes):
        super(WideResNet50, self).__init__()
        self.model = wide_resnet50_2(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, num_classes) # 18
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class WideResNet101(nn.Module):
    '''
    생성자 : 박승희
    수정자 : 박승희
    '''
    def __init__(self, num_classes):
        super(WideResNet101, self).__init__()
        self.model = wide_resnet101_2(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, num_classes) # 18
        
    def forward(self, x):
        x = self.model(x)
        return x
    
### densenet ### 
class DenseNet121(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()
        self.model = densenet121(pretrained=True)
        self.num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.num_ftrs, num_classes) # 18
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class DenseNet121_init(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(DenseNet121_init, self).__init__()
        self.model = densenet121(pretrained=True)
        self.num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.num_ftrs, num_classes) # 18
        initialize_weights(self.model.classifier)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class DenseNet161(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(densenet161, self).__init__()
        self.model = densenet161(pretrained=True)
        self.num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.num_ftrs, num_classes) # 18
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class DenseNet161_init(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(DenseNet161_init, self).__init__()
        self.model = densenet161(pretrained=True)
        self.num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.num_ftrs, num_classes) # 18
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class DenseNet169(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(DenseNet169, self).__init__()
        self.model = densenet169(pretrained=True)
        self.num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.num_ftrs, num_classes) # 18
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class DenseNet201(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(DenseNet201, self).__init__()
        self.model = densenet201(pretrained=True)
        self.num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.num_ftrs, num_classes) # 18
        
    def forward(self, x):
        x = self.model(x)
        return x
    
### EfficientNet ###
class EfficientNetB3_init(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(EfficientNetB3_init, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
        self.num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(self.num_ftrs, num_classes)
        initialize_weights(self.model._fc)

    def forward(self, x):
        x = self.model(x)
        return x
    
class EfficientNetB3_xavier(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB3_xavier, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
        self.num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(self.num_ftrs, num_classes)
        nn.init.xavier_uniform_(self.model._fc.weight)

    def forward(self, x):
        x = self.model(x)
        return x
    
class EfficientNetB3_xavier(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB3_xavier, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
        self.num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(self.num_ftrs, num_classes)
        nn.init.xavier_uniform_(self.model._fc.weight)

    def forward(self, x):
        x = self.model(x)
        return x
    
class EfficientNetB3_init(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(EfficientNetB3_init, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
        self.num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(self.num_ftrs, num_classes)
        initialize_weights(self.model._fc)

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
    
class EfficientNetB4_init(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(EfficientNetB4_init, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        self.num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(self.num_ftrs, num_classes)
        nn.init.xavier_uniform_(self.model._fc.weight)

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
    
class EfficientNetB6(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(EfficientNetB6, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b6')
        self.num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(self.num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
class EfficientNetB7(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(EfficientNetB7, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7')
        self.num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(self.num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
#####  timm model  #######
class Vit_p8(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(Vit_p8, self).__init__()
        self.model = timm.create_model('vit_base_patch8_224', pretrained=True)
        self.num_ftrs = self.model.head.in_features
        self.model.head = nn.Linear(self.num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
class Vit_p16(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(Vit_p16, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.num_ftrs = self.model.head.in_features
        self.model.head = nn.Linear(self.num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
class Vit_s_p16(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(Vit_s_p16, self).__init__()
        self.model = timm.create_model('vit_small_patch16_224', pretrained=True)
        self.num_ftrs = self.model.head.in_features
        self.model.head = nn.Linear(self.num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
class Vit_s_p32(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(Vit_s_p32, self).__init__()
        self.model = timm.create_model('vit_small_patch32_224', pretrained=True)
        self.num_ftrs = self.model.head.in_features
        self.model.head = nn.Linear(self.num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class Swin_p4(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(Swin_p4, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.num_ftrs = self.model.head.in_features
        self.model.head = nn.Linear(self.num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
class Swin_p4_l(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(Swin_p4_l, self).__init__()
        self.model = timm.create_model('swin_large_patch4_window7_224', pretrained=True)
        self.num_ftrs = self.model.head.in_features
        self.model.head = nn.Linear(self.num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
class Swin_p4_s(nn.Module):
    '''
    생성자 : 박승희
    '''
    def __init__(self, num_classes):
        super(Swin_p4_s, self).__init__()
        self.model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
        self.num_ftrs = self.model.head.in_features
        self.model.head = nn.Linear(self.num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x