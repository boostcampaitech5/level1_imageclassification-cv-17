import torch
import torch.nn as nn
import torch.nn.functional as F
import loss


    

# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    '''
    FocalLoss : 이진 분류와 다중 클래스 분류에서 불균형한 데이터셋에서 사용할 수 있는 손실 함수
    nn.Module을 상속받아 FocalLoss 클래스를 정의합니다
    - __init__
        weight : 각 클래스의 손실 가중치
        gamma : 샘플링 된 클래스의 가중치를 증가시키는 데 사용됨
        reduction : 반환된 손실값을 어떻게 계산할지를 나타냄

    - forward
        주어진 input_tensor (예측된 분류기의 출력)와 target_tensor (실제 분류 레이블)를 기반으로 Focal Loss를 계산
        log_prob :  input_tensor의 로그 확률을 계산하고
        prob     :  log_prob의 지수 값을 계산합니다.
        nll_loss :  Focal Loss를 계산합니다. 이때, Focal Loss는 교차 엔트로피 손실의 가중치를 변경한 것입니다.
    '''
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )

class FocalLoss_ce(nn.Module):
    def __init__(self, gamma=2):
        nn.Module.__init__(self)
        self.gamma = gamma

    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss()(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        
        return focal_loss
    

class LabelSmoothingLoss(nn.Module):
    '''
    Label Smoothing Loss는 모델이 학습할 때, 라벨에 대한 예측값이 극단적으로 높아지는 것을 방지하기 위해 사용됩니다.
    이 방법은 정답 라벨에 대한 확신을 줄이고, 모델이 더욱 일반화된 결정을 내릴 수 있도록 합니다
    Label smoothing을 적용하면 모델이 완전히 확신할 수 있는 클래스가 아니더라도 모든 클래스에 일정한 확률을 할당합니다.
    이를 통해 모델은 더욱 일반적인 패턴을 학습하게 됩니다.

    - __init__ 

    - forward
    pred는 : 벨 스무딩이 적용된 모델의 예측값인 pred는 log_softmax 함수를 통해 계산됨
    true_dist: 정답 분포, 실제 라벨 분포를 나타내는 텐서 
               true_dist에 target 값에 해당하는 위치에 self.confidence 값을 넣고
               나머지 위치에 self.smoothing / (self.cls - 1) 값을 넣어 생성됩니다.
               이후 true_dist와 pred의 원소별 곱의 합을 평균하여 반환합니다.

    '''
    def __init__(self, classes=18, smoothing=0.01, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class CrossEntropyLossWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    '''
    F1 score에 기반한 loss function을 구현한 클래스
    - __init__
        epsilon :  0으로 나누는 오류를 방지하기 위한 작은 값을 의미
                   F1 score의 분모가 0이 되는 것을 방지
    - forward
        assert : y_pred와 y_true가 각각 2차원과 1차원이라는 것을 assert
        y_true : one-hot encoding을 수행
        y_pred는 : softmax를 통해 확률값으로 변환
        tp : True Positive (TP) 값을 계산합니다. 이 값은 각 클래스마다 계산되어 1차원 텐서로 표현됩니다.
        tn : True Negative (TN) 값을 계산합니다. 이 값은 각 클래스마다 계산되어 1차원 텐서로 표현됩니다.
        fp : False Positive (FP)값을 계산합니다. 이 값은 각 클래스마다 계산되어 1차원 텐서로 표현됩니다.
        fn : False Negative (FN)값을 계산합니다. 이 값은 각 클래스마다 계산되어 1차원 텐서로 표현됩니다.
        precision : 정밀도(precision) = TP/(TP+FP)
        recall : 재현율(recall) = TP/(TP+FN)
        f1 : precision와 recall로 F1 score를 계산함
        return F1 score의 평균값을 1에서 빼주어 loss 값을 계산함
    '''
    def __init__(self, classes=18, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
#         if y_pred.ndim == 2:
#             y_pred = y_pred[1]
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()

class CrossEntropyLossWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    

# class WeightedCrossEntropy(nn.Module):
#     def __init__(self, weight=None):
#         """
#         - weight (torch.Tensor 또는 None): 클래스별 가중치를 지정하는 1D Tensor. 만약 None이면 가중치를 일정하게 적용합니다.
#         - reduction (str): 손실 함수의 감소 방식을 지정하는 문자열입니다. 'mean' (기본값)이면 평균을 구하고, 'sum'이면 합을 구합니다.
#         """
#         super().__init__()
#         weight = torch.ones(18)
#         weight[[14, 8, 2, 7]] += 1
#         self.weight = torch.FloatTensor(weight).cuda()

#     def forward(self, input, target):
        
#         loss=nn.CrossEntropyLoss(weight=self.weight)

#         return loss


_criterion_entrypoints = {
    'cross_entropy': nn.CrossEntropyLoss,
    'binaryCE': nn.BCELoss(),
    'focal': FocalLoss,
    'focal_ce': FocalLoss_ce,
    'label_smoothing': LabelSmoothingLoss,
    'cross_labelsmooth': CrossEntropyLossWithLabelSmoothing,
    'f1': F1Loss,
}


def criterion_entrypoint(criterion_name):
    '''
    주어진 손실 함수 이름에 해당하는 생성 함수를 반환
    '''
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    '''
    주어진 이름이 유효한 손실 함수 이름인지 확인
    '''
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    '''
    손실 함수의 이름과 추가 인자를 받아서, 
    criterion_entrypoint() 함수를 사용하여 해당 손실 함수를 생성합니다
    이때, 손실 함수 이름이 유효하지 않을 경우, 에러 메시지를 출력합니다.
    '''
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion
