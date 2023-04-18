from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
class get_F1_Score:
    def __init__(self):
        self.reset()
    def reset(self):
        self.y_pred = []
        self.y_true = []
    def update(self,prediction,target):
        self.y_true.extend(target.cpu())
        self.y_pred.extend(prediction.cpu())
    @property
    def get_score(self):
        f1 = f1_score(self.y_true,self.y_pred,average = 'weighted')
        return f1
    @property
    def get_cm(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        return cm