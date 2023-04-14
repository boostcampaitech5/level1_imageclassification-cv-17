from sklearn.metrics import f1_score
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