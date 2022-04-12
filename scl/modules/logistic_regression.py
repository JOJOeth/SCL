import torch.nn as nn
import torch
import os

class LogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        self.logistic_model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.logistic_model(x)

    def save(self, model_path, logistic_ps, epoch, model):
        out = os.path.join(model_path, logistic_ps + "_" + "logistic_checkpoint_{}.tar".format(epoch))

        torch.save(model.state_dict(), out)