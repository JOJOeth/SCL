import torch.nn as nn
import torchvision
import torch
from scl.modules.resnet_hacks import modify_resnet_model
from scl.modules.identity import Identity


class SCL(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, train, encoder, projection_dim, n_features):
        super(SCL, self).__init__()
        self.train = train
        self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()
        # self.encoder.classifier = Identity()
        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        if self.train == False:
            h_i = self.encoder(x_i.type(torch.FloatTensor).cuda())
            h_j = self.encoder(x_j.type(torch.FloatTensor).cuda())
            del x_i, x_j
            return h_i, h_j
        else:
            z_i = self.projector(self.encoder(x_i.type(torch.FloatTensor).cuda()).type(torch.FloatTensor).cuda())
            z_j = self.projector(self.encoder(x_j.type(torch.FloatTensor).cuda()).type(torch.FloatTensor).cuda())
            del x_i, x_j
            return z_i, z_j
class SCLL(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, projection_dim, n_features):
        super(SCLL, self).__init__()
        self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()
        # self.encoder.classifier = Identity()
        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i.type(torch.FloatTensor).cuda())
        h_j = self.encoder(x_j.type(torch.FloatTensor).cuda())
        # z_i = self.projector(self.encoder(x_i.type(torch.FloatTensor).cuda()).type(torch.FloatTensor).cuda())
        # z_j = self.projector(self.encoder(x_j.type(torch.FloatTensor).cuda()).type(torch.FloatTensor).cuda())
        del x_i, x_j
        return h_i, h_j