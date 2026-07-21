import torch
import torch.nn as nn
from variational_layer import VariationalConv

class VSA(nn.Module):
    """
    Simple Neural Network having 4 Convolution
    and 1 FC layers with Bayesian layers.
    """

    def __init__(self, num_cls, priors, device=None):
        super(VSA, self).__init__()
        self.device = device
        self.priors = priors

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=25, stride=2, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=15, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=8),
        )

        self.l2_VSA_Q = VariationalConv(32, 32, kernel_size=1, stride=1, padding=0, device=device, priors=self.priors)
        self.l2_VSA_K = VariationalConv(32, 32, kernel_size=1, stride=1, padding=0, device=device, priors=self.priors)
        self.l2_VSA_V = nn.Conv1d(32, 32, kernel_size=1, padding=0)

        self.l2_Res = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=1),
            nn.Linear(500, 60)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.l3_VSA_Q = VariationalConv(64, 64, kernel_size=1, stride=1, padding=0, device=device, priors=self.priors)
        self.l3_VSA_K = VariationalConv(64, 64, kernel_size=1, stride=1, padding=0, device=device, priors=self.priors)
        self.l3_VSA_V = nn.Conv1d(64, 64, kernel_size=1, padding=0)

        self.l3_Res = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1),
            nn.Linear(60, 56)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))

        self.l4_VSA_Q = VariationalConv(128, 128, kernel_size=1, stride=1, padding=0, device=device, priors=self.priors)
        self.l4_VSA_K = VariationalConv(128, 128, kernel_size=1, stride=1, padding=0, device=device, priors=self.priors)
        self.l4_VSA_V = nn.Conv1d(128, 128, kernel_size=1, padding=0)

        self.l4_Res = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.Linear(56, 52)
        )

        self.AdaptiveAvgpool = nn.AdaptiveAvgPool1d(4)

        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4, num_cls))

    def forward(self, x):

        B, _, _ = x.shape
        x = self.layer1(x)

        x1 = self.layer2(x)
        x2 = self.l2_Res(x)
        Q = self.l2_VSA_Q(x1)
        K = self.l2_VSA_K(x1)
        V = self.l2_VSA_V(x1)
        attn1_logit = Q.transpose(-2, -1) @ K
        attn = attn1_logit.softmax(dim=-1)
        x = V @ attn + x2
        attn1_logit = attn1_logit.reshape(B, -1)

        x1 = self.layer3(x)
        x2 = self.l3_Res(x)
        Q = self.l3_VSA_Q(x1)
        K = self.l3_VSA_K(x1)
        V = self.l3_VSA_V(x1)
        attn2_logit = Q.transpose(-2, -1) @ K
        attn = attn2_logit.softmax(dim=-1)
        x = V @ attn + x2
        attn2_logit = attn2_logit.reshape(B, -1)

        x1 = self.layer4(x)
        x2 = self.l4_Res(x)
        Q = self.l4_VSA_Q(x1)
        K = self.l4_VSA_K(x1)
        V = self.l4_VSA_V(x1)
        attn3_logit = Q.transpose(-2, -1) @ K
        attn = attn3_logit.softmax(dim=-1)
        x = V @ attn + x2
        attn3_logit = attn3_logit.reshape(B, -1)

        x = self.AdaptiveAvgpool(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)

        # attn concat
        attn_logits = torch.cat((attn1_logit, attn2_logit, attn3_logit), dim=1)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return x, attn_logits, kl

