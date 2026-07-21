import torch
import torch.nn as nn
from variational_layer import VariationalConv, VariatinalLinear

class VCBAM(nn.Module):
    """
    Simple Neural Network having 4 Convolution
    and 1 FC layers with Bayesian layers.
    """

    def   __init__(self, num_cls, priors, device=None):
        super(VCBAM, self).__init__()
        self.device = device
        self.priors = priors
        self.sigmoid = nn.Sigmoid()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=25, stride=2, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=15, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

        # Dual Variational Attention Mechanism in Layer 2
        # variational channel attention in Layer 2
        self.l2_VCA = nn.Sequential(
            VariatinalLinear(32, 16, device=device, priors=self.priors),
            nn.ReLU(inplace=True),
            VariatinalLinear(16, 32, device=device, priors=self.priors),
        )
        # variational position attention in Layer 2
        self.l2_VPA = VariationalConv(1, 1, kernel_size=15, device=device, stride=1, padding=7, priors=self.priors,
                                      use_flipout=True)

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        # Dual Variational Attention Mechanism in Layer 3
        # variational channel attention in Layer 3
        self.l3_VCA = nn.Sequential(
            VariatinalLinear(64, 32, device=device, priors=self.priors),
            nn.ReLU(inplace=True),
            VariatinalLinear(32, 64, device=device, priors=self.priors),
        )
        # variational position attention in Layer 3
        self.l3_VPA = VariationalConv(1, 1, kernel_size=15, device=device, stride=1, padding=7, priors=self.priors,
                                      use_flipout=True)

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))

        # Dual Variational Attention Mechanism in Layer 4
        # variational channel attention in Layer 4
        self.l4_VCA = nn.Sequential(
            VariatinalLinear(128, 64, device=device, priors=self.priors),
            nn.ReLU(inplace=True),
            VariatinalLinear(64, 128, device=device, priors=self.priors),
        )
        # variational position attention in Layer 4
        self.l4_VPA = VariationalConv(1, 1, kernel_size=15, device=device, stride=1, padding=7, priors=self.priors,
                                      use_flipout=True)

        self.AdaptiveAvgpool = nn.AdaptiveAvgPool1d(4)

        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4, num_cls))

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)
        B, C, W = x.shape
        x_cha_act = x.mean(dim=2)
        attnc1_logit = self.l2_VCA(x_cha_act)
        attnc1 = self.sigmoid(attnc1_logit)
        x = torch.mul(x, attnc1.view(B, C, 1))
        x_pos_act = x.mean(dim=1)
        x_pos_act = torch.unsqueeze(x_pos_act, dim=1)
        attnp1_logit = self.l2_VPA(x_pos_act)
        attnp1 = self.sigmoid(attnp1_logit)
        x = torch.mul(x, attnp1.view(B, 1, W))
        attn1_logit = torch.cat((attnc1_logit, attnp1_logit.reshape(B, W)), dim=1)

        x = self.layer3(x)
        B, C, W = x.shape
        x_cha_act = x.mean(dim=2)
        attnc2_logit = self.l3_VCA(x_cha_act)
        attnc2 = self.sigmoid(attnc2_logit)
        x = torch.mul(x, attnc2.view(B, C, 1))
        x_pos_act = x.mean(dim=1)
        x_pos_act = torch.unsqueeze(x_pos_act, dim=1)
        attnp2_logit = self.l3_VPA(x_pos_act)
        attnp2 = self.sigmoid(attnp2_logit)
        x = torch.mul(x, attnp2.view(B, 1, W))
        attn2_logit = torch.cat((attnc2_logit, attnp2_logit.reshape(B, W)), dim=1)

        x = self.layer4(x)
        B, C, W = x.shape
        x_cha_act = x.mean(dim=2)
        attnc3_logit = self.l4_VCA(x_cha_act)
        attnc3 = self.sigmoid(attnc3_logit)
        x = torch.mul(x, attnc3.view(B, C, 1))
        x_pos_act = x.mean(dim=1)
        x_pos_act = torch.unsqueeze(x_pos_act, dim=1)
        attnp3_logit = self.l4_VPA(x_pos_act)
        attnp3 = self.sigmoid(attnp3_logit)
        x = torch.mul(x, attnp3.view(B, 1, W))
        attn3_logit = torch.cat((attnc3_logit, attnp3_logit.reshape(B, W)), dim=1)

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






