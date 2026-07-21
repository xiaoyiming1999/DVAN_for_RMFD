import torch
import torch.nn as nn

class CBAM(nn.Module):
    """
    Simple Neural Network having 4 Convolution
    and 1 FC layers with Bayesian layers.
    """

    def   __init__(self, num_cls,):
        super(CBAM, self).__init__()

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

        self.l2_CA = nn.Sequential(
            nn.Linear(32, 16, ),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32, ),
        )
        self.l2_PA = nn.Conv1d(1, 1, kernel_size=15, stride=1, padding=7)

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.l3_CA = nn.Sequential(
            nn.Linear(64, 32, ),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64, ),
        )

        self.l3_PA = nn.Conv1d(1, 1, kernel_size=15, stride=1, padding=7, )

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))

        self.l4_CA = nn.Sequential(
            nn.Linear(128, 64, ),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128, ),
        )

        self.l4_PA = nn.Conv1d(1, 1, kernel_size=15, stride=1, padding=7, )

        self.AdaptiveAvgpool = nn.AdaptiveAvgPool1d(4)

        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4, num_cls))

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)
        B, C, W = x.shape
        x_cha_act = x.mean(dim=2)
        attnc1_logit = self.l2_CA(x_cha_act)
        attnc1 = self.sigmoid(attnc1_logit)
        x = torch.mul(x, attnc1.view(B, C, 1))
        x_pos_act = x.mean(dim=1)
        x_pos_act = torch.unsqueeze(x_pos_act, dim=1)
        attnp1_logit = self.l2_PA(x_pos_act)
        attnp1 = self.sigmoid(attnp1_logit)
        x = torch.mul(x, attnp1.view(B, 1, W))

        x = self.layer3(x)
        B, C, W = x.shape
        x_cha_act = x.mean(dim=2)
        attnc2_logit = self.l3_CA(x_cha_act)
        attnc2 = self.sigmoid(attnc2_logit)
        x = torch.mul(x, attnc2.view(B, C, 1))
        x_pos_act = x.mean(dim=1)
        x_pos_act = torch.unsqueeze(x_pos_act, dim=1)
        attnp2_logit = self.l3_PA(x_pos_act)
        attnp2 = self.sigmoid(attnp2_logit)
        x = torch.mul(x, attnp2.view(B, 1, W))

        x = self.layer4(x)
        B, C, W = x.shape
        x_cha_act = x.mean(dim=2)
        attnc3_logit = self.l4_CA(x_cha_act)
        attnc3 = self.sigmoid(attnc3_logit)
        x = torch.mul(x, attnc3.view(B, C, 1))
        x_pos_act = x.mean(dim=1)
        x_pos_act = torch.unsqueeze(x_pos_act, dim=1)
        attnp3_logit = self.l4_PA(x_pos_act)
        attnp3 = self.sigmoid(attnp3_logit)
        x = torch.mul(x, attnp3.view(B, 1, W))

        x = self.AdaptiveAvgpool(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)

        return x