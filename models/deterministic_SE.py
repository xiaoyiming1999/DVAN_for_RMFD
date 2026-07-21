import torch
import torch.nn as nn

class SE(nn.Module):
    """
    Simple Neural Network having 4 Convolution
    and 1 FC layers with Bayesian layers.
    """

    def __init__(self, num_cls, ):
        super(SE, self).__init__()

        self.sigmoid = nn.Sigmoid()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=25, stride=2, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=15, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

        self.l2_CA = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32, ),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.l3_CA = nn.Sequential(
            nn.Linear(64, 32, ),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64, ),
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))

        self.l4_CA = nn.Sequential(
            nn.Linear(128, 64, ),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128, ),
        )

        self.AdaptiveAvgpool = nn.AdaptiveMaxPool1d(4)

        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4, num_cls))


    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)
        B, C, W = x.shape
        x1 = x.mean(dim=2)
        attn1_logit = self.l2_CA(x1)
        attn1 = self.sigmoid(attn1_logit)
        x = torch.mul(x, attn1.view(B, C, 1))

        x = self.layer3(x)
        B, C, W = x.shape
        x1 = x.mean(dim=2)
        attn2_logit = self.l3_CA(x1)
        attn2 = self.sigmoid(attn2_logit)
        x = torch.mul(x, attn2.view(B, C, 1))

        x = self.layer4(x)
        B, C, W = x.shape
        x1 = x.mean(dim=2)
        attn3_logit = self.l4_CA(x1)
        attn3 = self.sigmoid(attn3_logit)
        x = torch.mul(x, attn3.view(B, C, 1))

        x = self.AdaptiveAvgpool(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)

        return x