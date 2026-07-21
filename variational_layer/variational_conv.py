import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from variational_layer.KL_loss import calculate_kl


class VariationalConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, device, stride,
                 padding, dilation=1, priors=None, use_flipout=False):
        super(VariationalConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size,)
        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_flipout = use_flipout

        self.prior_mu = priors['prior_mu']
        self.prior_std = priors['prior_std']

        self.posterior_mu_initial = (0, 0.1)
        self.posterior_std_initial = (-3, 0.1)

        self.W_mu = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.W_std = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))

        self.bias_mu = Parameter(torch.Tensor(out_channels))
        self.bias_std = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_std.data.normal_(*self.posterior_std_initial)

        self.bias_mu.data.normal_(*self.posterior_mu_initial)
        self.bias_std.data.normal_(*self.posterior_std_initial)

    def forward(self, x):

        self.W_normalized_std = torch.log1p(torch.exp(self.W_std))
        w_eps = torch.empty(self.W_normalized_std.size()).normal_(0, 1).to(self.device)
        act_w = self.W_mu + w_eps * self.W_normalized_std

        self.bias_normalized_std = torch.log1p(torch.exp(self.bias_std))
        bias_eps = torch.empty(self.bias_normalized_std.size()).normal_(0, 1).to(self.device)
        act_bias = self.bias_mu + bias_eps * self.bias_normalized_std

        if self.use_flipout:
            # Flipout
            batch_size = x.size(0)
            # generate random sign
            r_in = torch.randint(0, 2, (batch_size, self.in_channels, 1)).float() * 2 - 1  # [B, C_in, 1]
            r_out = torch.randint(0, 2, (batch_size, self.out_channels, 1)).float() * 2 - 1  # [B, C_out, 1]
            r_in = r_in.to(self.device)
            r_out = r_out.to(self.device)
            x_perturbed = x * r_in
            output = F.conv1d(x_perturbed, act_w, act_bias, self.stride, self.padding, self.dilation, self.groups)
            output = output * r_out
        else:
            output = F.conv1d(x, act_w, act_bias, self.stride, self.padding, self.dilation, self.groups)

        return output

    def kl_loss(self):
        kl = calculate_kl(self.prior_mu, self.prior_std, self.W_mu, self.W_normalized_std)

        return kl
