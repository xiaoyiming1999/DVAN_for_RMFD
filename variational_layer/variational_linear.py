import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from variational_layer.KL_loss import calculate_kl


class VariatinalLinear(nn.Module):

    def __init__(self, in_features, out_features, device, priors=None):
        super(VariatinalLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        self.prior_mu = priors['prior_mu']
        self.prior_std = priors['prior_std']

        self.posterior_mu_initial = (0, 0.1)
        self.posterior_std_initial = (-3, 0.1)

        self.W_mu = Parameter(torch.Tensor(out_features, in_features))
        self.W_std = Parameter(torch.Tensor(out_features, in_features))

        self.bias_mu = Parameter(torch.Tensor(out_features))
        self.bias_std = Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_std.data.normal_(*self.posterior_std_initial)

        self.bias_mu.data.normal_(*self.posterior_mu_initial)
        self.bias_std.data.normal_(*self.posterior_std_initial)

    def forward(self, x):

        batch_size = x.size(0)

        # generate random sign
        # r_in = torch.randint(0, 2, (batch_size, self.in_features)).float() * 2 - 1  # [B, D_in]
        # r_out = torch.randint(0, 2, (batch_size, self.out_features)).float() * 2 - 1  # [B, D_ou
        # r_in = r_in.to(self.device)
        # r_out = r_out.to(self.device)

        self.W_normalized_std = torch.log1p(torch.exp(self.W_std))
        w_eps = torch.empty(self.W_normalized_std.size()).normal_(0, 1).to(self.device)
        act_w = self.W_mu + w_eps * self.W_normalized_std

        self.bias_normalized_std = torch.log1p(torch.exp(self.bias_std))
        bias_eps = torch.empty(self.bias_normalized_std.size()).normal_(0, 1).to(self.device)
        act_bias = self.bias_mu + bias_eps * self.bias_normalized_std

        # Flipout
        # x_perturbed = x * r_in
        output = F.linear(x, act_w, act_bias)
        # output = output * r_out

        return output

    def kl_loss(self):

        kl = calculate_kl(self.prior_mu, self.prior_std, self.W_mu, self.W_normalized_std)
        kl += calculate_kl(self.prior_mu, self.prior_std, self.bias_mu, self.bias_normalized_std)

        return kl