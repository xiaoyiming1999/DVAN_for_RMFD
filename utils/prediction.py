import math

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def KL_divergence(p_mean, p):
    kl = 0.0
    for i in range(len(p)):
        kl += p_mean[i] * math.log(p_mean[i]/(p[i]+1e-16) + 1e-16, 2)
    return kl


def entropy(p, dim):

    return -torch.sum(p * torch.log(p + 1e-10), dim=dim)


# mutual information
def calculate_mi(tensor):

    tensor = torch.nn.functional.softmax(tensor, dim=-1)
    mean_probs = tensor.mean(dim=0)
    H_mean = entropy(mean_probs, dim=1)
    entropies = entropy(tensor, dim=2)
    E_H = entropies.mean(dim=0)
    mi = H_mean - E_H
    return mi


def calculate_epistemic_uncertainty(probs):
    probs = probs.softmax(dim=-1)
    mean_outputs = torch.mean(probs, dim=0)
    epistemic_uncertainty = []
    for data_index in range(probs.shape[1]):
        uncertainty = []
        for sampling_index in range(probs.shape[0]):
            outputs = probs[sampling_index, data_index, :].tolist()
            outputs_kl = KL_divergence(mean_outputs[data_index, ], outputs)
            uncertainty.append(outputs_kl)
        sum = 0
        for i in uncertainty:
            sum = sum + i
        epistemic_uncertainty.append(sum / probs.shape[0])
    epistemic_uncertainty = torch.tensor(epistemic_uncertainty).reshape(-1, 1)
    # print(epistemic_uncertainty.shape)
    return epistemic_uncertainty


def calculate_threshold(uncertainty):
    sorted_epistemic, _ = torch.sort(uncertainty, dim=0)
    Q1_index = ((sorted_epistemic.shape[0] + 1) * 0.25)
    Q3_index = ((sorted_epistemic.shape[0] + 1) * 0.75)
    Q1 = sorted_epistemic[int(Q1_index), ]
    Q3 = sorted_epistemic[int(Q3_index), ]
    IQR = Q3 - Q1
    uncertainty_threshold = Q3 + 1.5 * IQR

    return uncertainty_threshold


def calculate_MCSL_loss(attn_logits, MC_shaping_mu, MC_shaping_std, device):

    sorted, _ = torch.sort(attn_logits, dim=1, descending=False)
    cdf = 0.5 + 0.5 * torch.erf((sorted - MC_shaping_mu) / (2 ** 0.5 * MC_shaping_std))

    arrange =torch.arange(start=1/(attn_logits.shape[1] + 1), end=1, step=1/(attn_logits.shape[1] + 1))
    arrange = torch.unsqueeze(arrange, dim=0)
    arrange = torch.unsqueeze(arrange, dim=-1)
    arrange = arrange.to(device)
    cdf = torch.as_tensor(cdf).to(device)
    MC_shaping_loss = torch.mean((cdf - arrange) ** 2)

    return MC_shaping_loss


def vattn_bayes_predict(inputs, labels, beta, criterion, model, num_MC_sampling,
                        device, MC_shaping_mu, MC_shaping_std):

    probs = []
    attn = []
    loss_likelihood = 0.0
    loss_prior = 0.0

    for _ in range(num_MC_sampling):

        MC_outputs, attn_logits, kl = model(inputs)
        attn.append(attn_logits)
        probs.append(MC_outputs)
        loss_likelihood += criterion(MC_outputs, labels)
        loss_prior += beta * kl

    attn = torch.stack(attn, dim=1)
    probs = torch.stack(probs)
    outputs = probs.mean(dim=0)
    loss_likelihood = loss_likelihood / num_MC_sampling
    loss_prior = loss_prior / num_MC_sampling

    loss_MCSL = calculate_MCSL_loss(attn, MC_shaping_mu, MC_shaping_std, device)

    return outputs, loss_likelihood, loss_prior, loss_MCSL, probs


def vattn_bayes_test(inputs, model, num_MC_sampling):

    probs = []

    for _ in range(num_MC_sampling):

        MC_outputs, _, _ = model(inputs)
        probs.append(MC_outputs)

    probs = torch.stack(probs)
    outputs = probs.mean(dim=0)

    # calculate uncertainty
    epistemic_uncertainty = calculate_mi(probs)

    return outputs, epistemic_uncertainty


def cm_plot(original_label, predict_label, num_classes):

    # calculate confusion matrix
    classes = [i for i in range(num_classes)]
    classes += ['OOD']
    tick_marks = np.arange(len(classes))
    cm = confusion_matrix(original_label, predict_label)
    # plot confusion matrix
    thresh = cm.max() / 2.
    plt.figure(dpi=400)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.GnBu)
    # plt.colorbar()
    plt.xticks(tick_marks, classes, fontsize=16)
    plt.yticks(tick_marks, classes, fontsize=16, rotation=90)
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(y, x), horizontalalignment='center', fontsize=18,
                         verticalalignment='center', color="white" if cm[x, y] > thresh else "black")
    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 15,
             }
    plt.ylabel(u"True label", font1)
    plt.xlabel(u"Predicted label", font1)
    plt.show()

def calculate_FAR_MAR(predicted_label, true_label):
    TP = np.sum((true_label == 1) & (predicted_label == 1))
    TN = np.sum((true_label == 0) & (predicted_label == 0))
    FP = np.sum((true_label == 0) & (predicted_label == 1))
    FN = np.sum((true_label == 1) & (predicted_label == 0))

    FAR = FP / (FP + TN)
    MAR = FN / (TP + FN)

    return FAR, MAR

