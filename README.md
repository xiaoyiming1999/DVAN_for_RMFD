# A Uncertainty Awareness Method for Trustworthy Mechanical Fault Diagnosis Called DVAN, Developed by the HNU Intelligent Fault Diagnosis Group

## Abstract

This repository provides the official PyTorch implementation of **DVAN (Deep Variational Attention Network)**, an uncertainty-aware method for **Rotating Machinery Fault Diagnosis (RMFD)**. Deterministic deep models output overconfident softmax probabilities and cannot express how much they "don't know" when faced with unseen fault modes. DVAN addresses this by **making the attention mechanism Bayesian**: the attention weights are treated as probability distributions rather than point estimates, so the network can quantify the uncertainty in *where* it attends and *what* it predicts. Through Monte Carlo sampling of the weight posterior, DVAN estimates **epistemic uncertainty** (mutual information) for every input, enabling trustworthy open-set fault diagnosis — known faults are classified, while samples from unseen classes are rejected as out-of-distribution (OOD) via an uncertainty threshold.

## Key Highlights

- **Bayesian attention.** The standard squeeze-and-excitation (SE), CBAM, and self-attention (SA) modules are re-formulated as variational layers whose weights follow a learnable Gaussian posterior, injecting principled uncertainty into the attention maps.
- **Three variants in one framework.** DVAN covers variational SE (`VSE`), variational CBAM (`VCBAM`), and variational self-attention (`VSA`) under a unified backbone, all trained with the evidence lower bound (ELBO).
- **Epistemic-uncertainty OOD detection.** The uncertainty is computed as the mutual information across Monte Carlo weight samples (BALD-style), separating *known-unknown* (epistemic) uncertainty from aleatoric noise, and used to reject unseen fault classes.
- **MC shaping regularization.** An optional auxiliary loss shapes the distribution of attention logits to a target reference, improving the calibration of the attention uncertainty.
- **Comprehensive benchmark.** Two rotating-machinery datasets, nine baselines/variants, and both calibration metrics (ECE, NLL, Brier score) and OOD metrics (accuracy, FAR, MAR, confusion matrix).

## Method

### 1. Background: the need for uncertainty in fault diagnosis

In safety-critical fault diagnosis, a model must not only classify known fault types but also **know when it is facing something it has never seen** (e.g., a novel fault mode or a new operating condition). Deterministic networks trained with cross-entropy produce sharp, often miscalibrated softmax outputs that cannot distinguish a genuinely unfamiliar input from a confidently misclassified one.

### 2. DVAN: Deep Variational Attention Network

DVAN takes a 1-D convolutional backbone (Conv → BN → ReLU, with max-pooling) and replaces the attention modules inserted after intermediate layers with **variational (Bayesian) attention**:

- **Variational linear layer** (`VariationalLinear`) — each weight is parameterized by a posterior mean `μ` and standard deviation `σ` (the std is kept positive via a softplus reparameterization). A forward pass samples the weights with the reparameterization trick `w = μ + ε · softplus(σ)`.
- **Variational convolution** (`VariationalConv`) — the same treatment applied to convolutional weights, with an optional **Flipout** estimator to decorrelate the gradients of the sampled weights.

The three variants differ in *which* attention is made Bayesian:

| Variant | Attention | Variational components                         |
|---------|-----------|-------------------------------------------------|
| `VSE`   | Channel attention (SE-style)   | Variational linear bottleneck (32→16→32, etc.)  |
| `VCBAM` | Channel + position attention   | Variational linear (channel) + variational conv (position) |
| `VSA`   | Self-attention (Q/K/V)         | Variational conv for Q and K projections        |

Each variant returns the classification logits, the concatenated attention logits, and the total KL divergence of all variational layers.

### 3. Training objective (ELBO)

The model is trained by maximizing the evidence lower bound, written here as a loss:

```
L = L_NLL + β · KL(q(w)‖p(w)) + L_MCSL
```

- `L_NLL` — the negative log-likelihood (cross-entropy) averaged over `num_MC_sampling` Monte Carlo weight samples.
- `KL(q(w)‖p(w))` — the KL divergence between the variational posterior `q(w)` and the prior `p(w) = N(prior_mu, prior_std²)`, summed over all variational layers. The trade-off `β` follows the annealing schedule selected by `--beta_type` (`Blundell` by default).
- `L_MCSL` — an optional **MC shaping loss** that aligns the empirical CDF of the attention logits with a target Gaussian reference `N(MC_shaping_mu, MC_shaping_std²)`, regularizing the attention-uncertainty distribution (enabled by `--is_MC_shaping`).

### 4. Uncertainty quantification

At inference, the model is sampled `num_MC_sampling` times, producing a set of predictive distributions `{p_1, …, p_T}`. The **epistemic uncertainty** is estimated by the mutual information (BALD):

```
MI = H( E_t[p_t] ) − E_t[ H(p_t) ]
```

This decomposes the predictive entropy into a data-uncertainty (aleatoric) part and a model-uncertainty (epistemic) part; the latter is what DVAN uses to flag inputs it is unsure about.

### 5. OOD (open-set) fault detection

An uncertainty threshold is calibrated on the validation set using the IQR rule (`threshold = Q3 + 1.5 · IQR`). At test time, any sample whose epistemic uncertainty exceeds the threshold is assigned to the OOD (unseen) class, rather than being forced into one of the known classes. The framework evaluates OOD detection with accuracy, false-alarm rate (FAR), missed-alarm rate (MAR), and a confusion matrix that includes an explicit "OOD" row/column.

## Supported methods (`--method`)

| Method           | Type                          | Uncertainty                     |
|------------------|-------------------------------|---------------------------------|
| `VSE`            | Variational SE                | Mutual information (MC)         |
| `VCBAM`          | Variational CBAM (proposed)   | Mutual information (MC)         |
| `VSA`            | Variational self-attention    | Mutual information (MC)         |
| `softmax-output` | Deterministic 7-layer CNN     | Predictive entropy (MSP)        |
| `SE`             | Deterministic SE              | Predictive entropy              |
| `CBAM`           | Deterministic CBAM            | Predictive entropy              |
| `SA`             | Deterministic self-attention  | Predictive entropy              |
| `MC-dropout`     | 7-layer CNN + MC dropout      | Entropy of MC-averaged softmax  |
| `ensemble`       | 5-model ensemble + FGSM AT    | Entropy of averaged softmax     |

> The `ensemble` baseline trains five models (7-layer CNN, SE, CBAM, SA, Bi-LSTM) jointly with FGSM adversarial augmentation and averages their softmax outputs for uncertainty estimation.

## Datasets (`--data_name`)

| Name          | Object                      | Classes | Format  | Note                                 |
|---------------|-----------------------------|---------|---------|--------------------------------------|
| `THU_gearbox` | Tsinghua gearbox test rig    | 9       | `.tdms` | 7 operating conditions (16–40 Hz); includes a pseudo-OOD set |
| `locomotive`  | Locomotive bearing           | 8       | `.mat`  | single `.mat` file, no operating-condition split |

**Open-set protocol.** By default the model is trained/validated on the first five classes (`train_classes`/`val_classes = [0,1,2,3,4]`) and tested on six (`test_classes = [0,1,2,3,4,5]`), where class `5` is treated as the unseen OOD class. Adjust these lists to change the number of known vs. OOD classes.

> **Note:** dataset root paths are hard-coded as Windows paths (e.g. `D:\datasets`) inside `datasets/*.py`. Edit the `root` argument in the corresponding `data_split()` method to point to your own data location before running.

## Evaluation metrics

**Calibration (per epoch):** ECE, NLL, and Brier score on the validation set.

**OOD detection (on the test set):**

- **Accuracy** — including the OOD class in the closed-set decision.
- **FAR (false-alarm rate)** — fraction of known-class samples wrongly rejected as OOD.
- **MAR (missed-alarm rate)** — fraction of OOD samples wrongly accepted as known classes.
- **Confusion matrix** — with an explicit OOD row/column.

## Requirements

- Python ≥ 3.7
- PyTorch ≥ 1.8
- numpy, pandas, scipy, scikit-learn
- nptdms (for the `.tdms` THU gearbox dataset)
- matplotlib

Install with:

```bash
pip install torch numpy pandas scipy scikit-learn nptdms matplotlib
```

## Usage

### 1. Prepare data

Download the target dataset(s) and update the `root` path in the corresponding file under `datasets/`.

### 2. Run training

Train the proposed variational CBAM on the locomotive bearing dataset:

```bash
python DVAN_main.py --method VCBAM --data_name locomotive
```

Train another variant or a baseline:

```bash
python DVAN_main.py --method VSE --data_name THU_gearbox
python DVAN_main.py --method MC-dropout --data_name locomotive
python DVAN_main.py --method ensemble --data_name locomotive
```

The training script runs `setup()`, `train()`, and `test()` automatically, logging calibration metrics per epoch and OOD results at the end.

### 3. Key arguments

| Argument              | Type   | Default            | Description                                        |
|-----------------------|--------|--------------------|----------------------------------------------------|
| `--method`            | str    | `VCBAM`            | Method (see methods table)                         |
| `--data_name`         | str    | `locomotive`       | Dataset (`THU_gearbox` or `locomotive`)            |
| `--data_length`       | int    | `1024`             | Signal segment length (per sample)                 |
| `--num_classes`       | int    | `5`                | Number of known (training) classes                 |
| `--train_classes`     | list   | `[0,1,2,3,4]`      | Known classes for training                         |
| `--val_classes`       | list   | `[0,1,2,3,4]`      | Known classes for validation                       |
| `--test_classes`      | list   | `[0,1,2,3,4,5]`    | Test classes (last one is OOD)                     |
| `--num_train_samples` | int    | `200`              | Samples per class for training                     |
| `--num_val_samples`   | int    | `50`               | Samples per class for validation                   |
| `--num_test_samples`  | int    | `50`               | Samples per class for testing                      |
| `--train_noise_SNR`   | int    | `50`               | SNR (dB) of training AWGN                          |
| `--val_noise_SNR`     | int    | `50`               | SNR (dB) of validation AWGN                        |
| `--test_noise_SNR`    | int    | `50`               | SNR (dB) of test AWGN                              |
| `--pseudo_noise_SNR`  | int    | `-5`               | SNR for the pseudo-OOD set (THU only)              |
| `--epoch`             | int    | `30`               | Number of epochs                                   |
| `--batch_size`        | int    | `128`              | Batch size                                         |
| `--lr`                | float  | `1e-3`             | Initial learning rate                              |
| `--weight_decay`      | float  | `1e-5`             | L2 weight decay                                    |
| `--gamma`             | float  | `0.1`              | LR scheduler decay factor                          |
| `--steps`             | str    | `15,25`            | LR decay milestone epochs (comma-separated)        |
| `--num_MC_sampling`   | int    | `20`               | Monte Carlo samples for uncertainty estimation     |
| `--is_MC_shaping`     | bool   | `True`             | Enable the MC shaping loss                         |
| `--prior_mu`          | float  | `0`                | Prior mean of the weight distribution              |
| `--prior_std`         | float  | `0.2`              | Prior std of the weight distribution               |
| `--MC_shaping_mu`     | float  | `0`                | Target mean of the MC shaping reference            |
| `--MC_shaping_std`    | float  | `1`                | Target std of the MC shaping reference             |
| `--beta_type`         | str    | `Blundell`         | KL annealing schedule (`Blundell`/`Soenderby`/`Standard`) |
| `--cuda_device`       | str    | `0`                | GPU device id                                      |

Checkpoints and logs are written to `./checkpoint/<model_name>_<timestamp>/train.log`.

## Project structure

```
DVAN_for_RMFD/
├── DVAN_main.py                    # entry point: argument parsing & trainer dispatch
├── models/
│   ├── variational_SE.py           # VSE — variational squeeze-and-excitation
│   ├── variational_CBAM.py         # VCBAM — variational CBAM (proposed)
│   ├── variational_SA.py           # VSA — variational self-attention
│   ├── deterministic_SE.py         # SE — deterministic squeeze-and-excitation
│   ├── deterministic_CBAM.py       # CBAM — deterministic CBAM
│   ├── deterministic_SA.py         # SA — deterministic self-attention
│   ├── cnn_7layer.py               # 7-layer 1-D CNN (softmax-output / MC-dropout backbone)
│   ├── BILSTM.py                   # Bi-LSTM (ensemble member)
│   └── loss.py                     # ECE and Brier-score metrics
├── variational_layer/
│   ├── variational_linear.py       # Bayesian linear layer (reparameterization + KL)
│   ├── variational_conv.py         # Bayesian conv layer (reparameterization + Flipout)
│   └── KL_loss.py                  # KL divergence & β annealing schedules
├── datasets/
│   ├── THU_Gearbox.py              # Tsinghua gearbox loader (.tdms) + pseudo-OOD set
│   ├── Loco_bearing.py             # Locomotive bearing loader (.mat)
│   ├── SequenceDatasets.py         # torch Dataset wrapper
│   ├── sequence_aug.py             # Data augmentation (scale/stretch/crop/noise)
│   └── data_preprocess.py          # Mean-std / min-max normalization
└── utils/
    ├── DVAN_train.py               # Bayesian training loop + ELBO + threshold calibration
    ├── Vanilla_train.py            # Deterministic baseline training loop
    ├── MC_train.py                 # MC-dropout baseline training loop
    ├── Ensemble_train.py           # 5-model ensemble + FGSM adversarial training
    ├── prediction.py               # Uncertainty (MI/entropy), threshold, FAR/MAR, confusion matrix
    └── logger.py                   # File + console logger
```

## Citation

If you find this repository useful in your research, please consider citing the following paper:

> Yiming Xiao, Haidong Shao, Haomiao Zhang, Rongming Wei, Bin Liu. Uncertainty-aware deep variational attention network: A trustworthy mechanical fault diagnostic model assisted by out-of-distribution detection[J]. Engineering Applications of Artificial Intelligence, 2025, 157: 111386.

BibTeX:

```bibtex
@article{xiao2025uncertainty,
  author  = {Xiao, Yiming and Shao, Haidong and Zhang, Haomiao and Wei, Rongming and Liu, Bin},
  title   = {Uncertainty-aware deep variational attention network: A trustworthy mechanical fault diagnostic model assisted by out-of-distribution detection},
  journal = {Engineering Applications of Artificial Intelligence},
  volume  = {157},
  pages   = {111386},
  year    = {2025},
  doi     = {10.1016/j.engappai.2025.111386}
}
```

## Acknowledgements

This work is developed by the HNU Intelligent Fault Diagnosis Group. The variational-layer implementations are adapted from open-source Bayesian deep-learning codebases, and the 1-D attention backbones are adapted from public PyTorch model zoos.

## Contact

- **Author:** Yiming Xiao — xiaoym@hnu.edu.cn
- **Mentor:** Haidong Shao — hdshao@hnu.edu.cn

## License

The code is released for research and educational purposes. Please contact the authors for further usage.
