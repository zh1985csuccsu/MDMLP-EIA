# MDMLP-EIA: Multi-domain Dynamic MLPs with Energy Invariant Attention for Time Series Forecasting

[![arXiv](https://img.shields.io/badge/arXiv-2511.09924-b31b1b.svg)](https://arxiv.org/abs/2511.09924)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-ee4c2c.svg)](https://pytorch.org/)

This is the **official implementation** of **MDMLP-EIA** (Multi-domain Dynamic MLPs with Energy Invariant Attention) for time series forecasting.

**Authors:** Hu Zhang, Zhien Dai, Zhaohui Tang, Yongfang Xie

---

## Abstract

Time series forecasting is essential across diverse domains. While MLP-based methods have gained attention for achieving Transformer-comparable performance with fewer parameters and better robustness, they face critical limitations including loss of weak seasonal signals, capacity constraints in weight-sharing MLPs, and insufficient channel fusion in channel-independent strategies. To address these challenges, we propose MDMLP-EIA with three key innovations:

1. **Adaptive fused dual-domain seasonal MLP** — Categorizes seasonal signals into strong and weak components and employs an **adaptive zero-initialized channel fusion (AZCF)** strategy to minimize noise interference while effectively integrating predictions.
2. **Energy invariant attention (EIA)** — Adaptively focuses on different feature channels within trend and seasonal predictions across time steps while maintaining constant total signal energy to align with the decomposition–prediction–reconstruction framework.
3. **Dynamic capacity adjustment (DCA)** — Scales MLP neuron count with the square root of channel count for channel-independent MLPs, ensuring sufficient capacity as channels increase.

Extensive experiments across nine benchmark datasets demonstrate that MDMLP-EIA achieves state-of-the-art performance in both prediction accuracy and computational efficiency.

- **Paper:** [arXiv:2511.09924](https://arxiv.org/abs/2511.09924) | [PDF](https://arxiv.org/pdf/2511.09924)
- **DOI:** [10.48550/arXiv.2511.09924](https://doi.org/10.48550/arXiv.2511.09924)

---

## Environment

- Python 3.8+
- PyTorch ≥ 1.7.0
- CUDA (optional, for GPU training)

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/MDMLP-EIA.git
cd MDMLP-EIA
pip install -r requirements.txt
```

**Requirements:** See `requirements.txt`. Main dependencies: `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`.

---

## Data

Place datasets under `./dataset/` (or set `--root_path`). Supported datasets:

---

## Quick Start

### Training and testing (single run)

```bash
python run.py \
  --is_training 1 \
  --model_id ETTh1_96_96 \
  --model MDMLP_EIA \
  --data ETTh1 \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7 \
  --des exp \
  --itr 1
```

## Citation

If you use this code or the paper in your work, please cite:

```bibtex
@article{zhang2025mdmlp-eia,
  title   = {MDMLP-EIA: Multi-domain Dynamic MLPs with Energy Invariant Attention for Time Series Forecasting},
  author  = {Zhang, Hu and Dai, Zhien and Tang, Zhaohui and Xie, Yongfang},
  journal = {arXiv preprint arXiv:2511.09924},
  year    = {2025},
  url     = {https://arxiv.org/abs/2511.09924}
}
```

---

## License

This project is for academic use. Please refer to the paper and arXiv for terms of use.
