# Volatility Regime Detection in Cryptocurrency Markets
### A Comparative Study of Hilbert-Huang Transform and Short-Time Fourier Transform

**Implementation of:**
> Leung, T. & Zhao, T. (2021). *Adaptive Complementary Ensemble EMD and Energy-Frequency Spectra of Cryptocurrency Prices.* International Journal of Financial Engineering. [arXiv:2105.08133](https://arxiv.org/abs/2105.08133)

**Course:** EC211 — Digital Signal Processing Lab  
**Institution:** NIT Karnataka  

---

## Table of Contents
- [Overview](#overview)
- [Background & Motivation](#background--motivation)
- [Theory](#theory)
  - [1. EMD and its Limitations](#1-emd-and-its-limitations)
  - [2. ACE-EMD](#2-ace-emd-adaptive-complementary-ensemble-emd)
  - [3. Timescale Filtering & Volatility Analysis](#3-timescale-filtering--volatility-analysis)
  - [4. Hilbert Spectral Analysis (HHT)](#4-hilbert-spectral-analysis-hht)
  - [5. Frequency Synchronization](#5-frequency-synchronization)
  - [6. HHT vs STFT — Our Contribution](#6-hht-vs-stft--our-original-contribution)
- [Notebook Structure](#notebook-structure)
- [Results](#results)
- [Installation & Usage](#installation--usage)
- [Citation](#citation)
- [License](#license)

---

## Overview

This project implements the **ACE-EMD (Adaptive Complementary Ensemble Empirical Mode Decomposition)** framework from Leung & Zhao (2021) applied to Bitcoin (BTC-USD) and Ethereum (ETH-USD) price signals over January 2016 – March 2021.

We reproduce the paper's core results — IMF decomposition, timescale filtering, energy-frequency spectra, and cryptocurrency synchronization — and extend the paper with an **original comparison of HHT vs STFT** for time-frequency analysis of crypto volatility.

**What the paper proposes:**
1. Decompose cryptocurrency log-prices into multi-scale Intrinsic Mode Functions (IMFs) using ACE-EMD
2. Build adaptive low-pass / high-pass filters for trend vs. volatility separation
3. Compute instantaneous energy-frequency spectra via the Hilbert transform
4. Estimate power spectrum exponents and analyze cross-asset frequency synchronization

---

## Background & Motivation

Cryptocurrency prices are **non-stationary** and **nonlinear** — classical Fourier analysis assumes stationarity and linearity, making it ill-suited for this domain. Traditional volatility models (GARCH, rolling standard deviation) operate on fixed timescales, missing the multi-resolution structure embedded in price dynamics.

**Empirical Mode Decomposition (EMD)** offers a data-driven, adaptive alternative: it decomposes any signal into oscillatory components (IMFs) without assuming any basis, making it ideal for nonlinear, non-stationary financial time series.

---

## Theory

### 1. EMD and its Limitations

**Standard EMD** decomposes a signal $x(t)$ into $n$ Intrinsic Mode Functions (IMFs) via iterative sifting, plus a residual trend:

$$x(t) = \sum_{j=1}^{n} c_j(t) + r_n(t)$$

Each IMF $c_j(t)$ must satisfy:
- The number of extrema and zero-crossings differ by at most one
- The mean of the upper and lower envelopes is zero at every point

**The mode mixing problem:** In practice, plain EMD suffers from *mode mixing* — a single IMF can contain oscillations of widely disparate scales, blurring the physical interpretation.

**EEMD fix:** Ensemble EMD adds white noise $w_i(t)$ across $N$ trials and averages:

$$c_j(t) = \frac{1}{N} \sum_{i=1}^{N} c_{ij}(t)$$

The noise fills the scale gaps, reducing mode mixing. However, reconstruction is no longer exact since the added noise doesn't fully cancel.

**CEEMD improvement:** Complete EEMD with Adaptive Noise adds *complementary pairs* $\pm w_i(t)$, ensuring exact reconstruction:

$$\sum_{j=1}^{n} c_j(t) + r_n(t) = x(t) \quad$$

---

### 2. ACE-EMD: Adaptive Complementary Ensemble EMD

The paper's key contribution: rather than using a fixed noise amplitude, ACE-EMD makes the noise level **adaptive to local signal volatility**:

$$\text{Var}[w_i(t)] = \sigma^2 \cdot a_p^2(t)$$

where $a_p(t)$ is the instantaneous amplitude envelope of the *pilot IMF* $c_p(t)$ — the first IMF extracted from a plain EMD run. This ensures the injected noise is always proportional to the local signal energy, producing more consistent IMF decompositions across quiet and volatile market regimes.

**Algorithm:**
1. Run a pilot EMD on $x(t)$ → extract first IMF $c_p(t)$
2. Estimate local amplitude $a_p(t)$ via cubic spline interpolation on the extrema envelope
3. For each ensemble trial $i = 1, \ldots, N$:
   - Add adaptive noise: $x_i^{\pm}(t) = x(t) \pm \sigma \cdot a_p(t) \cdot \xi_i(t)$, where $\xi_i(t) \sim \mathcal{N}(0, 1)$
   - Run CEEMD on $x_i^+(t)$ and $x_i^-(t)$
4. Average IMFs across all trials

> **Implementation note:** We approximate ACE-EMD using `CEEMDAN` from the `PyEMD` library, which is the closest publicly available implementation. The adaptive noise scaling is the primary distinction from standard CEEMDAN.

---

### 3. Timescale Filtering & Volatility Analysis

ACE-EMD naturally acts as an **adaptive filter bank** — IMFs are ordered from highest to lowest frequency:

- **IMF 1** → highest frequency component (rapid noise / micro-fluctuations)
- **IMF n + Residual** → lowest frequency (long-term trend)

#### Low-pass Filter
Keep the last $m_l$ components (including residual) — retains the smooth trend (Eq. 8):

$$x_L^{(m_l)}(t) = x(t) - \sum_{j=1}^{n - m_l + 1} c_j(t)$$

#### High-pass Filter
Keep only the first $m_h$ IMFs — retains high-frequency volatility (Eq. 9):

$$x_H^{(m_h)}(t) = \sum_{j=1}^{m_h} c_j(t)$$

The paper uses $m_l = 4$ and $m_h = 2$ for both BTC and ETH, selected to balance trend smoothness against information loss.

#### Conditional (Asymmetric) Volatility
Upside and downside volatility are computed from the high-pass filtered returns $r_H(t) = \Delta x_H(t)$ (Equations 15–16):

$$\sigma^{(m_h)}_{+H} = \sqrt{\,\text{Var}\!\left(r_H^{(m_h)}(t) \;\middle|\; r_H^{(m_h)}(t-1) > \mu_H\right)}$$

$$\sigma^{(m_h)}_{-H} = \sqrt{\,\text{Var}\!\left(r_H^{(m_h)}(t) \;\middle|\; r_H^{(m_h)}(t-1) < \mu_H\right)}$$

This captures the well-known **leverage effect** in financial markets — downside volatility typically exceeds upside volatility.

---

### 4. Hilbert Spectral Analysis (HHT)

For each IMF $c_j(t)$, the **Hilbert transform** gives the analytic signal from which we extract time-varying amplitude and frequency:

| Quantity | Formula | 
|---|---|
| Instantaneous amplitude | $a_j(t) = \|Z_j(t)\|$ | 
| Instantaneous phase | $\theta_j(t) = \arg Z_j(t)$ | 
| Instantaneous frequency | $f_j(t) = \frac{1}{2\pi}\dot{\theta}_j(t)$ | 
| Instantaneous energy | $E_j(t) = \|a_j(t)\|^2$ | 

The combination of EMD + Hilbert transform is called the **Hilbert-Huang Transform (HHT)**.

#### Energy-Frequency Spectrum

To build a time-averaged energy-frequency picture, we compute the **central frequency** and **central energy** of each IMF using geometric means (Eqs. 28–29):

$$\bar{f}_j = \exp\!\left(\frac{1}{T}\int_0^T \log f_j(t)\,dt\right), \qquad
\bar{E}_j = \exp\!\left(\frac{1}{T}\int_0^T \log E_j(t)\,dt\right)$$

#### Power Spectrum Exponent

A log-log regression of central energy against central frequency gives the power-law exponent $\alpha$ (Eq. 30):

$$\bar{E}(f) \sim \frac{1}{f^{\,\alpha}}$$

$$\log \bar{E}_j = -\alpha \log \bar{f}_j + \text{const}$$

The paper finds:
| Asset | $\alpha$ | $R^2$ |
|-------|----------|--------|
| BTC | 1.2070 | 0.9911 |
| ETH | 1.1681 | 0.9898 |
| S&P 500 (reference) | ~0.89 | — |

Both cryptocurrencies have $\alpha > 1$, meaning energy decays *faster* with frequency compared to traditional equity indices — reflecting the more speculative, noise-driven nature of crypto markets.

---

### 5. Frequency Synchronization

The paper observes that BTC and ETH share strikingly similar instantaneous frequency profiles — evidence of **cross-asset synchronization** in a nonlinear dynamics sense.

The **frequency deviation** between two assets $x_1$ and $x_2$ is measured as (Eq. in Section 4.3):

$$D(x_1, x_2) = \left\| \log\bar{f}^{(1)} - \log\bar{f}^{(2)} \right\|_2$$

$D = 0$ indicates perfect synchronization (identical central frequencies across all IMFs). The paper shows $D(\text{BTC}, \text{ETH}) \ll D(\text{BTC}, \text{SandP 500})$, confirming that cryptocurrencies form a tightly coupled cluster driven by common speculative dynamics.

---

### 6. HHT vs STFT — Our Original Contribution

This section extends beyond Leung and Zhao (2021) by directly comparing HHT against the classical **Short-Time Fourier Transform (STFT)** on BTC log-returns.

| Property | STFT | HHT |
|---|---|---|
| Basis | Sinusoids (predefined) | IMFs (data-driven) |
| Window | Fixed (Hann, length $L$) | Adaptive |
| Time-frequency resolution | Limited by Heisenberg uncertainty | Arbitrary |
| Handles non-stationarity | ❌ No | ✅ Yes |
| Handles nonlinearity | ❌ No | ✅ Yes |
| Computational cost | $O(L \log L)$ per frame | Higher (iterative sifting) |

**STFT limitation — the Heisenberg-Gabor uncertainty principle:**

$$\Delta t \cdot \Delta f \geq \frac{1}{4\pi}$$

A short window gives good time resolution but poor frequency resolution; a long window does the opposite. This fixed trade-off is ill-suited to financial signals where volatility regimes shift abruptly.

**HHT advantage:** By decomposing the signal into IMFs first and then applying the Hilbert transform, HHT achieves a time-frequency representation with no predefined window — frequency resolution adapts to the signal's own structure. This makes HHT superior for detecting sudden volatility regime changes in crypto markets.

We apply both methods to BTC log-returns and compare the resulting spectrograms, demonstrating that HHT resolves short-lived high-energy events (e.g., COVID crash in March 2020) with greater clarity than STFT.

---

## Notebook Structure

| Section | Content | Paper Reference |
|---------|---------|----------------|
| 0 | Imports & Setup | — |
| 1 | Data Acquisition (BTC, ETH — 2016–2021) | — |
| 2 | ACE-EMD Decomposition | Section 2, Fig. 1 |
| 3 | Timescale Filtering & Volatility Analysis | Section 3, Fig. 2–4 |
| 4 | Energy-Frequency Spectrum (HHT) | Section 4, Fig. 6 |
| 5 | Frequency Synchronization BTC vs ETH | Section 4.3, Fig. 7 |
| 6 | **HHT vs STFT Comparison** | **Our contribution** |
| 7 | Results Summary | — |

---

## Results

| Metric | BTC | ETH |
|--------|-----|-----|
| Number of IMFs | 7 | 7 |
| Power spectrum exponent $\alpha$ | 1.1001 | 1.0679 |
| $R^2$ of power-law fit | 0.9736 | 0.9860 |
| Freq. deviation $D(\text{BTC}, \text{ETH})$ | 1.1965 | — |

---

## Installation & Usage

### Requirements
```
Python 3.8+
numpy
pandas
matplotlib
scipy
EMD-signal       # provides PyEMD / CEEMDAN
yfinance         # for live BTC/ETH data download
```

### Install dependencies
```bash
pip install numpy pandas matplotlib scipy EMD-signal yfinance
```

### Run
```bash
jupyter notebook Final_Code.ipynb
```

Run all cells top-to-bottom. The notebook will automatically download BTC-USD and ETH-USD data from Yahoo Finance. If `yfinance` is unavailable, it falls back to simulated data with a warning.

---

## Citation

If you use this code or build upon it, please cite the original paper:

```bibtex
@article{leung2021adaptive,
  title     = {Adaptive Complementary Ensemble {EMD} and Energy-Frequency Spectra of Cryptocurrency Prices},
  author    = {Leung, Tim and Zhao, Theodore},
  journal   = {International Journal of Financial Engineering},
  year      = {2021},
  note      = {arXiv:2105.08133}
}
```

**arXiv preprint:** https://arxiv.org/abs/2105.08133

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The implementation is for educational and research purposes. All financial data is retrieved from Yahoo Finance via the `yfinance` library.
