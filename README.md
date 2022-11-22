# Topological Metrics

## Abstract
> We propose a robust and reliable evaluation metric for generative models by
introducing topological and statistical treatments for a rigorous support manifold
estimation. Existing metrics, such as Inception Score (IS), Fréchet Inception
Distance (FID), and the variants of Precision and Recall (P&R), heavily
rely on support manifolds that are estimated from sample features. However, the
reliability of their estimation has not been seriously discussed (and overlooked)
even though the quality of the evaluation entirely depends on it. In this paper, we
propose Topological Precision and Recall (TopP&R, pronounced “topper”), which
provides a systematic approach to estimating support manifolds, retaining only
topologically and statistically important features with a certain level of confidence.
This not only makes TopP&R strong for noisy features, but also provides statistical
consistency. Our theoretical and experimental results show that TopP&R is robust
to outliers and non-independent and identically distributed (Non-IID) perturbations,
while accurately capturing the true trend of change in samples. To the best of our
knowledge, this is the first evaluation metric focused on the robust estimation of
the support manifold and provides its statistical consistency under noise.

## Overview of topological precision and recall (TopP&R)
![toppr_overview](https://user-images.githubusercontent.com/102020840/203247514-3f64b9e6-bf74-434e-8c40-c6dfdfec7e59.png)
The proposed metric TopP&R is defined in the following three steps: (a) Confidence band estimation with bootstrapping in section 2,
(b) Robust support estimation, and (c) Evaluationn via TopP&R in section 3 of our paper.

## Top precision and recall metric
We define the precision and recall of data points as

$$precision_P(\mathcal{Y}):={\sum_{j=1}^m1(Y_j\in supp(P)\cap supp(Q)) / \sum^m_{j=1}1(Y_j\in supp(Q))}$$

$$recall_Q(\mathcal{X}):={\sum_{i=1}^n 1(X_i\in supp(Q)\cap supp(P)) / \sum_{i=1}^n 1(X_i\in supp(P))}$$

In practice, $supp(P)$ and $supp(Q)$ are not known a priori and need to be estimated, and since we allow noise,
these estimates should be robust to noise. For this, we use the kernel density estimator (KDE) and 
the bootstrap bandwidth to robustly estimate the support. 
Using the estimated support (superlevel set at $c_{\mathcal{X}}$ and $c_{\mathcal{Y}}$), we define
the topological precision (TopP) and recall (TopR) as bellow:

$$TopP_{\mathcal{X}}(\mathcal{Y}):=\sum^m_{j=1}1(\hat{p_{h_n}}(Y_j)>c_{\mathcal{X}},\hat{q_{h_m}}(Y_j)>c_{\mathcal{Y}}) / 
\sum^m_{j=1} 1(\hat{q_{h_m}}(Y_j)>c_{\mathcal{Y}})$$

$$TopR_{\mathcal{Y}}(\mathcal{X}):=\sum^n_{i=1}1(\hat{q_{h_m}}(X_i)>c_{\mathcal{Y}},\hat{p_{h_n}}(X_i)>c_{\mathcal{X}}) / 
\sum^n_{i=1} 1(\hat{p_{h_n}}(X_i)>c_{\mathcal{X}})$$

The kernel bandwidths $h_n$ and $h_m$ are hyperparameters that users need to choose. We also provide our guide line to select 
the optimal bandwidths $h_n$ and $h_m$ in practice (see our Appendix F.3).

## How to use
```python
from module import top_pr.top_pr as TopPR
import numpy as np

# Real data
REAL = np.random.normal(loc=0.0, scale=1.0, size=[10000, 64])
# Fake data
FAKE = np.random.normal(loc=0.4, scale=1.0, size=[10000, 64])

# Evaluation
TopPR(REAL, FAKE, alpha = 0.1, kernel = "cosine", random_proj = True, f1_score = True)
print(Top_PR.get('fidelity'), Top_PR.get('diversity'), Top_PR.get('Top_F1'))
```
