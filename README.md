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

## How TopP&R is defined?
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
# Call packages
import matplotlib.pyplot as plot
import seaborn as sns
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances
import torch
from module import mode_drop
from module.top_pr import top_pr as TopPR
from prdc import compute_prdc

# Get dataset ('simultaneous' mode drop case. User can change simultaneous to sequential for 'sequential' mode drop case)
simul_data = np.array([mode_drop.gaussian_mode_drop(method = 'simultaneous', ratio = 0), 
    mode_drop.gaussian_mode_drop(method = 'simultaneous', ratio = 0.1),
    mode_drop.gaussian_mode_drop(method = 'simultaneous', ratio = 0.2),
    mode_drop.gaussian_mode_drop(method = 'simultaneous', ratio = 0.3),
    mode_drop.gaussian_mode_drop(method = 'simultaneous', ratio = 0.4),
    mode_drop.gaussian_mode_drop(method = 'simultaneous', ratio = 0.5),
    mode_drop.gaussian_mode_drop(method = 'simultaneous', ratio = 0.6),
    mode_drop.gaussian_mode_drop(method = 'simultaneous', ratio = 0.7),
    mode_drop.gaussian_mode_drop(method = 'simultaneous', ratio = 0.8),
    mode_drop.gaussian_mode_drop(method = 'simultaneous', ratio = 0.9),
    mode_drop.gaussian_mode_drop(method = 'simultaneous', ratio = 1.0)
    ])

# Evaluation step
for iloop in range(10):
    start = 0
    for Ratio in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # Define real and fake dataset
        REAL = mode_drop.gaussian_mode_drop(method = 'simultaneous', ratio = 0)
        FAKE = mode_drop.gaussian_mode_drop(method = 'simultaneous', ratio = Ratio)
        
        # Evaluation with TopPR
        Top_PR = TopPR(REAL, FAKE, alpha = 0.1, kernel = "cosine", random_proj = True, f1_score = True)
        
        # Evaluation with P&R and D&C
        PR = compute_prdc(REAL, FAKE, 3)
        DC = compute_prdc(REAL, FAKE, 5)
        
        if (start == 0):
            pr = [PR.get('precision'), PR.get('recall')]
            dc = [DC.get('density'), DC.get('coverage')]
            Top_pr = [Top_PR.get('fidelity'), Top_PR.get('diversity'), Top_PR.get('Top_F1')]
            start = 1
            
        else:
            pr = np.vstack((pr, [PR.get('precision'), PR.get('recall')]))
            dc = np.vstack((dc, [DC.get('density'), DC.get('coverage')]))
            Top_pr = np.vstack((Top_pr, [Top_PR.get('fidelity'), Top_PR.get('diversity'), Top_PR.get('Top_F1')]))

# Visualization of Result
x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
fig = plot.figure(figsize = (12,3))
for i in range(1,3):
    axes = fig.add_subplot(1,2,i)
    
    # Fidelity
    if (i == 1):
        axes.set_title("Fidelity",fontsize = 15)
        plot.ylim([0.5, 1.5])
        plot.plot(x, Top_pr[:,0], color = [255/255, 110/255, 97/255], linestyle = '-', linewidth = 3, marker = 'o', label = "TopP")
        plot.plot(x, pr[:,0], color = [77/255, 110/255, 111/255], linestyle = ':', linewidth = 3, marker = 'o', label = "precision (k=3)")
        plot.plot(x, dc[:,0], color = [15/255, 76/255, 130/255], linestyle = '-.', linewidth = 3, marker = 'o', label = "density (k=5)")
        plot.plot(x, np.linspace(1.0, 1.0, 11), color = 'black', linestyle = ':', linewidth = 2)
        plot.legend(fontsize = 9)
    
    # Diversity
    elif (i == 2):
        axes.set_title("Diversity",fontsize = 15)
        plot.plot(x, Top_pr[:,1], color = [255/255, 110/255, 97/255], linestyle = '-', linewidth = 3, marker = 'o', label = "TopR")
        plot.plot(x, pr[:,1], color = [77/255, 110/255, 111/255], linestyle = ':', linewidth = 3, marker = 'o', label = "recall (k=3)")
        plot.plot(x, dc[:,1], color = [15/255, 76/255, 130/255], linestyle = '-.', linewidth = 3, marker = 'o', label = "coverage (k=5)")
        plot.plot(x, np.linspace(1.0, 0.14, 11), color = 'black', linestyle = ':', linewidth = 2)
        plot.legend(fontsize = 9)
```
Above test code will result in the following estimates (may fluctuate due to randomness).

```python
{'fidelity': 0.5809355409098216, 'diversity': 0.5653883972468043, 'Top_F1': 0.5730565391609778}
```
