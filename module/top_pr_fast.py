import multiprocessing
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.spatial import distance

###########################################
#   MultiProcessing KDE
###########################################
def parrallel_score_samples(kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())):
    with multiprocessing.Pool(thread_count) as p:
        return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))

###########################################
#   Fit and score sample KDE
###########################################
def score_sample_KDE(kde, grid: np.array, multiprocess=True):
    """
    :kde KernelDensity: KernelDensity
    :data np_array: data samples
    :grid np_array: grid data
    :multiprocess bool: Default = False, If you use multiprocessing for KDE, please set True
    :returns: p_hat
    """
    if multiprocess:
        kde_results = parrallel_score_samples(kde, grid)
    else:
        kde_results = kde.score_samples(grid)
    p_hat = np.exp(kde_results)
    return p_hat

# Top P&R calculation module
def calculate_tpr(num_real, num_fake, real_features, fake_features, num_fake_on_real, num_real_on_fake):
    # topological precision
    if (num_real ==0):
        num_real = 0.00000000001
        t_precision = min([(len(real_features) * num_fake_on_real) / (len(fake_features) * num_real),1])
    else:
        t_precision = min([(len(real_features) * num_fake_on_real) / (len(fake_features) * num_real),1])

    # topological recall
    if (num_fake == 0):
        num_fake = 0.00000000001
        t_recall = min([(len(fake_features) * num_real_on_fake) / (len(real_features) * num_fake),1])
    else:
        t_recall = min([(len(fake_features) * num_real_on_fake) / (len(real_features) * num_fake),1])
        
    return t_precision, t_recall

###########################################
# Confidence Band
###########################################
def confband_est(data, h, kernel = 'epanechnikov', alpha = .1, repeat = 100, KDE_score_est = True):
    # Set "p_hat = True" to return the estimated p_hat
    # Set "isnumpy = True" to return the not to transform the data into numpy
    # !!! We implement "p_hat" and "isnumpy" options for using this function in Bandwidth Estimator !!! #
    # data as numpy array
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    # p_hat
    # non-compact kernel list = {'gaussian','exponential'} | compact kernel list = {'tophat','epanechnikov','linear','cosine'}
    KDE = KernelDensity(kernel = str(kernel), bandwidth = h)
    kde = KDE.fit(data)
    p_hat = score_sample_KDE(kde, data, multiprocess= True)

    # p_tilde
    theta_star = np.array([])
    for iloop in range(repeat):
        data_bs = data[np.random.choice(np.arange(len(data)), size = len(data), replace = True)]
        kde_tilde = KDE.fit(data_bs)
        p_tilde = score_sample_KDE(kde_tilde, data, multiprocess= True)

        # theta
        theta_star = np.append(theta_star, np.sqrt(len(data))*np.max(np.abs(p_hat-p_tilde)))
    
    # q_alpha
    q_range = np.linspace(min(theta_star), max(theta_star), 5000)
    q_alpha = np.array([])
    for q in q_range:
        if (((1/repeat)*sum(theta_star>=q)) <= alpha):
                q_alpha = np.append(q_alpha, q)
        # treat exceptions (like multi-gaussian with mu=0 and sigma=1)
    if (len(q_alpha) == 0):
        q_alpha = 0
    else:
        q_alpha = np.min(q_alpha)
    
    # confidence band
    if (KDE_score_est == False):
        return q_alpha/np.sqrt(len(data))
    else:
        return q_alpha/np.sqrt(len(data)), kde

###########################################
#   Top P&R
###########################################
def top_pr_fast(real_features, fake_features, f1_score = True, kernel = 'epanechnikov'):
    # non-compact kernel list = {'gaussian','exponential'} | compact kernel list = {'tophat','epanechnikov','linear','cosine'}
    # epanechinikov kernel is known to perform optimaly among the list of choices

    # match real data format
    if not isinstance(real_features, np.ndarray):
        real_features = np.asarray(real_features)

    # match fake data format
    if not isinstance(fake_features, np.ndarray):
        fake_features = np.asarray(fake_features)

    # estimate bandwidth candidates with "balloon estimator" (variable-bandwidth estimation) :: (k = 5, ref. D&C)
    dist = distance.cdist(real_features, real_features, metric = "euclidean")
    dist = dist[:-5,] # ignore last three rows in pairwise distance
    for iloop in range(len(dist)):
        if (iloop == 0):
            balloon_est = np.array(sorted(dist[iloop,(iloop+1):])[4])
        else:
            balloon_est = np.append(balloon_est, sorted(dist[iloop,(iloop+1):])[4])
    balloon_est = sorted(balloon_est)
    # bandwidth = balloon_est[int(len(balloon_est)*0.5)-1] # Median estimated bandwidth
    bandwidth = balloon_est[int(len(balloon_est)*0.95)-1] # 95% percentile estimated bandwidth

    # estimation of confidence band and manifold
    c_r, KDE_r = confband_est(data = real_features, h = bandwidth, kernel = str(kernel))
    c_g, KDE_g = confband_est(data = fake_features, h = bandwidth, kernel = str(kernel))

    # count significant real samples on real manifold
    num_real = 0
    score_rr = score_sample_KDE(KDE_r, real_features, multiprocess= True)
    num_real = (score_rr > c_r).sum()

    # count significant fake samples on real manifold
    num_fake_on_real = 0
    score_gr = score_sample_KDE(KDE_r, fake_features, multiprocess= True)
    num_fake_on_real = (score_gr > c_r).sum()
    
    # count significant fake samples on fake manifold
    num_fake = 0
    score_gg = score_sample_KDE(KDE_g, fake_features, multiprocess= True)
    num_fake = (score_gg > c_g).sum()

    # count significant real samples on fake manifold
    num_real_on_fake = 0
    score_rg = score_sample_KDE(KDE_g, real_features, multiprocess= True)
    num_real_on_fake = (score_rg > c_g).sum()

    # topological precision and recall
    t_precision, t_recall = calculate_tpr(num_real, num_fake, real_features, fake_features, num_fake_on_real, num_real_on_fake)

    # top f1-score
    if (f1_score == True and (t_precision != 0 or t_recall != 0)):
        if (t_precision != 0 and t_precision != 0):
            F1_score = 2/((1/t_precision) + (1/t_recall))
        elif (t_precision == 0 and t_recall != 0):
            F1_score = 2/(1/t_recall)
        else:
            F1_score = 2/(1/t_precision)
        return dict(fidelity = t_precision, diversity = t_recall, Top_F1 = F1_score)
    else:
        return dict(fidelity = t_precision, diversity = t_recall)
