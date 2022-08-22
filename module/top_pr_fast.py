import multiprocessing
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.spatial import distance
import torch

###########################################
#   Automatic Grid Search
###########################################
def set_grid(data):
    import numpy as np

    # find min max
    dim = len(data[0])
    mins = np.array([])
    maxs = np.array([])
    for dims in range(dim):
        mins = np.append(mins, min(data[:,dims]))
        maxs = np.append(maxs, max(data[:,dims]))
    
    # set grid
    # 2 dimensional data
    if (len(mins) == 2):
        xval = np.linspace(mins[0], maxs[0], 1000)
        yval = np.linspace(mins[1], maxs[1], 1000)
        positions = np.array([[u,v] for u in xval for v in yval])
    # 3 dimensional data
    elif (len(mins) == 3):
        xval = np.linspace(mins[0], maxs[0], 100)
        yval = np.linspace(mins[1], maxs[1], 100)
        zval = np.linspace(mins[2], maxs[2], 100)
        positions = np.array([[u,v,k] for u in xval for v in yval for k in zval])
    
    return positions

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
    #if (len(grid[0]) > 50): # if data dimension is greater than 50
    #    p_hat = np.exp(-int(np.max(kde_results)) + kde_results)
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
        t_recall = (len(fake_features) * num_real_on_fake) / (len(real_features) * num_fake)
    else:
        t_recall = (len(fake_features) * num_real_on_fake) / (len(real_features) * num_fake)
        
    return t_precision, t_recall

###########################################
# Confidence Band
###########################################
def confband_est(data, h, kernel = 'epanechnikov', alpha = 0.2, repeat = 25, KDE_score_est = True, multiprocess = True):
    # Set "p_hat = True" to return the estimated p_hat
    # Set "isnumpy = True" to return the not to transform the data into numpy
    # !!! We implement "p_hat" and "isnumpy" options for using this function in Bandwidth Estimator !!! #
    # data as numpy array
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    # non-compact kernel list = {'gaussian','exponential'} | compact kernel list = {'tophat','epanechnikov','linear','cosine'}
    KDE = KernelDensity(kernel = str(kernel), bandwidth = h)
    kde = KDE.fit(data)
    
    if (len(data[0]) <= 3):
        positions = set_grid(data)
        p_hat = score_sample_KDE(kde, positions, multiprocess = str(multiprocess))
        # p_tilde
        theta_star = np.array([])
        for iloop in range(repeat):
            # to only take account the outlying samples, we use the np.unique function (significant point with duplicate will result in a highly estimated cn)
            data_bs = data[np.random.choice(np.arange(len(data)), size = len(data), replace = True)]
            kde_tilde = KDE.fit(data_bs)
            p_tilde = score_sample_KDE(kde_tilde, positions, multiprocess= str(multiprocess))

            # theta
            theta_star = np.append(theta_star, np.sqrt(len(data))*np.max(np.abs(p_hat-p_tilde)))
    
        # q_alpha
        # past code
        #q_range = np.linspace(np.min(theta_star), np.max(theta_star), 10000)
        #q_alpha = np.array([])
        #for q in q_range:
        #    if (((1/repeat)*sum(theta_star>=q)) <= alpha):
        #            q_alpha = np.append(q_alpha, q)

        # refined code
        q_alpha = np.quantile(theta_star, 1 - alpha)

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

    else:
        p_hat = score_sample_KDE(kde, data, multiprocess = str(multiprocess))
        index = round(len(p_hat) * alpha)
        q_alpha = sorted(p_hat)[index]

        # confidence band
        if (KDE_score_est == False):
            return q_alpha
        else:
            return q_alpha, kde

###########################################
#   Top P&R
###########################################
def top_pr_fast(real_features, fake_features, alpha = 0.1, f1_score = True, kernel = 'epanechnikov', multiprocess = True):
    # non-compact kernel list = {'gaussian','exponential'} | compact kernel list = {'tophat','epanechnikov','linear','cosine'}
    # epanechinikov kernel is known to perform optimaly among the list of choices

    # match real data format
    if not isinstance(real_features, np.ndarray):
        real_features = np.asarray(real_features)

    # match fake data format
    if not isinstance(fake_features, np.ndarray):
        fake_features = np.asarray(fake_features)

    # random projection
    if (len(real_features[0]) > 3):
        projection = torch.nn.Linear(len(real_features[0]), 32, bias = False).eval()
        torch.nn.init.kaiming_normal_(projection.weight)
        for param in projection.parameters():
            param.requires_grad_(False)
        real_features = projection(torch.Tensor(real_features)).detach().numpy()
        fake_features = projection(torch.Tensor(fake_features)).detach().numpy()

    # estimate bandwidth for real with "balloon estimator" (variable-bandwidth estimation)
    k = 50
    dist = distance.cdist(real_features, real_features, metric = "euclidean")
    dist = dist[:-k,] # ignore last three rows in pairwise distance
    for iloop in range(len(dist)):
        if (iloop == 0):
            balloon_est = np.array(sorted(dist[iloop,(iloop+1):])[k-1])
        else:
            balloon_est = np.append(balloon_est, sorted(dist[iloop,(iloop+1):])[k-1])
    balloon_est = sorted(balloon_est)
    bandwidth_r = balloon_est[int(len(balloon_est)*0.5)-1] # median balloon estimated bandwidth

    # estimate bandwidth for fake with "balloon estimator" (variable-bandwidth estimation)
    dist = distance.cdist(fake_features, fake_features, metric = "euclidean")
    dist = dist[:-k,] # ignore last three rows in pairwise distance
    for iloop in range(len(dist)):
        if (iloop == 0):
            balloon_est = np.array(sorted(dist[iloop,(iloop+1):])[k-1])
        else:
            balloon_est = np.append(balloon_est, sorted(dist[iloop,(iloop+1):])[k-1])
    balloon_est = sorted(balloon_est)
    bandwidth_f = balloon_est[int(len(balloon_est)*0.5)-1] # median balloon estimated bandwidth

    # estimation of confidence band and manifold
    c_r, KDE_r = confband_est(data = real_features, h = bandwidth_r, kernel = str(kernel), alpha = alpha, multiprocess= str(multiprocess))
    c_g, KDE_g = confband_est(data = fake_features, h = bandwidth_f, kernel = str(kernel), alpha = alpha, multiprocess= str(multiprocess))

    # count significant real samples on real manifold
    num_real = 0
    score_rr = score_sample_KDE(KDE_r, real_features, multiprocess= str(multiprocess))
    num_real = (score_rr > c_r).sum()

    # count significant fake samples on real manifold
    num_fake_on_real = 0
    score_gr = score_sample_KDE(KDE_r, fake_features, multiprocess= str(multiprocess))
    num_fake_on_real = (score_gr > c_r).sum()
    
    # count significant fake samples on fake manifold
    num_fake = 0
    score_gg = score_sample_KDE(KDE_g, fake_features, multiprocess= str(multiprocess))
    num_fake = (score_gg > c_g).sum()

    # count significant real samples on fake manifold
    num_real_on_fake = 0
    score_rg = score_sample_KDE(KDE_g, real_features, multiprocess= str(multiprocess))
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

###########################################
#   Outlier Detector
###########################################
def outlier_detector(features, alpha = 0.1, kernel = 'epanechnikov', multiprocess = True):
    # match data format
    if not isinstance(features, np.ndarray):
        features = np.asarray(features)

    # random projection
    if (len(features[0]) > 3):
        projection = torch.nn.Linear(len(features[0]), 32, bias = False).eval()
        torch.nn.init.kaiming_normal_(projection.weight)
        for param in projection.parameters():
            param.requires_grad_(False)
        features = projection(torch.Tensor(features)).detach().numpy()

    # estimate bandwidth for real with "balloon estimator" (variable-bandwidth estimation)
    k = 100
    dist = distance.cdist(features, features, metric = "euclidean")
    dist = dist[:-k,] # ignore last three rows in pairwise distance
    for iloop in range(len(dist)):
        if (iloop == 0):
            balloon_est = np.array(sorted(dist[iloop,(iloop+1):])[k-1])
        else:
            balloon_est = np.append(balloon_est, sorted(dist[iloop,(iloop+1):])[k-1])
    balloon_est = sorted(balloon_est)
    bandwidth = balloon_est[int(len(balloon_est)*0.5)-1] # median balloon estimated bandwidth

    # estimation of confidence band and manifold
    cn, KDE = confband_est(data = features, h = bandwidth, kernel = str(kernel), alpha = alpha, multiprocess= str(multiprocess))
    sample_density = score_sample_KDE(KDE, features, multiprocess= str(multiprocess))

    # outlier detection
    outlier = np.array([])
    for iloop in range(len(sample_density)):
        if (sample_density[iloop] < cn):
            outlier = np.append(outlier, iloop)
    return list(map(int, outlier)), cn