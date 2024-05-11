#
# This file is part of https://github.com/TRAILab/UncertainShapePose
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#


import numpy as np
import scipy
import time

# Sobel sequence for Quasi-Monte Carlo approximation
def sobel_seq(N):
    p = scipy.stats.qmc.Sobol(1, scramble=True, seed=7)
    return p.random(N)

# Slow to initialize so define here
L = 2 ** 7 # 128 points, may decrease if too slow
SSQ = sobel_seq(L)

# Always call confine_0_1 before any beta and logit-normal functions
def confine_0_1(x):
    x = np.array(x)
    if np.isscalar(x):
        if x <= 0:
            x = 1e-9
        elif x >= 1:
            x = 1-1e-9
    else:
        x[np.where(x<=0)] = 1e-9
        x[np.where(x>=1)] = 1-1e-9  
    return x

# Logit function
def logit(x):
    return np.log(x / (1 - x))

# Flipped sigmoid function with k as the slope parameter
def sigmoid(x,k):
    return 1 / (1 + np.exp(k * x))

# PDF of the logit-normal distribution with k as the sigmoid slope parameter
def logit_normal_pdf(x, mu, sigma, k):
    x = confine_0_1(x)
    y = 1.0 / (sigma * np.sqrt(2 * np.pi) * k * x * (1 - x)) * np.exp(-1/2 * ((-logit(x) / k - mu) / sigma) ** 2)
    return y

# CDF of the logit-normal distribution with k as the sigmoid slope parameter
def logit_normal_cdf(x, mu, sigma, k):
    x = confine_0_1(x)
    y = 1 - 1/2 * (1 + scipy.special.erf((-logit(x) / k - mu) / (np.sqrt(2) * sigma)))
    return y

# Inverse CDF of the logit-normal distribution with k as the sigmoid slope parameter
def logit_normal_ppf(x, mu, sigma, k):
    x = confine_0_1(x)
    y = 1 - sigmoid(-k * (np.array([np.sqrt(2)]) * sigma * scipy.special.erfinv(2 * (1 - x) - 1) + mu), 1)
    return y

# Inverse CDF of Gaussian distribution
def norminv(x, mu, sigma):
    return scipy.stats.norm.ppf(x, mu, sigma)

# Inverse CDF of Beta distribution
def betainv(x, alpha, beta):
    return scipy.stats.beta.ppf(x, alpha, beta)

# Inverse CDF of Beta distribution
def logitninv(x, mu, sigma, k):
    return logit_normal_ppf(x, mu, sigma, k)

# Determine the n-th moment of the logit-normal distribution using L sample-MC approximation
def logit_normal_moment_mc(n, L, k, mu, sigma):
    ret = 0.0
    xs = SSQ
    sv = sigmoid(norminv(xs, mu, sigma), k) ** n
    return np.mean(sv)

# Estimate the mean and variance of the logit-normal distribution using L sample-MC approximation
def logit_normal_mean_and_variance_mc(k, mu, sigma):
    mean = logit_normal_moment_mc(1, L, k, mu, sigma)
    var = logit_normal_moment_mc(2, L, k, mu, sigma) - mean ** 2
    return mean, var


# Use truncated sum to approximate the analytical mean of the logit-normal distribution
# https://www.tandfonline.com/doi/epdf/10.1080/03610926.2020.1752723?needAccess=true&role=button
def get_truncation_terms(mu, sigma):
    eps = 1e-4
    n1 = abs(mu / (sigma ** 2) - np.sqrt(2.0 * np.log(1.0 / eps)) / sigma)
    n2 = abs(mu / (sigma ** 2) + np.sqrt(2.0 * np.log(1.0 / eps)) / sigma)
    n3 = np.ceil(0.5 + sigma * np.sqrt(np.log(2.0 / eps) / 2.0) / np.pi)
    n = round(np.max([n1, n2, n3]))
    return np.min([n, 160])

def ep_t1(mu, sigma, n):
    return np.exp(-sigma**2 * n**2 / 2.0) * np.sinh(-n * mu) * np.tanh(n * sigma**2 / 2.0)

def ep_t2(mu, sigma, n):
    y = (2 * n - 1) * np.pi ** 2 / sigma**2
    if y > 150:
        return 0
    return (np.exp(-(2 * n - 1)**2 * np.pi**2 / (2 * sigma**2)) * np.sin(-(2 * n - 1) * np.pi * mu / sigma**2)) / np.sinh(y)

def ep_t3(mu, sigma, n):
    return np.exp(-sigma**2 * n**2 / 2.0) * np.cosh(n * mu)

def dep_t1(mu, sigma, n):
    return -n * np.tanh((n * sigma**2)/2) * np.exp(-(n**2 * sigma**2) / 2.0) * np.cosh(mu * n)

def dep_t2(mu, sigma, n):
    y = (19.739208802178716*n - 9.8696044) / (sigma**2)
    if y > 150:
        return 0
    return -(np.pi * np.exp(-(4.9348022*(2*n - 1)**2) / (sigma**2)) * np.cos((mu * np.pi * (2*n - 1)) / sigma**2) * (2*n - 1)) / (sigma**2 * np.sinh(y))

def dep_t3(mu, sigma, n):
    return n * np.exp(-(n**2 * sigma**2) / 2.0) * np.sinh(mu * n)

def logit_normal_mean_ts(N, k, mu, sigma):
    u = k * mu
    std = k * sigma

    denominator = 1.0
    numerator = 0.0
    for n in range(1,N+1):
        denominator += 2 * ep_t3(u, std, n)
        numerator += ep_t1(u, std, n) + 2 * np.pi / (std ** 2) * ep_t2(u, std, n)
    ret = 0.5 + numerator / denominator
    return ret

def logit_normal_dEp_dmu_ts(N, k, mu, sigma):
    u = k * mu
    std = k * sigma

    denominator_sqrt = 1.0
    numerator_t1 = 0.0
    numerator_t2 = 0.0
    numerator_t3 = 1.0
    numerator_t4 = 0.0
    for n in range(1, N+1):
        denominator_sqrt += 2 * ep_t3(u, std, n)
        numerator_t1 += 2 * dep_t3(u, std, n)
        numerator_t2 += ep_t1(u, std, n) + 2 * np.pi / (std**2) * ep_t2(u, std, n)
        numerator_t3 += 2 * ep_t3(u, std, n)
        numerator_t4 += dep_t1(u, std, n) + 2 * np.pi / (std**2) * dep_t2(u, std, n)
    ret = (numerator_t1 * numerator_t2 - numerator_t3 * numerator_t4) / (denominator_sqrt**2)
    return ret

def logit_normal_mean_and_variance_ts(N, k, mu, sigma):
    mean = logit_normal_mean_ts(N, k, mu, sigma)
    dEp_dmu = logit_normal_dEp_dmu_ts(N, k, mu, sigma)
    var = mean * (1 - mean) - dEp_dmu
    return mean, var

# Approximate the mean and variance of the logit-normal distribution
# Use Truncated Sum or Monte Carlo based on shape of the distribution
def logit_normal_mean_and_variance(k, mu, sigma):
    N_ts = get_truncation_terms(k*mu, k*sigma)
    if N_ts < 160:
        return logit_normal_mean_and_variance_ts(N_ts, k, mu, sigma)
    else:
        return logit_normal_mean_and_variance_mc(k, mu, sigma)

# Estimate the parameters of the beta distribution from the mean and variance
def beta_param_estimator(mu, var):
    alpha = mu * (mu * (1 - mu) / var - 1)
    beta = alpha * (1 - mu) / mu
    return alpha, beta

# Beta distribution
def beta_pdf(x, alpha, beta):
    return scipy.stats.beta.pdf(x, alpha, beta)

# Expected value of the beta distribution
def beta_mean(alpha, beta):
    return alpha / (alpha + beta)

# Variance of the beta distribution
def beta_variance(alpha, beta):
    return alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))

# Product of multiple beta distributions
def beta_product_mc(alphas, betas):
    xs = SSQ
    prods = np.prod(betainv(xs, alphas, betas), 1)
    mu = np.mean(prods)
    variance = np.var(prods)
    return mu, variance

# Product of multiple logit-normal distributions
def logitn_product_mc(k, mus, sigmas):
    xs = SSQ
    prods = np.prod(logitninv(xs, mus, sigmas, k), 1)
    mu = np.mean(prods)
    variance = np.var(prods)
    return mu, variance

# Estimate the parameters of the beta approximation of the product of multiple beta distributions
def beta_product_param_estimator(alphas, betas):
    mu, variance = beta_product_mc(alphas, betas)
    alpha, beta = beta_param_estimator(mu, variance)
    return alpha, beta

# Estimate the parameters of the beta approximation of the product of multiple logit-normal distributions
def logitn_product_beta_param_estimator(k, mus, sigmas):
    mu, variance = logitn_product_mc(k, mus, sigmas)
    alpha, beta = beta_param_estimator(mu, variance)
    return alpha, beta

# Calculate termination probability (Beta) parameters for voxels
# Given the Beta parameters for voxels m = 0, 1, ..., M-1 along the ray
# determine the termination probability parameters for voxels m = 0, 1, ..., M-1, M(passed through)
def termination_probability_params_from_betas(alphas, betas):
    M = len(alphas)

    term_alphas = [alphas[0]]
    term_betas = [betas[0]]
    
    for i in range(1, M):
        term_alphas_i = [alphas[i]]
        term_betas_i = [betas[i]]
        for j in range(0, i):
            term_alphas_i.append(betas[j])
            term_betas_i.append(alphas[j])
        term_alpha_i, term_beta_i = beta_product_param_estimator(term_alphas_i, term_betas_i)

        term_alphas.append(term_alpha_i)
        term_betas.append(term_beta_i)

    term_alpha_end, term_beta_end = beta_product_param_estimator(betas, alphas)
    term_alphas.append(term_alpha_end)
    term_betas.append(term_beta_end)

    return term_alphas, term_betas

# Calculate termination probability (Beta) parameters for voxels
# Given the logit-normal parameters for voxels m = 0, 1, ..., M-1 along the ray
# determine the termination probability parameters for voxels m = 0, 1, ..., M-1, M(passed through)
def termination_probability_params_from_logitn(k, mus, sigmas):
    M = len(mus)

    term_alpha_first, term_beta_first = beta_param_estimator(*logit_normal_mean_and_variance(k, mus[0], sigmas[0]))
    term_alphas = [term_alpha_first]
    term_betas = [term_beta_first]
    
    for i in range(1, M):
        term_mus_i = [mus[i]]
        term_sigmas_i = [sigmas[i]]
        for j in range(0, i):
            term_mus_i.append(-mus[j])
            term_sigmas_i.append(sigmas[j])
        term_alpha_i, term_beta_i = logitn_product_beta_param_estimator(k, term_mus_i, term_sigmas_i)

        term_alphas.append(term_alpha_i)
        term_betas.append(term_beta_i)

    term_alpha_end, term_beta_end = logitn_product_beta_param_estimator(k, -np.array(mus), sigmas)
    term_alphas.append(term_alpha_end)
    term_betas.append(term_beta_end)

    return term_alphas, term_betas

# Calculate the expected value and variance of the rendered depth
def depth_render(depths, alphas, betas):
    N = len(depths)
    E_d = 0.0
    Var_d = 0.0
    means = np.array([beta_mean(alphas[i], betas[i]) for i in range(0, N)])
    variances = np.array([beta_variance(alphas[i], betas[i]) for i in range(0, N)])
    normalizer = np.sum(means)
    means /= normalizer
    variances /= normalizer

    for i in range(0, N):
        E_d += depths[i] * means[i]
        Var_d += depths[i] ** 2 * variances[i]
    return E_d, Var_d



if __name__ == "__main__":

    # Test approximation of the logit-normal distribution

    import matplotlib.pyplot as plt

    sdf_mean = np.array([-0.002])
    sdf_sigma = np.array([0.008])

    k = 400
    # Num sample for Qausi-Monte Carlo approximation, must be power of 2
    

    # sample from gaussian distribution
    N_samples = 10000
    x_samples = np.random.normal(sdf_mean, sdf_sigma, N_samples)

    # Pass through the sigmoid function
    transformed_samples = sigmoid(x_samples,k)

    time_start = time.time()
    logit_normal_mean, logit_normal_variance = logit_normal_mean_and_variance(k, sdf_mean, sdf_sigma)
    time_end = time.time()
    print("Approx time: ", time_end - time_start)

    print("Mean of logit-normal distribution: ", logit_normal_mean)
    print("Variance of logit-normal distribution: ", logit_normal_variance)

    print("Approx time: ", time_end - time_start)

    xs = np.linspace(0.0, 1.0, 1000)

    plt.figure(1)
    plt.hist(transformed_samples, bins=100, density=True)
    plt.plot(xs, beta_pdf(xs, *beta_param_estimator(logit_normal_mean, logit_normal_variance)))
    #plt.plot(xs, beta_pdf(xs, *beta_param_estimator(logit_normal_mean_ts, logit_normal_variance_ts)))
    plt.legend(["Estimated", "Samples"])
    plt.xlabel("Occupancy")
    plt.ylabel("PDF")

    #plt.show()


   

    # Test product of beta distributions
    alphas = [0.6, 3, 7]
    betas = [0.6, 5, 1]

    beta1_samples = np.random.beta(alphas[0], betas[0], N_samples)
    beta2_samples = np.random.beta(alphas[1], betas[1], N_samples)
    beta3_samples = np.random.beta(alphas[2], betas[2], N_samples)
    prod_samples = beta1_samples * beta2_samples * beta3_samples

    time_prod_start = time.time()
    alpha_prod, beta_prod = beta_product_param_estimator(alphas, betas)
    time_prod_end = time.time()
    print("Product of beta duration", time_prod_end - time_prod_start)

    plt.figure(2)
    plt.plot(xs, beta_pdf(xs,alphas[0],betas[0]))
    plt.plot(xs, beta_pdf(xs,alphas[1],betas[1]))
    plt.plot(xs, beta_pdf(xs,alphas[2],betas[2]))
    plt.plot(xs, beta_pdf(xs, alpha_prod, beta_prod))
    plt.hist(beta1_samples, bins=100, density=True, alpha=0.4)
    plt.hist(beta2_samples, bins=100, density=True, alpha=0.4)
    plt.hist(beta3_samples, bins=100, density=True, alpha=0.4)
    plt.hist(prod_samples, bins=100, density=True, alpha=0.4)
    plt.legend(["Beta 1", "Beta 2", "Beta 3", "Product","Beta 1 Samples", "Beta 2 Samples", "Beta 3 Samples", "Product Samples"])
    plt.xlabel("Occupancy")
    plt.ylabel("PDF")

    #plt.show()
    
    # test termination probability
    voxel_alphas = [1,1,2,3,8,9]
    voxel_betas = [12,9,9,4,2,1]

    plt.figure(3)
    for i in range(len(voxel_betas)):
        print(beta_mean(voxel_alphas[i],voxel_betas[i]))
        plt.plot(xs, beta_pdf(xs,voxel_alphas[i],voxel_betas[i]))
    plt.legend(["Beta 1", "Beta 2", "Beta 3", "Beta 4", "Beta 5", "Beta 6"])

    term_alphas, term_betas = termination_probability_params_from_betas(voxel_alphas,voxel_betas)
    plt.figure(4)
    for i in range(len(term_alphas)):
        print(beta_mean(term_alphas[i],term_betas[i]))
        plt.plot(xs, beta_pdf(xs,term_alphas[i],term_betas[i]))
    plt.legend(["Beta 1", "Beta 2", "Beta 3", "Beta 4", "Beta 5", "Beta 6","Beta 7"])


    # test termination probability with logit-normal
    k = 5
    voxel_mus = [0.8,0.6,0.4,0.1,-0.1, -0.3]
    voxel_sigmas= [0.05,0.05,0.05,0.05,0.05,0.05]

    plt.figure(5)
    for i in range(len(voxel_mus)):
        plt.plot(xs, beta_pdf(xs,*beta_param_estimator(*logit_normal_mean_and_variance(k, voxel_mus[i],voxel_sigmas[i]))))
    plt.legend(["Beta 1", "Beta 2", "Beta 3", "Beta 4", "Beta 5", "Beta 6"])

    time_rt_begin = time.time()
    term_alphas, term_betas = termination_probability_params_from_logitn(k,voxel_mus,voxel_sigmas)
    time_rt_end = time.time()
    print("RT runtime duration", time_rt_end - time_rt_begin)
    plt.figure(6)
    for i in range(len(term_alphas)):
        print(beta_mean(term_alphas[i],term_betas[i]))
        plt.plot(xs, beta_pdf(xs,term_alphas[i],term_betas[i]))
    plt.legend(["Beta 1", "Beta 2", "Beta 3", "Beta 4", "Beta 5", "Beta 6","Beta 7"])

    
    # Test product of logit-normal distributions
    k = 400
    mus = [0.008, -0.008, 0.]
    sigmas = [0.0003, 0.0003, 0.0003]

    logitn1_samples = sigmoid(np.random.normal(mus[0], sigmas[0], N_samples), k)
    logitn2_samples = sigmoid(np.random.normal(mus[1], sigmas[1], N_samples), k)
    logitn3_samples = sigmoid(np.random.normal(mus[2], sigmas[2], N_samples), k)
    prod_samples = logitn1_samples * logitn2_samples * logitn3_samples

    time_prod_indirect_start = time.time()
    alphas = []
    betas = []
    for i in range(len(mus)):
        alpha_i, beta_i = beta_param_estimator(*logit_normal_mean_and_variance(k, mus[i], sigmas[i]))
        alphas.append(alpha_i)
        betas.append(beta_i)

    alpha_indirect_est, beta_indirect_est = beta_product_param_estimator(alphas, betas)
    time_prod_indirect_end = time.time()
    alpha_direct_est, beta_direct_est = logitn_product_beta_param_estimator(k, mus, sigmas)
    time_prod_direct_end = time.time()

    print("Direct duration", time_prod_direct_end - time_prod_indirect_end)
    print("Indirect duration", time_prod_indirect_end - time_prod_indirect_start)

    print("Direct Estimated alpha: ", alpha_direct_est)
    print("Direct Estimated beta: ", beta_direct_est)
    print("Indirect Estimated alpha: ", alpha_indirect_est)
    print("Indirect Estimated beta: ", beta_indirect_est)

    plt.figure(7)
    plt.hist(logitn1_samples, bins=100, density=True, alpha=0.4)
    plt.hist(logitn2_samples, bins=100, density=True, alpha=0.4)
    plt.hist(logitn3_samples, bins=100, density=True, alpha=0.4)
    plt.hist(prod_samples, bins=100, density=True, alpha=0.4)

    #plt.plot(xs, logitninv(xs, mus[0], sigmas[0], k))
    #plt.plot(xs, logitninv(xs, mus[1], sigmas[1], k))
    #plt.plot(xs, logitninv(xs, mus[2], sigmas[2], k))

    plt.plot(xs, beta_pdf(xs, alpha_direct_est, beta_direct_est))
    plt.plot(xs, beta_pdf(xs, alpha_indirect_est, beta_indirect_est))

    plt.legend(["Direct", "Indirect", "Logit-Normal 1", "Logit-Normal 2", "Logit-Normal 3", "Product"])

    plt.show()