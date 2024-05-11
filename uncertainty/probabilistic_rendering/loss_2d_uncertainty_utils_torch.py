##########################
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

import math
import scipy
import time
import torch
import numpy as np


# device = torch.device("cpu") # macbook uses mps
device = torch.device("cuda") # macbook uses mps
cpu_device = torch.device("cpu")
dtype = torch.float64

min_val = 1e-20

BACKGROUND_DEPTH = 9.0

# Sobel sequence for Quasi-Monte Carlo approximation
def sobel_seq(N):
    p = scipy.stats.qmc.Sobol(1, scramble=True, seed=7)
    return p.random(N)

# Slow to initialize so define here
L = 2 ** 7 # 128 points, may decrease if too slow
SSQ = torch.reshape(torch.tensor(sobel_seq(L), device=device, dtype=dtype, requires_grad = False), (1,L))

# Always call confine_0_1 before any beta and logit-normal functions
def confine_0_1(x):
    eps = 1e-9
    return torch.clip(x, eps, 1 - eps)

# Logit function
def logit(x):
    return torch.log(x / (1 - x))

# Flipped sigmoid function with k as the slope parameter
def sigmoid(x,k):
    return 1 / (1 + torch.exp(k * x))

# PDF of the logit-normal distribution with k as the sigmoid slope parameter
def logit_normal_pdf(x, mu, sigma, k):
    y = 1.0 / (sigma * math.sqrt(2 * math.pi) * k * x * (1 - x)) * torch.exp(-1/2 * ((-logit(x) / k - mu) / sigma) ** 2)
    return y

# CDF of the logit-normal distribution with k as the sigmoid slope parameter
def logit_normal_cdf(x, mu, sigma, k):
    y = 1 - 1/2 * (1 + torch.special.erf((-logit(x) / k - mu) / (math.sqrt(2) * sigma)))
    return y

# Inverse CDF of the logit-normal distribution with k as the sigmoid slope parameter
def logit_normal_ppf(x, mu, sigma, k):
    mu = mu.reshape((len(mu),len(mu[0]),1))
    sigma = sigma.reshape((len(mu),len(mu[0]),1))
    mus = mu.repeat(1, 1, x.shape[2]).reshape((mu.shape[0], mu.shape[1], L))
    sigmas = sigma.repeat(1, 1, x.shape[2]).reshape((mu.shape[0], mu.shape[1], L))
    y = 1 - sigmoid(-k * (math.sqrt(2) * sigmas * torch.erfinv(2 * (1 - x) - 1) + mus), 1)
    return y

# Inverse CDF of Gaussian distribution
def norminv(x, mu, sigma):
    return mu.reshape((len(mu),1)) + sigma.reshape((len(mu),1)) * math.sqrt(2) * torch.erfinv(2 * x - 1)

# Inverse CDF of Beta distribution, not right...
'''
def betainv(x, alpha, beta):
    alphas = alpha.reshape((-1,1)).repeat(1, x.shape[1])
    betas = beta.reshape((-1,1)).repeat(1, x.shape[1])
    y1 = alphas * torch.exp(torch.lgamma(alphas + betas) - torch.lgamma(alphas) - torch.lgamma(betas) + torch.log(x))
    y2 = betas * torch.exp(torch.lgamma(alphas + betas) - torch.lgamma(alphas) - torch.lgamma(betas) + torch.log(1 - x))
    ret = y1 + y2
    return  ret
'''

# Inverse CDF of Logit-Normal distribution
def logitninv(x, mu, sigma, k):
    return logit_normal_ppf(x, mu, sigma, k)

# Determine the n-th moment of the logit-normal distribution using L sample-MC approximation
def logit_normal_moment_mc(n, L, k, mu, sigma):
    xs = SSQ.repeat(mu.shape[0], 1)
    sv = sigmoid(norminv(xs, mu, sigma), k) ** n
    return torch.mean(sv, dim=1)

# Estimate the mean and variance of the logit-normal distribution using L sample-MC approximation
def logit_normal_mean_and_variance_mc(k, mu, sigma):
    mean = logit_normal_moment_mc(1, L, k, mu, sigma)
    var = logit_normal_moment_mc(2, L, k, mu, sigma) - mean ** 2
    return mean, var


# Use truncated sum to approximate the analytical mean of the logit-normal distribution
# https://www.tandfonline.com/doi/epdf/10.1080/03610926.2020.1752723?needAccess=true&role=button
def get_truncation_terms(mu, sigma):
    eps = torch.tensor(1e-4, device=device, dtype=dtype)
    eps.requires_grad = False
    n1 = torch.abs(mu / (sigma ** 2) - torch.sqrt(2.0 * torch.log(1.0 / eps)) / sigma)
    n2 = torch.abs(mu / (sigma ** 2) + torch.sqrt(2.0 * torch.log(1.0 / eps)) / sigma)
    n3 = torch.ceil(0.5 + sigma * torch.sqrt(torch.log(2.0 / eps) / 2.0) / math.pi)

    candidates_f, _ = torch.max(torch.cat((n1,n2,n3), dim=0).reshape(3,-1), dim=0)
    candidates = candidates_f.ceil().int()

    max_default =  torch.tensor([160], device=device, dtype=dtype).int()
    Ns = torch.min(candidates, max_default)
    N =  torch.min(torch.max(Ns), max_default)

    return N, Ns

def ep_t1(mu, sigma, n):
    return torch.nan_to_num(torch.exp(-sigma**2 * n**2 / 2.0) * torch.sinh(-n * mu) * torch.tanh(n * sigma**2 / 2.0))

def ep_t2(mu, sigma, n):
    y = (2 * n - 1) * torch.pi ** 2 / sigma**2
    return torch.nan_to_num((torch.exp(-(2 * n - 1)**2 * math.pi**2 / (2 * sigma**2)) * torch.sin(-(2 * n - 1) * math.pi * mu / sigma**2)) / torch.sinh(y))

def ep_t3(mu, sigma, n):
    return torch.nan_to_num(torch.exp(-sigma**2 * n**2 / 2.0) * torch.cosh(n * mu))

def dep_t1(mu, sigma, n):
    return torch.nan_to_num(-n * torch.tanh((n * sigma**2) / 2) * torch.exp(-(n**2 * sigma**2) / 2.0) * torch.cosh(mu * n))

def dep_t2(mu, sigma, n):
    y = (19.739208802178716*n - 9.8696044) / (sigma**2)
    return torch.nan_to_num(-(math.pi * torch.exp(-(4.9348022*(2*n - 1)**2) / (sigma**2)) * torch.cos((mu * math.pi * (2*n - 1)) / sigma**2) * (2*n - 1)) / (sigma**2 * torch.sinh(y)))

def dep_t3(mu, sigma, n):
    return torch.nan_to_num(n * torch.exp(-(n**2 * sigma**2) / 2.0) * torch.sinh(mu * n))

def logit_normal_mean_ts(N, Ns, k, mu, sigma):
    u = k * mu
    std = k * sigma

    denominator = torch.tensor([1.0], device=device, dtype=dtype).repeat(mu.shape[0],)
    numerator = torch.tensor([0.0], device=device, dtype=dtype).repeat(mu.shape[0],)
    for n in range(1,N.item()+1):
        indicator = ((Ns-n) >= 0).int()
        denominator += indicator * 2 * ep_t3(u, std, n)
        numerator += indicator * (ep_t1(u, std, n) + 2 * math.pi / (std ** 2) * ep_t2(u, std, n))
    ret = torch.tensor(0.5, device=device, dtype=dtype) + numerator / denominator
    return ret

def logit_normal_dEp_dmu_ts(N, Ns, k, mu, sigma):
    u = k * mu
    std = k * sigma

    denominator_sqrt = torch.tensor([1.0], device=device, dtype=dtype).repeat(mu.shape[0],)
    numerator_t1 = torch.tensor([0.0], device=device, dtype=dtype).repeat(mu.shape[0],)
    numerator_t2 = torch.tensor([0.0], device=device, dtype=dtype).repeat(mu.shape[0],)
    numerator_t3 = torch.tensor([1.0], device=device, dtype=dtype).repeat(mu.shape[0],)
    numerator_t4 = torch.tensor([0.0], device=device, dtype=dtype).repeat(mu.shape[0],)
    for n in range(1, N.item()+1):
        indicator = ((Ns-n) >= 0).int() 
        denominator_sqrt += indicator * 2 * ep_t3(u, std, n)
        numerator_t1 += indicator * 2 * dep_t3(u, std, n)
        numerator_t2 +=  indicator * (ep_t1(u, std, n) + 2 * math.pi / (std**2) * ep_t2(u, std, n))
        numerator_t3 += indicator * 2 * ep_t3(u, std, n)
        numerator_t4 +=  indicator * (dep_t1(u, std, n) + 2 * math.pi / (std**2) * dep_t2(u, std, n))
    ret = (numerator_t1 * numerator_t2 - numerator_t3 * numerator_t4) / (denominator_sqrt**2)
    return ret

def logit_normal_mean_and_variance_ts(N, Ns, k, mu, sigma):
    mean = logit_normal_mean_ts(N, Ns, k, mu, sigma)
    dEp_dmu = logit_normal_dEp_dmu_ts(N, Ns, k, mu, sigma)
    var = mean * (1 - mean) - dEp_dmu
    return mean, var

# Approximate the mean and variance of the logit-normal distribution
# Use Truncated Sum or Monte Carlo based on shape of the distribution
def logit_normal_mean_and_variance(k, mu, sigma):
    #N_ts, Ns_ts = get_truncation_terms(k*mu, k*sigma)
    #if N_ts < 160:
    #    return logit_normal_mean_and_variance_ts(N_ts, Ns_ts, k, mu, sigma) # how to use it in batch?
    #else:
    return logit_normal_mean_and_variance_mc(k, mu, sigma)

# Estimate the parameters of the beta distribution from the mean and variance
def beta_param_estimator(mu, var):
    alpha = mu * (mu * (1 - mu) / var - 1)
    beta = alpha * (1 - mu) / mu
    return alpha, beta

# Beta distribution
def beta_pdf(x, alpha, beta):
    return torch.exp(torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta) + (alpha - 1) * torch.log(x) + (beta - 1) * torch.log(1 - x))

# Expected value of the beta distribution
def beta_mean(alpha, beta):
    y = alpha / (alpha + beta)
    return y

# Variance of the beta distribution
def beta_variance(alpha, beta):
    y = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
    return y

# Product of multiple beta distributions
'''
def beta_product_mc(alphas, betas):
    xs = SSQ.repeat(alphas.shape[0], 1)
    prods = torch.prod(betainv(xs, alphas, betas), dim=0) # how to implement betainv?
    mu = torch.mean(prods)
    variance = torch.var(prods)
    return mu, variance
'''

# Product of multiple logit-normal distributions
def logitn_product_mc(k, mus, sigmas):
    xs = SSQ.repeat(mus.shape[0] * mus.shape[1], 1).reshape((mus.shape[0], mus.shape[1], L)) 
    prods = torch.prod(logitninv(xs, mus, sigmas, k), dim=1)
    mu = torch.mean(prods, axis=1)
    variance = torch.var(prods, axis=1)

    open_clamp = True
    if open_clamp:
        variance_out = torch.clamp(variance, min_val, 0.25-min_val)
    else:
        variance_out = variance

    return torch.clamp(mu, min_val, 1-min_val), variance_out

# Estimate the parameters of the beta approximation of the product of multiple beta distributions
'''
def beta_product_param_estimator(alphas, betas):
    mu, variance = beta_product_mc(alphas, betas)
    alpha, beta = beta_param_estimator(mu, variance)
    return alpha, beta
'''

# Estimate the parameters of the beta approximation of the product of multiple logit-normal distributions
def logitn_product_beta_param_estimator(k, mus, sigmas):
    mu, variance = logitn_product_mc(k, mus, sigmas)
    alpha, beta = beta_param_estimator(mu, variance)
    return alpha, beta

# Calculate termination probability (Beta) parameters for voxels
# Given the Beta parameters for voxels m = 0, 1, ..., M-1 along the ray
# determine the termination probability parameters for voxels m = 0, 1, ..., M-1, M(passed through)
'''
def termination_probability_params_from_betas(alphas, betas):
    M = len(alphas)

    term_alphas = torch.tensor([alphas[0]])
    term_betas = torch.tensor([betas[0]])
    
    for i in range(1, M):
        term_alphas_i = torch.tensor([alphas[i]])
        term_betas_i = torch.tensor([betas[i]])
        for j in range(0, i):
            term_alphas_i = torch.cat((term_alphas_i, betas[j]))
            term_betas_i = torch.cat((term_betas_i, alphas[j]))
        term_alpha_i, term_beta_i = beta_product_param_estimator(term_alphas_i, term_betas_i)

        term_alphas = torch.cat((term_alphas, term_alpha_i))
        term_betas = torch.cat((term_betas, term_beta_i))

    term_alpha_end, term_beta_end = beta_product_param_estimator(betas, alphas)
    term_alphas = torch.cat((term_alphas, term_alpha_end))
    term_betas = torch.cat((term_betas, term_beta_end))

    return term_alphas, term_betas
'''

# Calculate termination probability (Beta) parameters for voxels
# Given the logit-normal parameters for voxels m = 0, 1, ..., M-1 along the ray
# determine the termination probability parameters for voxels m = 0, 1, ..., M-1, M(passed through)
'''
def termination_probability_betas_from_logitn(k, mus, sigmas):
    M = len(mus[0])
    N = len(mus)

    term_alpha_first, term_beta_first = beta_param_estimator(*logit_normal_mean_and_variance(k, mus[:,0], sigmas[:,1]))

    term_alphas = term_alpha_first.reshape((N,1))
    term_betas = term_beta_first.reshape((N,1))
    
    for i in range(1, M):
        term_mus_i = mus[:,i].reshape((N,1))
        term_sigmas_i = sigmas[:,i].reshape((N,1))
        for j in range(0, i):
            term_mus_i = torch.hstack((term_mus_i, -mus[:,j].reshape((N,1))))
            term_sigmas_i = torch.hstack((term_sigmas_i, sigmas[:,j].reshape((N,1))))
        term_alpha_i, term_beta_i = logitn_product_beta_param_estimator(k, term_mus_i, term_sigmas_i)

        term_alphas = torch.hstack((term_alphas, term_alpha_i.reshape((N,1))))
        term_betas = torch.hstack((term_betas, term_beta_i.reshape((N,1))))

    term_alpha_end, term_beta_end = logitn_product_beta_param_estimator(k, -mus, sigmas)
    term_alphas = torch.hstack((term_alphas, term_alpha_end.reshape((N,1))))
    term_betas = torch.hstack((term_betas, term_beta_end.reshape((N,1))))

    return term_alphas, term_betas
'''

# Calculate the expected value and variance of the rendered depth
'''
def depth_render_from_beta(depths, alphas, betas):
    N = len(depths)

    means = beta_mean(alphas, betas)
    variances = beta_variance(alphas, betas)

    mask = torch.isnan(means) | torch.isinf(means) | torch.isnan(variances) | torch.isinf(variances) #| (depths >= BACKGROUND_DEPTH-0.1)

    means[mask] = 0
    variances[mask] = 0

    normalizer = torch.sum(means, dim=1).reshape((N,1))
    means /= normalizer
    variances /= normalizer

    E_d = torch.sum(depths * means, dim=1)
    Var_d = torch.sum(depths ** 2 * variances, dim=1)

    return E_d, Var_d
'''

# determine the termination probability parameters for voxels m = 0, 1, ..., M-1, M(passed through)
def termination_probability_params_from_logitn(k, mus, sigmas):
    M = len(mus[0])
    N = len(mus)

    term_mus_first, term_vars_first = logit_normal_mean_and_variance(k, mus[:,0], sigmas[:,1])

    term_mus = term_mus_first.reshape((N,1))
    term_vars = term_vars_first.reshape((N,1))
    
    for i in range(1, M):
        term_mus_i = mus[:,i].reshape((N,1))
        term_sigmas_i = sigmas[:,i].reshape((N,1))
        for j in range(0, i):
            term_mus_i = torch.hstack((term_mus_i, -mus[:,j].reshape((N,1))))
            term_sigmas_i = torch.hstack((term_sigmas_i, sigmas[:,j].reshape((N,1))))
        term_mus_i, term_vars_i = logitn_product_mc(k, term_mus_i, term_sigmas_i)

        term_mus = torch.hstack((term_mus, term_mus_i.reshape((N,1))))
        term_vars = torch.hstack((term_vars, term_vars_i.reshape((N,1))))

    term_mus_end, term_vars_end = logitn_product_mc(k, -mus, sigmas)
    term_mus = torch.hstack((term_mus, term_mus_end.reshape((N,1))))
    term_vars = torch.hstack((term_vars, term_vars_end.reshape((N,1))))

    return term_mus, term_vars

# Calculate the expected value and variance of the rendered depth
def depth_render(depths, mus, variances):
    N = len(depths)

    normalizer = torch.sum(mus, dim=1).reshape((N,1))
    mus /= normalizer
    variances /= normalizer

    E_d = torch.sum(depths * mus, dim=1)
    Var_d = torch.sum(depths ** 2 * variances, dim=1)

    return E_d, Var_d


if __name__ == "__main__":

    # Test approximation of the logit-normal distribution

    import matplotlib.pyplot as plt

    sdf_mean = torch.tensor([0.003, 0, 0.2], device=device, dtype=dtype)
    sdf_sigma = torch.tensor([0.002, 0.01, 0.2], device=device, dtype=dtype)

    k = 400

    # sample from gaussian distribution
    N_samples = 10000
    #x_samples = torch.normal(sdf_mean.item(), sdf_sigma.item(), size=(N_samples,))

    # Pass through the sigmoid function
    #transformed_samples = sigmoid(x_samples,k)

    time_mc_start = time.time()
    ln_mean_mc, ln_variance_mc = logit_normal_mean_and_variance_mc(k, sdf_mean, sdf_sigma)
    time_mc_end = time.time()

    N_ts, Ns_ts = get_truncation_terms(k*sdf_mean, k*sdf_sigma)
    print("N_ts: ", N_ts)
    print("Ns_ts: ", Ns_ts)
    ln_mean_ts, ln_variance_ts = logit_normal_mean_and_variance_ts(N_ts, Ns_ts, k, sdf_mean, sdf_sigma)
    time_ts_end = time.time()



    print("MC approx time: ", time_mc_end - time_mc_start)
    print("MC Mean of logit-normal distribution: ", ln_mean_mc)
    print("MC Variance of logit-normal distribution: ", ln_variance_mc)

    print("TS approx time: ", time_ts_end - time_mc_end)
    print("TS Mean of logit-normal distribution: ", ln_mean_ts)
    print("TS Variance of logit-normal distribution: ", ln_variance_ts)


    xs = torch.linspace(0.0, 1.0, 1000)
    xs = confine_0_1(xs)

    '''
    plt.figure(1)
    plt.hist(transformed_samples, bins=100, density=True)
    plt.plot(xs, beta_pdf(xs, *beta_param_estimator(logit_normal_mean_mc, logit_normal_variance_mc)))
    plt.plot(xs, beta_pdf(xs, *beta_param_estimator(logit_normal_mean_ts, logit_normal_variance_ts)))
    plt.legend(["Estimated MC", "Estimated TS", "Samples"])
    plt.xlabel("Occupancy")
    plt.ylabel("PDF")
    '''

    #plt.show()

    
   
    '''
    # Test product of beta distributions
    alphas = torch.tensor([0.6, 3, 7])
    betas = torch.tensor([0.6, 5, 1])

    dists = torch.distributions.beta.Beta(alphas, betas)
    samples = dists.sample((N_samples,))

    beta1_samples = samples[:,0]
    beta2_samples = samples[:,1]
    beta3_samples = samples[:,2]
    prod_samples = beta1_samples * beta2_samples * beta3_samples

    time_prod_start = time.time()
    alpha_prod, beta_prod = beta_product_param_estimator(alphas, betas)
    time_prod_end = time.time()

    print("Product of beta alpha: ", alpha_prod)
    print("Product of beta beta: ", beta_prod)
    print("Duration", time_prod_end - time_prod_start)

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
    '''


    # Test product of logit-normal distributions
    k = 400
    #mus = torch.tensor([[0.008, -0.008, 0.],[0.008, -0.008, 0.]])
    #sigmas = torch.tensor([[0.0003, 0.0003, 0.0003],[0.0003, 0.0003, 0.0003]])

    mus = torch.tensor([[0.008, -0.008, 0.]], device=device, dtype=dtype)
    sigmas = torch.tensor([[0.0003, 0.0003, 0.0003]], device=device, dtype=dtype)

    #logitn1_samples = sigmoid(torch.tensor(np.random.normal(mus[0], sigmas[0], N_samples)), k)
    #logitn2_samples = sigmoid(torch.tensor(np.random.normal(mus[1], sigmas[1], N_samples)), k)
    #logitn3_samples = sigmoid(torch.tensor(np.random.normal(mus[2], sigmas[2], N_samples)), k)
    #prod_samples = logitn1_samples * logitn2_samples * logitn3_samples

    time_prod_direct_start = time.time()
    alpha_direct_est, beta_direct_est = logitn_product_beta_param_estimator(k, mus, sigmas)
    time_prod_direct_end = time.time()

    print("Direct duration", time_prod_direct_end - time_prod_direct_start)
    print("Direct Estimated alpha: ", alpha_direct_est)
    print("Direct Estimated beta: ", beta_direct_est)

    #plt.figure(7)
    #plt.hist(logitn1_samples, bins=100, density=True, alpha=0.4)
    #plt.hist(logitn2_samples, bins=100, density=True, alpha=0.4)
    #plt.hist(logitn3_samples, bins=100, density=True, alpha=0.4)
    #plt.hist(prod_samples, bins=100, density=True, alpha=0.4)

    ##plt.plot(xs, logitninv(xs, mus[0], sigmas[0], k))
    ##plt.plot(xs, logitninv(xs, mus[1], sigmas[1], k))
    ##plt.plot(xs, logitninv(xs, mus[2], sigmas[2], k))

    #plt.plot(xs, beta_pdf(xs, alpha_direct_est, beta_direct_est))

    #plt.legend(["Direct", "Logit-Normal 1", "Logit-Normal 2", "Logit-Normal 3", "Product"])

    #plt.show()


    '''
    
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

    '''
    # test termination probability with logit-normal
    k = 5


    voxel_mus = torch.tensor([[0.8,0.6,0.4,0.1,-0.1, -0.3],[0.7,0.5,0.3,0.0,-0.2, -0.3]], device=device, dtype=dtype)
    voxel_sigmas= torch.tensor([[0.05,0.05,0.05,0.05,0.05,0.05],[0.05,0.05,0.05,0.05,0.05,0.05]], device=device, dtype=dtype)
    voxel_depths = torch.tensor([[1.1,1.2,1.3,1.4,1.5,1.6],[1.1,1.2,1.3,1.4,1.5,1.6]], device=device, dtype=dtype)

    time_rt_begin = time.time()
    term_mus, term_sigmas = termination_probability_params_from_logitn(k,voxel_mus,voxel_sigmas)
    time_rt_end = time.time()
    print("RT runtime duration", time_rt_end - time_rt_begin)

    for n in range(len(term_mus)):
        for i in range(len(term_mus[0])):
            print("Termination Prob for ray", n, "at index", i, "is", term_mus[n][i].item())
        print('\n')


    E_d, Var_d = depth_render(voxel_depths, term_mus[:,:-1], term_sigmas[:,:-1])
    print("Expected depth", E_d)
    print("Variance of depth", Var_d)