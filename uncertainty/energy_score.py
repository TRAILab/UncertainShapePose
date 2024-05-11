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

from numpy import full
import torch

def get_mu_sigma(outputs):
    len_code = round(outputs.size()[1] / 2)
    # half value
    value = outputs[:,:len_code]

    # alpha = log(sigma^2)
    alpha = outputs[:,len_code:]
    # e_neg_alpha = torch.exp(-alpha)

    var = torch.exp(alpha)

    return value, var

def generate_random_vector(M_sample, mu, var):
    # 1. generate a random vector, whose i-th dimension is sampled from N(0,var[i])
    size_mu = mu.shape[0]
    device = mu.device
    guassian_norm = torch.randn((M_sample,size_mu)).to(device)
    sigma = torch.sqrt(var)
    vec = guassian_norm.multiply(sigma) + mu
    return vec

def visualize_check_sample(Z, mu, var):

    for dim in range(Z.shape[1]):
        print('for dim:', dim)

        random_z = Z[:,dim]
        sample_mean = torch.mean(random_z)
        sample_var = torch.var(random_z)

        print('mean: %f/%f' %(sample_mean, mu[dim]))
        print('var: %f/%f' %(sample_var, var[dim]))


# the MC approximation version
def loss_energy_score(outputs,Y,M_sample=1000):
    mu_list, var_list = get_mu_sigma(outputs)
    N = mu_list.shape[0]

    # sample M times from distribution N(mu,sigma)
    full_term = 0
    for i in range(N):
        mu = mu_list[i]
        var = var_list[i]

        # sample M times
        Z = generate_random_vector(M_sample, mu, var)

        # visualize specific dimensions of the vector
        # visualize_check_sample(Z, mu, var)

        zn = Y[i]
        term1_s1 = Z - zn
        term1_s2 = torch.linalg.norm(term1_s1, dim=1)
        term1_s3 = torch.sum(term1_s2)

        term1 =  term1_s3 / M_sample # check axis
        
        term2 = 0
        for n in range(M_sample-1):
            term2 += (Z[n] - Z[n+1]).norm()
        term2 = term2 / 2.0 / (M_sample-1)

        full_term += (term1-term2)
    
    full_term = full_term / N

    return full_term

def loss_energy_score_batch_mu_var(mu_list, var_list, goals, M_sample=1000):
    '''
    :param mu_list: (batch_size, dims)
    '''
    batch_size = mu_list.shape[0]

    # sample M times from distribution N(mu,sigma)
    # full_term = 0
    # sample M times:  get vector of batch_size, M, 64
    preds_samples = generate_random_vector_batch(M_sample, mu_list, var_list)

    # term 1
    # goals: batch_size, dims
    # preds_samples: batch_size, M_samples, dims
    diff = preds_samples - goals.unsqueeze(1).repeat(1,M_sample,1)
    diff_norm = torch.linalg.norm(diff,dim=-1)
    diff_aver = torch.sum(diff_norm,dim=-1) / M_sample
    # diff_aver_all = torch.sum(diff_aver,dim=0) / batch_size
    
    # all samples: preds_samples (1 to M-1)
    preds_samples_1 = preds_samples[:,:-1]
    # next samples: preds_samples (2 to M)
    preds_samples_2 = preds_samples[:,1:]
    norm_term = torch.norm(preds_samples_1-preds_samples_2, dim=-1)
    norm_term_aver = torch.sum(norm_term,dim=-1) / 2.0 / (M_sample-1)

    full_term = diff_aver - norm_term_aver
    full_term_aver = torch.sum(full_term) / batch_size
    return full_term_aver

# the MC approximation version
# speed up batch version
def loss_energy_score_batch(preds,goals,M_sample=1000):
    mu_list, var_list = get_mu_sigma(preds)
    return loss_energy_score_batch_mu_var(mu_list, var_list, goals, M_sample)

def generate_random_vector_batch(M_sample, mu, var):
    batch, dims = mu.shape
    device = mu.device
    # batch, M_sample, dims
    guassian_norm = torch.randn((batch,M_sample,dims)).to(device)
    # sigma: batch, dims -> batch, M_sample, dims
    sigma_expand = torch.sqrt(var).unsqueeze(1).repeat(1,M_sample,1)
    # mu: batch, dims
    mu_expand = mu.unsqueeze(1).repeat(1,M_sample,1)
    vec = guassian_norm.multiply(sigma_expand) + mu_expand
    return vec