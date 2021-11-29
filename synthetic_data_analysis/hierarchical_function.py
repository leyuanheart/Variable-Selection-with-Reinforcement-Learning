import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import datetime
import random
import copy
from tqdm import tqdm

# import sklearn
from sklearn.linear_model import LinearRegression, LassoLarsIC, LassoCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SequentialFeatureSelector

from scipy.stats import multivariate_normal

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.distributions import Bernoulli
# from torchsummary import summary

import multiprocessing as mp


# ===================== linear model =====================================================
# def generate_data(m=100, n=20, signal=1, sigma=5, num_support=8, seed=1):
#     "Generates data matrix X and observations Y."
#     np.random.seed(seed)
#     # beta_star = np.random.randn(n)
#     # beta_star[8:] = 0
#     beta_star = np.zeros(n)
#     beta_star[:num_support] = signal
#     X = np.random.randn(m,n)
#     Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
#     return X, Y, beta_star, np.diag(np.ones(n))

# ===================== covariates with correlations ====================================
# def generate_data(m=100, n=20, signal=1, sigma=5, num_support=8, seed=1):
#     "Generates data matrix X and observations Y."
#     np.random.seed(seed)
        
#     mean = np.random.uniform(-5, 5, n)
#     cov = np.ones((n, n))
#     for i in range(n):
#         for j in range(n):
#             cov[i, j] = 0.8**abs(i-j)
#     X = np.random.multivariate_normal(mean, cov, m)
    
#     beta_star = np.zeros(n)
#     beta_star[:num_support] = signal
    
#     Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
#     return X, Y, beta_star, cov

# ===================== linear model with intersection =====================================================
# def generate_data(m=100, n=20, signal=1, sigma=5, num_support=8, seed=1):
#     "Generates data matrix X and observations Y."
#     np.random.seed(seed)
    
#     X = np.random.randn(m,n)
#     X_s = X[:, :num_support]
#     beta_star = np.array([signal] * 6)
    
#     Y = beta_star[0] * X_s[:, 0] + beta_star[1] * X_s[:, 1] * X_s[:, 2] + \
#         beta_star[2] * X_s[:, 3] + beta_star[3] * X_s[:, 4] * X_s[:, 5] + beta_star[4] * X_s[:, 6] + \
#         beta_star[5] * X_s[:, 7] + np.random.normal(0, sigma, size=m)
#     return X, Y, beta_star, np.diag(np.ones(n))


# ===================== qudratic model =====================================================
# def generate_data(m=100, n=20, signal=1, sigma=1, num_support=8, seed=1):
#     '''
#     Generates data matrix X and observations Y.
#     @m: sample_size
#     @n: # covariates
#     @sigma: sigma of Gaussian distribution for generate X
#     @num_support: # covariate in true model
#     @seed: random seed
#     '''
    
#     np.random.seed(seed)
#     mean = np.random.uniform(-5, 5, n)
#     cov = np.diag(np.random.uniform(1, 3, n))
#     # cov = np.random.randn(n, n)
#     # cov = cov.dot(cov.T)
#     # cov = (cov - cov.min()) / (cov.max()-cov.min())
#     # cov = cov + np.diag(np.random.uniform(1, 3, n))
#     X = np.random.multivariate_normal(mean, cov, m)
#     # idx = np.random.choice(range(d), num_support, replace=False)
#     X_s = X[:, :num_support]
    
#     beta_star = np.array([signal] * num_support)
    
#     Y = np.dot(X_s[:,:4]**2, beta_star[:4]) + np.dot(X_s[:, 4:], beta_star[4:]) + np.random.normal(0, sigma, m) 
    
#     return X, Y, beta_star, cov


# ===================== Hierarchical model =====================================================
def generate_data(m=100, n=20, signal=1, sigma=5, num_support=8, seed=1):
    "Generates data matrix X and observations Y."
    np.random.seed(seed)
    X = np.random.randn(m, n)
    X_s = X[:, :num_support]
    
    W1 = np.random.randn(num_support, 32)
    W2 = np.random.randn(32, )
    Y = np.maximum(0, X_s.dot(W1)).dot(W2) + np.random.normal(0, sigma, size=m)
    return X, Y, (W1, W2), np.diag(np.ones(n))





# ========================== Architecture =====================================================
def get_data(x, y, batch_size=32):
#     x = StandardScaler(with_mean=True, with_std=True).fit_transform(x)
    sample_size = x.shape[0]
    idx = np.random.choice(range(sample_size), batch_size, replace=False)
    return x[idx, :], y[idx, np.newaxis]



class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        '''
        obs_dim: dim_x or (dim_x + dim_y)
        action_dim: dim_x
        '''
        super(Actor, self).__init__()
        
        
        self.fc1 = nn.Linear(in_features=obs_dim, out_features=256)
        self.fc2 = nn.Linear(256, action_dim)
        
    def forward(self, obs):
        obs = torch.tensor(obs, dtype=torch.float)
        logits = F.relu(self.fc1(obs))
        logits = self.fc2(logits)
        
        m = Bernoulli(logits=logits)
        
        actions = m.sample()
        log_probs = m.log_prob(actions)
        entropy = m.entropy()
        
        return actions, log_probs, entropy


# class Critic(nn.Module):
#     def __init__(self, obs_dim):
        
#         super(Critic, self).__init__()
#         self.f1 = nn.Linear(in_features=obs_dim, out_features=128)
#         self.f2 = nn.Linear(in_features=128, out_features=1)
        
#     def forward(self, obs): 
#         obs = torch.tensor(obs, dtype=torch.float)
#         r_baseline = F.relu(self.f1(obs))
#         r_baseline = self.f2(r_baseline)        
#         return r_baseline


def compute_reward(X_train, Y_train, X_test, Y_test, actions, num_iter=500, lr=1e-3, batch_size='auto', dictionary=dict()):
    reward_list = []
    for action in actions.detach().numpy():
        
        idx = np.where(action == 1)[0]
        
        if tuple(idx) in dictionary:
            reward_list.append(dictionary[tuple(idx)])
        else:
            X_select = X_train[:, idx]        
            regressor = MLPRegressor(hidden_layer_sizes=(128,), random_state=1, learning_rate='adaptive', batch_size=batch_size,
                                     learning_rate_init=lr, max_iter=num_iter, tol=1e-3, alpha=0.01)
#             regressor = LinearRegression(fit_intercept=False)
            regressor.fit(X_select, Y_train)
            X_select = X_test[:, idx] 
            score = regressor.score(X_select, Y_test)
            # mse = np.mean((Y_test - regressor.predict(X_select))**2)
            dictionary[tuple(idx)] = 1 - score
            reward_list.append(1 - score)
        
    return np.array(reward_list)




# ========================= training steps ====================================================
m = 100
n = 24
sigma = 1
num_support = 8
signal = 1
y_true = np.zeros(n, dtype=np.int)
y_true[:num_support] = 1


def run(seed):
    start = time.time()
    print(f'random seed: {seed} is running')
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    X, Y, beta_star, cov = generate_data(m, n, signal, sigma, num_support, seed=seed)   
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)
    
    actor = Actor(obs_dim=n, action_dim=n)
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    # critic = Critic(obs_dim=n)
    # critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)
        
    action_select = []
    dictionary = dict()
    r_list = []
    
    r_baseline = torch.tensor(0)
    
    for step in range(300):
        # print('step: ', step)
        
        X_train, Y_train = get_data(x_train, y_train, batch_size=64)
            
        actions, log_probs, entropy = actor(X_train)
        action_select.append(actions.detach().numpy().mean(axis=0))
        
        # r_baseline = critic(X_train)
        # r_baseline = r_baseline.squeeze()
        
        
        rewards = compute_reward(x_train, y_train, x_test, y_test, actions, num_iter=1000, lr=1e-2, batch_size=64, dictionary=dictionary)
        r_list.append(rewards.mean())
        # print(f'average reward: {rewards.mean()}')
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        r_baseline = 0.95 * r_baseline + 0.05 * rewards.mean()
        
        # update actor
        actor_loss =  ((rewards - r_baseline) * log_probs.sum(dim=-1)).mean()
        # actor_loss =  (rewards * log_probs.sum(dim=-1)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        # print(f'actor loss: {actor_loss.item()}')
        
        # actor_loss =  (rewards * log_probs.sum(dim=-1)).mean()
        # actor_optimizer.zero_grad()
        # actor_loss.backward()
        # actor_optimizer.step()
        # print(f'actor loss: {actor_loss.item()}\n')
        
        # update critic
        # critic_loss = F.mse_loss(r_baseline, rewards)
        # critic_optimizer.zero_grad()
        # critic_loss.backward()
        # critic_optimizer.step()
        # print(f'critic loss: {critic_loss.item()}\n')
        
#         if step > 6:
#             if (abs(r_list[-1] - r_list[-2]) < 1e-3) & (abs(r_list[-2] - r_list[-3]) < 1e-3) & (abs(r_list[-3] - r_list[-4]) < 1e-3) & (abs(r_list[-4] - r_list[-5]) < 1e-3):
# #             print(f'converge at step {step}')
#                 break
    
    action_select = np.array(action_select)
            
    
    tmp = sorted(dictionary.items(), key=lambda x: x[1])
    s = set(range(n))
    for item in tmp[:10]:
        s = s & set(item[0])
    # print(s)
    
    with torch.no_grad():
        actions, log_probs, _ = actor(X)
                        
#     y_pred_rl1 = np.where(action_select[-10:].mean(axis=0) >= 0.9, 1, 0)
#     y_pred_rl2 = np.where(actions.mean(dim=0) >= 0.9, 1, 0)
    y_pred_rl1 = action_select[-10:].mean(axis=0)
    y_pred_rl2 = actions.mean(dim=0).numpy()
    y_pred_rl3 = np.where([i in s for i in range(n)], 1, 0)
    
    
    lasso_bic = LassoLarsIC(criterion='bic', fit_intercept=False, normalize=False)
    lasso_bic.fit(X, Y)
    y_pred_bic = np.where(lasso_bic.coef_ != 0, 1, 0)
    lasso_aic = LassoLarsIC(criterion='aic', fit_intercept=False, normalize=False)
    lasso_aic.fit(X, Y)
    y_pred_aic = np.where(lasso_aic.coef_ != 0, 1, 0)
    rf = RandomForestRegressor(max_depth=5, random_state=seed)    
    rf.fit(X, Y)
    sfm = SelectFromModel(rf, prefit=True)
    y_pred_sfm = np.where(sfm.get_support() != 0, 1, 0)
    
    dat = np.vstack((y_pred_rl1, y_pred_rl2, y_pred_rl3, y_pred_aic, y_pred_bic, y_pred_sfm))
    
    # cm1 = confusion_matrix(y_true, y_pred_rl1)
    # cm2 = confusion_matrix(y_true, y_pred_rl2)
    # cm_bic = confusion_matrix(y_true, y_pred_bic)
    # cm_aic = confusion_matrix(y_true, y_pred_aic)
    # # return cm1, cm2, cm_bic, cm_aic
    
    # dat = pd.DataFrame(np.zeros((3, 4)), index=['precision', 'specificity', 'recall'])
    #                     # columns=['cm1', 'cm2', 'bic', 'aic'])
    
    # for i, cm in enumerate([cm1, cm2, cm_bic, cm_aic]):
    #     tn, fp, fn, tp = cm.ravel()
    #     dat.loc['precision', i] = tp/(tp+fn)
    #     dat.loc['specificity', i] = tn/(tn+fp)
    #     dat.loc['recall', i] = 0 if tp + fp == 0 else tp/(tp+fp)
        
    # dat.columns = ['cm1', 'cm2', 'bic', 'aic']
    
    
#     regr = LogisticRegression(penalty='l2', fit_intercept=False, max_iter=1e6)
#     regr.fit(X, Y)
#     sfm = SelectFromModel(regr, prefit=True)
#     y_pred_log = np.where(sfm.get_support(), 1, 0)
    
    
    end = time.time()
    print(f'rd: {seed} take {datetime.timedelta(seconds = end - start)}')
    
    return dat




if __name__ == '__main__':   
    # results = []
    # for sd in tqdm(range(20)):
    #     results.append(run(sd))

    # print("CPU的核数为：{}".format(mp.cpu_count()))
    start = time.time()
    pool = mp.Pool(4)
    dats = pool.map(run, range(50))
    pool.close()
    end = time.time()
    print(datetime.timedelta(seconds = end - start))
    
    
    dats = np.array([dat for dat in dats])

    np.save('./results/m100_n24_hierarchical_actor_step300_1e-3_regressor_step1000_lr1e-2_last10_5.2h', dats)