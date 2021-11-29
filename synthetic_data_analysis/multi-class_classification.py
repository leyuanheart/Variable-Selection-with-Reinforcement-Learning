import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import datetime
import random
import copy
from tqdm import tqdm

# import sklearn
from sklearn.linear_model import LinearRegression, LassoLarsIC, LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.datasets import make_classification, make_moons, make_blobs
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from scipy.stats import multivariate_normal

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.distributions import Bernoulli
# from torchsummary import summary

import multiprocessing as mp


# ===================== Binary classification =====================================================
# def generate_data(m=100, n=20, signal=1, sigma=1, num_support=8, seed=1):
#     "Generates data matrix X and observations Y."
#     np.random.seed(seed)
#     # beta_star = np.random.randn(n)
#     # beta_star[num_support:] = 0
        
#     beta_star = np.zeros(n)
#     beta_star[:num_support] = signal
#     X = np.random.randn(m,n)
#     logits = X.dot(beta_star) # + np.random.normal(0, sigma, size=m)   # 误差加在这结果会好一些
#     p = 1 / (1 + np.exp(-logits))                                    # 误差如果加在这，就会出现一些问题，p和x的关系就会被破坏
#     Y = (p > 0.5).astype(np.int)
#     return X, Y, beta_star, np.diag(np.ones(n))


# ===================== Multi-class classification =====================================================
def generate_data(n_samples, n_features, n_informative, n_redundant, n_repeated, n_classes, seed=1):
    X, Y = make_classification(n_samples=n_samples, 
                               n_features=n_features, 
                               n_informative=n_informative, 
                               n_redundant=n_redundant,
                               n_repeated = n_repeated,
                               n_classes=n_classes, 
                               shuffle=False, random_state=seed)
    return X, Y


# ==============================================================================================
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




def compute_reward(X_train, Y_train, X_test, Y_test, actions, num_iter=500, lr=1e-3, batch_size='auto', dictionary=dict()):
    reward_list = []
    for action in actions.detach().numpy():
        
        idx = np.where(action == 1)[0]
        
        if tuple(idx) in dictionary:
            reward_list.append(dictionary[tuple(idx)])
        else:
            X_select = X_train[:, idx]        
            classifier = MLPClassifier(hidden_layer_sizes=(128,), random_state=1, learning_rate='adaptive', batch_size=batch_size,
                                     learning_rate_init=lr, max_iter=num_iter, tol=1e-3)
#             classifier = RandomForestClassifier(max_depth=5)
            classifier.fit(X_select, Y_train)
            X_select = X_test[:, idx] 
#             probs = classifier.predict_proba(X_select)
#             eps = np.where(probs < 1e-4, 1e-4, 0)
#             log_probs = np.log(probs + eps)
#             log_likelihood = (log_probs[:, 1] * Y_test + log_probs[:, 0] * (1 - Y_test)).mean()
#             predict_proba = classifier.predict_proba(X_select)
#             loss = log_loss(Y_test, predict_proba)
#             dictionary[tuple(idx)] = loss
#             reward_list.append(loss)
            
#             classifier = RandomForestClassifier(max_depth=5)
#             classifier = SVC(gamma='auto')
            # classifier = LogisticRegression()
#             classifier.fit(X_select, Y_train)
            
#             X_select = X_test[:, idx] 
            score = classifier.score(X_select, Y_test)
            dictionary[tuple(idx)] = 1 - score
            reward_list.append(1 - score)
        
    return np.array(reward_list)

# ========================= training steps ====================================================
# training steps
m = 200
n = 40
sigma = 0.5
num_support = 8
signal = 1
y_true = np.zeros(n, dtype=np.int)
y_true[:num_support] = 1


# For multi-class classification
n_informative = 8
n_redundant = 0
n_repeated = 0
n_classes = 5
num_support = n_informative + n_redundant



def run(seed):
    start = time.time()
    print(f'random seed: {seed} is running')
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # X, Y, beta_star, cov = generate_data(m, n, signal, sigma, num_support, seed=seed)
    X, Y = generate_data(m, n, n_informative, n_redundant, n_repeated, n_classes, seed=seed)  
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)
    
    actor = Actor(obs_dim=n+1, action_dim=n)
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)

        
    action_select = []
    dictionary = dict()
    r_list = []
    
    r_baseline = torch.tensor(0)
    
    for step in range(200):
        # print('step: ', step)
        
        X_train, Y_train = get_data(x_train, y_train, batch_size=64)
        obs = np.concatenate((X_train, Y_train), axis=1)    
        actions, log_probs, entropy = actor(obs)
        action_select.append(actions.detach().numpy().mean(axis=0))
        
        # r_baseline = critic(X_train)
        # r_baseline = r_baseline.squeeze()
        
        
        rewards = compute_reward(x_train, y_train, x_test, y_test, actions, num_iter=800, lr=1e-2, batch_size=64, dictionary=dictionary)
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
        obs = np.concatenate((X, Y[:, np.newaxis]), axis=1)
        actions, log_probs, _ = actor(obs)
                        
#     y_pred_rl1 = np.where(action_select[-10:].mean(axis=0) >= 0.9, 1, 0)
#     y_pred_rl2 = np.where(actions.mean(dim=0) >= 0.9, 1, 0)
    y_pred_rl1 = action_select[-10:].mean(axis=0)
    y_pred_rl2 = actions.mean(dim=0).numpy()
    y_pred_rl3 = np.where([i in s for i in range(n)], 1, 0)
    
    
    logistic_cv = LogisticRegressionCV(cv=5, fit_intercept=False, penalty='l1', max_iter=1e6, solver='saga')
    logistic_cv.fit(X, Y)
    y_pred_logcv = np.where(logistic_cv.coef_.ravel() != 0, 1, 0)
#     logistic = LogisticRegression(penalty='l2', fit_intercept=False, max_iter=1e6)
#     logistic.fit(X, Y)
#     logistic_sfm = SelectFromModel(logistic, prefit=True)
#     y_pred_logsfm = np.where(logistic_sfm.get_support() != 0, 1, 0)
    rf = RandomForestClassifier(max_depth=5, random_state=seed)    
    rf.fit(X, Y)
    sfm = SelectFromModel(rf, prefit=True)
    y_pred_sfm = np.where(sfm.get_support() != 0, 1, 0)
    

    dat = np.vstack((y_pred_rl1, y_pred_rl2, y_pred_rl3, y_pred_logcv, y_pred_sfm))
    
    
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

    np.save('./results/m200_n40_makeclassification_actor+1_step200_1e-3_predictor_step800_lr1e-2_last10_bz64', dats)