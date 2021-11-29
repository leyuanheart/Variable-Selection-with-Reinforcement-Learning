# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 11:12:27 2021

@author: leyuan
"""

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
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix  #, f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from scipy.stats import multivariate_normal

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.distributions import Bernoulli
# from torchsummary import summary

import multiprocessing as mp

import os
import sys



dir_name = './real_data'
paths = os.listdir('./real_data')
paths

crime_p, spam_p = [os.path.join(dir_name, path) for path in paths]

dat = pd.read_csv(crime_p)





# ====================== data preprocessing ==========================


Y = dat.iloc[:, -1].to_numpy()
X = dat.iloc[:, :-1].to_numpy()
# X = np.concatenate((X_num, X_cat), axis=1)
X.shape, Y.shape


# =====================================================================
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
            regressor = MLPRegressor(hidden_layer_sizes=(128,), random_state=1, learning_rate='adaptive', batch_size=batch_size,
                                      learning_rate_init=lr, max_iter=num_iter, tol=1e-3)
            # regressor = SVR()
            regressor.fit(X_select, Y_train)
            X_select = X_test[:, idx] 
            score = regressor.score(X_select, Y_test)
            # mse = np.mean((Y_test - regressor.predict(X_select))**2)
            dictionary[tuple(idx)] = 1 - score
            reward_list.append(1 - score)
        
    return np.array(reward_list)



def compare_methods(x_train, y_train):
    # print('lasso')
    lasso_bic = LassoLarsIC(criterion='bic', fit_intercept=False, normalize=False)
    lasso_bic.fit(x_train, y_train)
    # print(np.where(lasso_bic.coef_ != 0)[0] + 1)
    lasso_aic = LassoLarsIC(criterion='aic', fit_intercept=False, normalize=False)
    lasso_aic.fit(x_train, y_train)
    # print(np.where(lasso_aic.coef_ != 0)[0] + 1)
    # print('random forest')
    regr = RandomForestRegressor(max_depth=5)
    regr.fit(x_train, y_train)
    sfm = SelectFromModel(regr, prefit=True)
    # print(np.where(sfm.get_support())[0] + 1)
    
    return np.where(lasso_aic.coef_ != 0)[0], np.where(lasso_bic.coef_ != 0)[0], np.where(sfm.get_support())[0]


def metrics(idx, x_train, y_train, x_test, y_test):
    idx = idx

    x_train_sel = x_train[:, idx]
    x_test_sel = x_test[:, idx]
    
    
    reg = SVR()
    reg.fit(x_train_sel, y_train)
    
    return len(idx), np.mean(np.abs(y_test - reg.predict(x_test_sel))), reg.score(x_test_sel, y_test)


# ========================== training ======================================================================
m = 1395
n = 101

def run(seed):
    start = time.time()
    print(f'random seed: {seed} is running')
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)


    scaler_x = StandardScaler()
    x_train = scaler_x.fit_transform(x_train)
    x_test = scaler_x.transform(x_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train[:, np.newaxis]).ravel()
    y_test = scaler_y.transform(y_test[:, np.newaxis]).ravel()



    actor = Actor(obs_dim=n+1, action_dim=n)
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)


    action_select = []
    dictionary = dict()
    r_list = []

    r_baseline = torch.tensor(0)


    x_tt, x_val, y_tt, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=seed)


    for step in range(500):
#         print('step: ', step)

        X_train, Y_train = get_data(x_tt, y_tt, batch_size=64)

        obs = np.concatenate((X_train, Y_train), axis=1) 
        actions, log_probs, entropy = actor(obs)
        action_select.append(actions.detach().numpy().mean(axis=0))

        # r_baseline = critic(X_train)
        # r_baseline = r_baseline.squeeze()


        rewards = compute_reward(x_tt, y_tt, x_val, y_val, actions, num_iter=800, lr=1e-2, batch_size=64, dictionary=dictionary)
        r_list.append(rewards.mean())
#         print(f'average reward: {rewards.mean()}')
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

    #     if step > 6:
    #         if (abs(r_list[-1] - r_list[-2]) < 1e-3) & (abs(r_list[-2] - r_list[-3]) < 1e-3) & (abs(r_list[-3] - r_list[-4]) < 1e-3) & (abs(r_list[-4] - r_list[-5]) < 1e-3):
    #             print(f'converge at step {step}')
    #             break
    
    
    action_select = np.array(action_select)
    
    tmp = sorted(dictionary.items(), key=lambda x: x[1])
    s = set(range(n))
    for item in tmp[:5]:
        s = s & set(item[0])


    with torch.no_grad():  
        obs = np.concatenate((x_train, y_train[:, np.newaxis]), axis=1)
        actions, log_probs, _ = actor(obs)


    idx1 = np.where(np.array(action_select[-10:]).mean(axis=0) > 0.8)[0]
    idx2 = (torch.where(actions.mean(dim=0) > 0.8)[0]).numpy()
    idx3 = np.array(list(s))
    
    idx_aic, idx_bic, idx_rf = compare_methods(x_train, y_train)
    
    result = np.zeros((7, 3))

    for i, idx in enumerate([idx1, idx2, idx3, idx_aic, idx_bic, idx_rf, range(n)]):
        result[i] = metrics(idx, x_train, y_train, x_test, y_test)
        
    end = time.time()
    print(f'rd: {seed} take {datetime.timedelta(seconds = end - start)}')
    
    return result



start = time.time()
results = []
for sd in tqdm([0, 1, 5, 6, 8]): 
    results.append(run(sd))
end = time.time()
print(datetime.timedelta(seconds = end - start))



dats = np.array([dat for dat in results])

np.save('./results/crime.npy', dats)

dats.mean(axis=0)
dats.std(axis=0)



