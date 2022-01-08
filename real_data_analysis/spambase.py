# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:02:57 2021

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
from sklearn.linear_model import LinearRegression, LassoLarsIC, LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix  #, f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

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


# ================= data cleaning ==============================================
dir_name = './real_data'
paths = os.listdir('./real_data')
crime_p, spam_p = [os.path.join(dir_name, path) for path in paths]


spam = pd.read_csv(spam_p)


X = spam.iloc[:, :-2].to_numpy()
Y = spam.ham.to_numpy() * 1



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
            classifier.fit(X_select, Y_train)
            # X_select = X_test[:, idx] 
            # probs = classifier.predict_proba(X_select)
            # eps = np.where(probs < 1e-4, 1e-4, 0)
            # log_probs = np.log(probs + eps)
            # log_likelihood = (log_probs[:, 1] * Y_test + log_probs[:, 0] * (1 - Y_test)).mean()
            # predict_proba = classifier.predict_proba(X_select)
            # loss = log_loss(Y_test, predict_proba)
            # dictionary[tuple(idx)] = loss
            # reward_list.append(loss)
            
            # classifier = RandomForestClassifier(max_depth=5)
            # classifier = SVC(gamma='auto')
            # classifier = LogisticRegression()
            # classifier.fit(X_select, Y_train)
            
            X_select = X_test[:, idx] 
            score = classifier.score(X_select, Y_test)
            dictionary[tuple(idx)] = 1 - score
            reward_list.append(1 - score)
        
    return np.array(reward_list)

# ==============================================================================================

# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=3)

def metrics(idx, x_train, x_test, y_train, y_test):
    x_train_sel = x_train[:, idx]
    x_test_sel = x_test[:, idx]
    
    
    svc_sel = SVC(gamma='auto')
    svc_sel.fit(x_train_sel, y_train)
    
    return svc_sel.score(x_test_sel, y_test)


def metrics_cv(idx, X, Y, cv=5):
#     clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf = SVC(gamma='auto')
    return cross_val_score(clf, X[:, idx], Y, cv=cv)


m = 2576
n = 57

def run(seed):
    start = time.time()
    print(f'random seed: {seed} is running')
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)
    
    actor = Actor(obs_dim=n, action_dim=n)
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)

        
    action_select = []
    dictionary = dict()
    r_list = []
    
    r_baseline = torch.tensor(0)
    
    x_tt, x_val, y_tt, y_val = train_test_split(x_train, y_train, test_size=0.01, random_state=seed)  # 0.3 for metrics, 0.01 for metrics_cv

    
    for step in range(200):
        # print('step: ', step)
        
        X_train, Y_train = get_data(x_tt, y_tt, batch_size=64)
            
        actions, log_probs, entropy = actor(X_train)
        action_select.append(actions.detach().numpy().mean(axis=0))
        
        # r_baseline = critic(X_train)
        # r_baseline = r_baseline.squeeze()
        
        
        rewards = compute_reward(x_tt, y_tt, x_val, y_val, actions, num_iter=800, lr=1e-2, batch_size=64, dictionary=dictionary)
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
        
        if step > 6:
            if (abs(r_list[-1] - r_list[-2]) < 1e-3) & (abs(r_list[-2] - r_list[-3]) < 1e-3) & (abs(r_list[-3] - r_list[-4]) < 1e-3) & (abs(r_list[-4] - r_list[-5]) < 1e-3):
#             print(f'converge at step {step}')
                break
    
    action_select = np.array(action_select)
            
    
    tmp = sorted(dictionary.items(), key=lambda x: x[1])
    s = set(range(n))
    for item in tmp[:5]:
        s = s & set(item[0])
    # print(s)
    
    with torch.no_grad():
        actions, log_probs, _ = actor(x_train)
    
    
    acp1 = np.where(np.array(action_select[-10:]).mean(axis=0) > 0.9)[0]
    acp2 = (torch.where(actions.mean(dim=0) > 0.9)[0]).numpy()
    acp3 = np.array(list(s))
    
    regr = LogisticRegression(penalty='l2', fit_intercept=False, max_iter=1e6)
    regr.fit(x_train, y_train)
    lr_sfm = SelectFromModel(regr, prefit=True)
    lr2 = np.where(lr_sfm.get_support())[0]
    
    
    regr = RandomForestClassifier(max_depth=5, random_state=0)
    regr.fit(x_train, y_train)
    rf_sfm = SelectFromModel(regr, prefit=True)
    rf = np.where(rf_sfm.get_support())[0]
    
    
    svc = SVC(gamma='auto')
    svc.fit(x_train, y_train)
    
    
    # dat = np.zeros((2, 6))
    # dat[0, 5] = 57; dat[1, 5] = svc.score(x_test, y_test)
    # dat[0, 0] = len(acp1); dat[0, 1] = len(acp2); dat[0, 2] = len(acp3); dat[0, 3] = len(lr2); dat[0, 4] = len(rf)
    # dat[1, 0] = metrics(acp1, x_train, x_test, y_train, y_test)
    # dat[1, 1] = metrics(acp2, x_train, x_test, y_train, y_test)
    # dat[1, 2] = metrics(acp3, x_train, x_test, y_train, y_test)
    # dat[1, 3] = metrics(lr2, x_train, x_test, y_train, y_test)
    # dat[1, 4] = metrics(rf, x_train, x_test, y_train, y_test) 

    dat = np.zeros((2, 6))
    dat[0, 5] = 57; dat[1, 5] = metrics_cv(range(n), X, Y).mean()
    dat[0, 0] = len(acp1); dat[0, 1] = len(acp2); dat[0, 2] = len(acp3); dat[0, 3] = len(lr2); dat[0, 4] = len(rf)
    dat[1, 0] = metrics_cv(acp1, X, Y).mean()
    dat[1, 1] = metrics_cv(acp2, X, Y).mean()
    dat[1, 2] = metrics_cv(acp3, X, Y).mean()
    dat[1, 3] = metrics_cv(lr2, X, Y).mean()
    dat[1, 4] = metrics_cv(rf, X, Y).mean()  
    
    end = time.time()
    print(f'rd: {seed} take {datetime.timedelta(seconds = end - start)}')
    
    return dat



start = time.time()
results = []
for sd in tqdm([1, 2, 6, 7, 11]): 
    results.append(run(sd))
end = time.time()
print(datetime.timedelta(seconds = end - start))



dats = np.array([dat for dat in results])

np.save('./results/spambase.npy', dats)

dats.mean(axis=0)
dats.std(axis=0)
