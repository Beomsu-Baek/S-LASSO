# Author: Jongkwon Jo <jongkwon.jo@gmail.com>
# License: MIT
# Date: 10, Aug 2021

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from . import util, glmnet_model
from tqdm import tqdm
import numpy as np
import math

class HiLasso2:
    def __init__(self, q='auto', r=30, alpha=0.05, max_procedure = 1,
                 logistic=False, random_state=None):
        self.q = q
        self.r = r
        self.alpha = alpha
        self.max_procedure = max_procedure
        self.logistic = logistic
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        self.X = np.array(X)
        self.y = np.array(y).ravel()
        self.n, self.p = X.shape
        self.q = self.n if self.q == 'auto' else self.q
        self.sample_weight = np.ones(
            self.n) if sample_weight is None else np.asarray(sample_weight)

        print('Procedure 1')
        b1 = self._bootstrapping(mode='procedure1')
        self.coef1_, self.coef_ = b1, b1
        
        if self.max_procedure == 2:
            print('Procedure 2')
            b1_mean = abs(self.coef1_).mean(axis = 0)
            importance_score = np.where(b1_mean == 0, 1e-10, b1_mean)
            self.select_prob = importance_score / importance_score.sum()
            self.penalty_weights = 1 / (self.select_prob * 100)
            
            b2 = self._bootstrapping(mode='procedure2')
            self.coef_ = b2
        return self

    def _bootstrapping(self, mode):
        if mode == 'procedure1':
            self.method = 'ElasticNet'
            self.select_prob = None
        else:
            self.method = 'AdaptiveLASSO'
 
        betas = np.array([list(self._estimate_coef(bootstrap_number)) for bootstrap_number in tqdm(np.arange(self.r))])
        return betas

    def _estimate_coef(self, bootstrap_number):
        """
        Estimate coefficients for each bootstrap samples.
        """
        beta = np.zeros((self.p))

        # Set random seed as each bootstrap_number.
        self.rs = np.random.RandomState(
            bootstrap_number + self.random_state) if self.random_state else np.random.default_rng()
        
        # Generate bootstrap index of sample.
        bst_sample_idx = self.rs.choice(np.arange(self.n), size=self.n, replace=True, p=None)
        bst_predictor_idx_list, dup_sample = self._variable_sampling(bootstrap_number, self.select_prob)
        
        for bst_predictor_idx in bst_predictor_idx_list:
            # Standardization.
            X_sc, y_sc, x_std = util.standardization(self.X[bst_sample_idx, :][:, bst_predictor_idx],
                                                     self.y[bst_sample_idx])
            # Estimate coef.
            if self.method == 'ElasticNet':
                coef = glmnet_model.ElasticNet(X_sc, y_sc, logistic=self.logistic,
                                               sample_weight=self.sample_weight[bst_sample_idx], random_state=self.rs)
            else:
                coef = glmnet_model.AdaptiveLasso(X_sc, y_sc, logistic=self.logistic,
                                                  sample_weight=self.sample_weight[bst_sample_idx], random_state=self.rs,
                                                  adaptive_weights=self.penalty_weights[bst_predictor_idx])
            beta[bst_predictor_idx] = beta[bst_predictor_idx] + (coef / x_std)
        beta[dup_sample] = beta[dup_sample]/2
        return beta
    
    def _variable_sampling(self, bootstrap_number, select_prob = None):
        idx_set = np.arange(self.p)
        select_prob = np.ones(self.p) if select_prob is None else select_prob
        bst_predictor_idx_list = []
        for i in range(self.p//self.q):
            prob = select_prob[idx_set]
            bst_predictor_idx_list.append(self.rs.choice(idx_set, self.q, replace=False, p = prob/prob.sum()))
            idx_set = np.setdiff1d(idx_set, bst_predictor_idx_list[-1])
        if self.p%self.q != 0:
            sub = np.concatenate(bst_predictor_idx_list)
            prob = select_prob[sub]
            dup_sample = self.rs.choice(sub, self.q-self.p%self.q, replace=False, p = prob/prob.sum())
            bst_predictor_idx_list.append(np.concatenate((idx_set, dup_sample)))
        else:
            dup_sample = []
        return bst_predictor_idx_list, dup_sample

