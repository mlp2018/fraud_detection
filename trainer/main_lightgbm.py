# Copyright 2018 Aloisio Dourado
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

import preprocessing as pp

def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', 
                      objective='binary', metrics='auc', feval=None, 
                      early_stopping_rounds=20, num_boost_round=3000, 
                      verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.08,
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 0,
        'metric':metrics
    }

    lgb_params.update(params)

    print("Preparing validation datasets...")

    xgtrain = lgb.Dataset(dtrain[predictors].values, 
                          label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features)
    xgvalid = lgb.Dataset(dvalid[predictors].values, 
                          label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features)

    evals_results = {}

    bst1 = lgb.train(lgb_params, xgtrain, valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], evals_result=evals_results, 
                     num_boost_round=num_boost_round, 
                     early_stopping_rounds=early_stopping_rounds, 
                     verbose_eval=10, feval=feval)

    n_estimators = bst1.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

    #ax = lgb.plot_importance(bst1,max_num_features=60)
    #plt.show()

    return bst1

def main(train_file, test_file, job_dir):

    train_df, val_df, test_df = pp.run(train_file, test_file)

    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')
    
    target = 'is_attributed'
    predictors = ['app','device','os', 'channel', 'hour']
    categorical = ['app','device','os', 'channel', 'hour']

    print("Training...")
    params = {
        'learning_rate': 0.2,
        'num_leaves': 1400,  # we should let it be smaller than 2^(max_depth)
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': .8,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':80
    }
    
    bst = lgb_modelfit_nocv(params, train_df, val_df, predictors, target, 
                            objective='binary', metrics='auc', 
                            early_stopping_rounds=40, verbose_eval=True, 
                            num_boost_round=500, 
                            categorical_features=categorical)

    del train_df
    del val_df
    gc.collect()

    print("Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors])
    print("Writing...")
    sub.to_csv(job_dir + '/lgb_sub_tint.csv',index=False)
    print("Done...")
    print(sub.info())
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Input arguments
    parser.add_argument(
      '--train-file',
      help='GCS or local paths to training data',
      required=True
    )

    parser.add_argument(
      '--test-file',
      help='GCS or local paths to test data',
      required=True
    )

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    
    args = parser.parse_args()
    arguments = args.__dict__
    print('args: {}'.format(arguments))

    main(args.train_file, args.test_file, args.job_dir)