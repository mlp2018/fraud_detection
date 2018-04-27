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

from __future__ import absolute_import, division, print_function
from builtins import (bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)
import argparse
from copy import deepcopy
import gc
import logging
import os
from os import path
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .cross_validation import stratified_kfold
import .preprocessing as pp


# Some default parameters
LGBM_PARAMS = {
    'boosting_type':      'gbdt',
    'objective':          'binary',
    'metric':             'auc',
    'learning_rate':      0.08,
    'num_leaves':         31,  # we should let it be smaller than 2^(max_depth)
    'max_depth':          -1,  # -1 means no limit
    'min_child_samples':  20,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin':            255,  # Number of bucketed bin for feature values
    'subsample':          0.6,  # Subsample ratio of the training instance.
    'subsample_freq':     0,  # frequence of subsample, <=0 means no enable
    'colsample_bytree':   0.3,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight':   5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'subsample_for_bin':  200000,  # Number of samples for constructing bin
    'min_split_gain':     0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha':          0,  # L1 regularization term on weights
    'reg_lambda':         0,  # L2 regularization term on weights
    'nthread':            8,
    'verbose':            0,
}


def lgb_cv(params, training_data, predictors, target,
           validation_data=None, categorical_features=None,
           n_splits=2, early_stopping_rounds=20):
    """
    Returns the average score after performing cross validation on
    `training_data` with `n_splits` splits. At each iteration, LightDBM
    algorithm with `params` is used for making predictions.
    """
    # Loads default parameters and then updates them using provided parameters.
    lgb_params = deepcopy(LGBM_PARAMS)
    lgb_params.update(params)
    gbm = lgb.LGBMRegressor(**lgb_params)
    # Dictionary with additional parameters to pass to .fit method.
    fit_params = {
        'feature_name': predictors,
        'categorical_feature': categorical_features,
        # 'callbacks': [lgb.print_evaluation(period=10)]
    }
    if validation_data is not None:
        # If we're given some validation data, we can use it for early
        # stopping.
        fit_params['eval_set'] = [(validation_data[predictors].values,
                                   validation_data[target].values)]
        fit_params['early_stopping_rounds'] = early_stopping_rounds
        fit_params['eval_metric'] = 'auc'
    logging.info('Running cross validation...')
    skf = stratified_kfold(n_splits=n_splits)
    scores = cross_val_score(gbm, training_data[predictors].values,
                             training_data[target].values,
                             scoring='roc_auc', cv=skf, n_jobs=1, verbose=1,
                             fit_params=fit_params)
    logging.info('Scores: {}'.format(scores))
    return scores.mean()


def lgb_train(params, training_data, predictors, target,
              validation_data=None, categorical_features=None,
              early_stopping_rounds=20):
    lgb_params = deepcopy(LGBM_PARAMS)
    lgb_params.update(params)
    gbm = lgb.LGBMRegressor(**lgb_params)
    fit_params = {
        'feature_name': predictors,
        'categorical_feature': categorical_features,
        # 'callbacks': [lgb.print_evaluation(10)],
    }
    if validation_data is not None:
        fit_params['eval_set'] = [(validation_data[predictors].values,
                                   validation_data[target].values)]
        fit_params['early_stopping_rounds'] = early_stopping_rounds
        fit_params['eval_metric'] = 'auc'
    logging.info('Training the model...')
    gbm = gbm.fit(training_data[predictors].values,
                  training_data[target].values, **fit_params)
    return gbm


class StoreLoggingLevel(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError('`nargs` is not supported.')
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        level = getattr(logging, value.upper(), None)
        if not isinstance(level, int):
            raise ValueError('Invalid log level: {}'.format(value))
        setattr(namespace, self.dest, level)


def make_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--train-file', help='Path to training data', required=True)
    parser.add_argument(
      '--valid-file', help='Path to validation data', required=False)
    parser.add_argument(
      '--test-file', help='Path to test data', required=False)
    parser.add_argument(
        '--job-dir',
        help='Directory where to store checkpoints and exported models.',
        default='.')
    parser.add_argument(
        '--log', help='Logging level', default=logging.DEBUG,
        action=StoreLoggingLevel)
    return parser


def main():
    args = make_args_parser().parse_args()
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s',
                        level=args.log)

    logging.info('Preprocessing...')
    # Load training data set, i.e. "the 90%"
    train_df = pp.load_train(args.train_file)
    # Load validation data set, i.e. "the 10%"
    valid_df = pp.load_train(args.valid_file) if args.valid_file is not None \
        else None
    # Load the test data set, i.e. data for which we need to make predictions.
    test_df = pp.load_test(args.test_file) if args.test_file is not None \
        else None
    
    # Column we're trying to predict
    target = 'is_attributed'
    # Columns our predictions are based on.
    predictors = ['app','device','os', 'channel', 'hour']
    categorical = ['app','device','os', 'channel', 'hour']
    params = {
        'learning_rate':     0.2,
        'num_leaves':        1400,  # we should let it be smaller than 2^(max_depth)
        'max_depth':         3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin':           100,  # Number of bucketed bin for feature values
        'subsample':         0.8,  # Subsample ratio of the training instance.
        'subsample_freq':    1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree':  0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight':  0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':  80
    }

    logging.info('Cross-validation part...')
    score = lgb_cv(params, train_df, predictors, target,
                   categorical_features=categorical, n_splits=5,
                   validation_data=valid_df)
    logging.info('Score: {}'.format(score))

    logging.info('Training on all data...')
    gbm = lgb_train(params, train_df, predictors, target,
                    categorical_features=categorical,
                    validation_data=valid_df)
    if not path.exists(args.job_dir) \
       and path.exists(path.dirname(path.abspath(args.job_dir))):
        os.mkdir(args.job_dir)
    model_file = path.join(args.job_dir, 'model.txt')
    logging.info('Saving trained model to {!r}'.format(model_file))
    gbm.booster_.save_model(model_file)

    if test_df is not None:
        logging.info('Making predictions ...')
        predictions = gbm.predict(test_df)
        predictions_file = path.join(args.job_dir, 'model.txt')
        pd.DataFrame({'click_id': test_df['click_id'], 'is_attributed':
                      predictions}).to_csv(predictions_file)
    
    
if __name__ == '__main__':
    main()
