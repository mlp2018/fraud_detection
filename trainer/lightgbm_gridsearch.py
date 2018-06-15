# Copyright 2018 Aloisio Dourado
# Copyright 2018 Sophie Arana
# Copyright 2018 Johanna de Vos
# Copyright 2018 Tom Westerhout
# Copyright 2018 Andre Vargas 
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
import json
import logging
import os
from os import path

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit

import trainer.lightgbm_functions as lf
import trainer.preprocessing as pp


# Default parameters
LGBM_PARAMS = {
    'boosting_type':      'gbdt', # Gradient Boosting Decision Tree
    'objective':          'binary', # Binary classification
    'metric':             'auc', # Area under the ROC curve
    'learning_rate':      0.08, # Controls how much each tree is weighted
    'num_leaves':         31, # Max number of leaves per tree (should not exceed 2^max_depth)
    'min_data_in_leaf':   20, # Min number of data points per leaf
    'max_depth':          -1, # Tree depth. -1 means no limit
    'max_bin':            255, # How memory will be auto-compressed (number of bucketed bins for feature values)
    'subsample':          0.6, # Ratio of observations that are randomly sampled per tree
    'subsample_freq':     0, # How often bagging should happen
    'colsample_bytree':   0.3, # Subsample ratio of columns when constructing each tree.
    'min_child_weight':   5, # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'subsample_for_bin':  200000, # Number of samples for constructing histogram bins
    'min_split_gain':     0, # Minimum loss reduction needed for a node to split
    'reg_alpha':          0, # L1 regularization term on weights
    'reg_lambda':         0, # L2 regularization term on weights
    'nthread':            8, # Number of threads
    'verbose':            0, # Verbosity
    'n_estimators':       2000, # Number of boosting iterations. Very high because of early stopping
	 'scale_pos_weight':   1.0, # Weight of the positive class in binary classification
}


# Parameters to be optimized
LGBM_PARAM_GRID = {
    'learning_rate': [.0001, .001, .01, 0.08, .1],
    'num_leaves': [11, 21, 31],
    'min_data_in_leaf': [10, 20, 100, 1000],
    'max_depth': [-1, 6, 12],
    'subsample': [0.3, 0.6, 1],
    'colsample_bytree': [0.3, 0.6, 1],
    'min_child_weight': [0.001, 0.01, 0.1, 1, 5],
    'scale_pos_weight': [1, 10, 100, 1000],
}


def lgb_gridsearch(default_params, param_grid, training_data, predictors, 
                   target, validation_data=None, categorical_features=None,
                   split_where=None, early_stopping_rounds=20):
    """
    Performs a grid search to find the optimal value for all parameters.
    The grid search makes use of k-fold cross-validation.
    Returns a dictionary with the optimal value for all parameters."

    """
    
    # Instantiate classification model with the default parameter values
    gbm = lgb.LGBMRegressor(**default_params)
    
    # Dictionary with additional parameters to pass to .fit method
    fit_params = {
        'feature_name': predictors,
        'categorical_feature': categorical_features,
        # 'callbacks': [lgb.print_evaluation(period=10)]
    }

    # If we're given some validation data, we can use it for early stopping    
    if validation_data is not None:

        fit_params['eval_set'] = [(validation_data[predictors].values,
                                   validation_data[target].values)]
        fit_params['early_stopping_rounds'] = early_stopping_rounds
        fit_params['eval_metric'] = 'auc'
    
    # Instantiate the grid
    skf = PredefinedSplit(test_fold=[-1]*(len(training_data) - split_where) + 
                          [0]*split_where)
    grid = RandomizedSearchCV(estimator=gbm, param_distributions=param_grid, 
                              cv=skf, scoring='roc_auc', n_jobs=1, verbose=1, 
                              fit_params=fit_params, n_iter=3)

    # Use the below code if you want to do a full grid search
# =============================================================================
#     grid = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=skf,
#                         scoring='roc_auc', n_jobs=1, verbose=1, 
#                         fit_params=fit_params)
# =============================================================================

    
    # Fit the grid with data
    logging.info('Running the grid search...')
    grid.fit(training_data[predictors].values, training_data[target].values)

    # Examine the results
    scores = grid.cv_results_['mean_test_score']
    best_score = grid.best_score_
    best_params = grid.best_params_
    logging.info('The best score from the grid search was: {}'
                 .format(best_score))
    logging.info('It was obtained with the following parameters: {}'
                 .format(best_params))
    
    return best_params


def main():
    args = lf.make_args_parser().parse_args()
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s',
                        level=args.log)

    logging.info('Preprocessing...')
    
    # Load the training data, i.e. "the 90%"
    train_df = pp.load_train_raw(args.train_file, int(args.number_lines))
    #train_df = pp.preprocess_confidence(train_df)
    
    # Use the last 10% of the training data as validation data
    ten_percent = int(int(args.number_lines) * 0.10)
    valid_df = train_df[-ten_percent:]
    train_df = train_df[:-ten_percent]

    # Process validation separately
    valid_df = pp.preprocess_confidence(train_df[:-ten_percent], 
                                        pp._preprocess_common(valid_df))
    
    # Process train separately
    train_split = pp.preprocess_confidence(pp._preprocess_common(
            train_df[:-ten_percent]))
    
    # Process test separately
    test_split = pp.preprocess_confidence(train_split, pp._preprocess_common(
            train_df[-ten_percent:]))
    
    # Merge train and test data again
    training_data = train_split.append(test_split)
    del train_df
    del train_split
    del test_split
    
    # Column we're trying to predict
    target = 'is_attributed'
    
    # Run grid search
    logging.info('Running the grid search...')
    best_params = lgb_gridsearch(LGBM_PARAMS, LGBM_PARAM_GRID, training_data, 
                                 pp.predictors, target, 
                                 categorical_features=pp.categorical, 
                                 split_where=ten_percent, validation_data=
                                 valid_df)
    
    # Check whether job-dir exists    
    if not os.path.exists(args.job_dir):
        os.makedirs(args.job_dir)

    # Write best hyperparameter values to file
    output_file = path.join(args.job_dir, 'optimal_lgbm_param_values.txt')
    logging.info('Saving the optimal hyperparameter values to {!r}...'
                 .format(output_file))
    with pp.open_dispatching(output_file, mode='wb') as f:
        json.dump(best_params, f)
    
    
# Run code    
if __name__ == '__main__':
    main()
    