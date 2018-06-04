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
from os import path

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import trainer.lightgbm_functions as lf
import trainer.preprocessing as pp


# Default parameters
LGBM_PARAMS = {
    'boosting_type':      'gbdt',
    'objective':          'binary',
    'metric':             'auc',
    'learning_rate':      0.08,
    'num_leaves':         31,  # We should let it be smaller than 2^(max_depth)
    'max_depth':          -1,  # -1 means no limit
    'min_child_samples':  20,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin':            255,  # Number of bucketed bin for feature values
    'subsample':          0.6,  # Subsample ratio of the training instance.
    'subsample_freq':     0,  # Frequency of subsample, <=0 means no enable
    'colsample_bytree':   0.3,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight':   5,  # Minimum sum of instance weight(Hessian) needed in a child(leaf)
    'subsample_for_bin':  200000,  # Number of samples for constructing bin
    'min_split_gain':     0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha':          0,  # L1 regularization term on weights
    'reg_lambda':         0,  # L2 regularization term on weights
    'nthread':            8,
    'verbose':            0,
    'n_estimators':       99999999,
	'scale_pos_weight':   100
}



# Parameters to be optimized

LGBM_PARAM_GRID = {
    'scale_pos_weight': [100, 500, 1000, 5000],
    'min_data_in_leaf': [20, 100, 300, 500, 700, 900, 1100, 1300, 2000], # TODO: Why is this name different from 'min_child_samples'?
    'max_bin': [255, 270, 290, 300, 350, 400, 500],
    'reg_alpha': [0, .0001, .001, .003, .01, .03, .1],
    'reg_lambda': [0, .0001, .001, .003, .01, .03, .1],
    'learning_rate': [.0001, .001, .003, .01, .03, 0.08, .1], # NB: Use 'range' or something similar
    'num_leaves': [27, 29, 31, 32],  # We should let it be smaller than 2^(max_depth)
}



def lgb_gridsearch(default_params, param_grid, training_data, predictors, 
                   target, validation_data=None, categorical_features=None, 
                   n_splits=5, early_stopping_rounds=20):
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
    #TODO: Is this relevant?
    if validation_data is not None:

        fit_params['eval_set'] = [(validation_data[predictors].values,
                                   validation_data[target].values)]
        fit_params['early_stopping_rounds'] = early_stopping_rounds
        fit_params['eval_metric'] = 'auc'
    
    # Instantiate the grid
    skf = StratifiedKFold(n_splits=n_splits, random_state=1)
    grid = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=skf,
                        scoring='roc_auc', n_jobs=1, verbose=1, fit_params=fit_params)
    
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
    
    # Load training data set, i.e. "the 90%"
    train_df = pp.load_train(args.train_file)
    
    # Load validation data set, i.e. "the 10%"
    valid_df = pp.load_train(args.valid_file) if args.valid_file is not None \
        else None

    train_df, valid_df = pp.preprocess_confidence(train_df, valid_df)
    
    # Column we're trying to predict
    target = 'is_attributed'
    
    # Run grid search
    logging.info('Running the grid search...')
    best_params = lgb_gridsearch(LGBM_PARAMS, LGBM_PARAM_GRID, train_df, 
                                 pp.predictors, target, 
                                 categorical_features=pp.categorical, 
                                 n_splits=5, validation_data=valid_df)
    
    # Write best parameters to file
    output_file = path.join(args.job_dir, 'optimal_lgbm_param_values.txt')
       
    with open(output_file, "w") as param_file:
        json.dump(best_params, param_file)
        

# Run code    
if __name__ == '__main__':
    main()
