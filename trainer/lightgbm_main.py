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
from copy import deepcopy
import json
import logging
import os
from os import path

import lightgbm as lgb
import pandas as pd

from trainer.cross_validation import cross_val_score
from sklearn.model_selection import StratifiedKFold
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
	 'scale_pos_weight':   1, # Weight of the positive class in binary classification
}


def lgb_cv(params, training_data, predictors, target, validation_data=None,
           categorical_features=None, n_splits=5, early_stopping_rounds=20):
    """
    Returns the average score after performing cross validation on
    `training_data` with `n_splits` splits. At each iteration, LightDBM
    algorithm with `params` is used for making predictions.
    """

    # Load default parameters and update them using provided parameters
    lgb_params = deepcopy(LGBM_PARAMS)
    lgb_params.update(params)

    # Instantiate classification model with the default parameter values
    gbm = lgb.LGBMRegressor(**lgb_params)

    # Dictionary with additional parameters to pass to .fit method.
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

    # Run k-fold cross-validation
    logging.info('Running cross validation...')
    skf = StratifiedKFold(n_splits=n_splits, random_state=1)
    scores = cross_val_score(gbm, training_data[predictors].values,
                             training_data[target].values,
                             scoring='roc_auc', cv=skf, n_jobs=1, verbose=1,
                             fit_params=fit_params)

    return scores.mean()


def lgb_train(params, training_data, predictors, target,
              validation_data=None, categorical_features=None,
              early_stopping_rounds=20):

    # Load default parameters and update them using provided parameters
    lgb_params = deepcopy(LGBM_PARAMS)
    lgb_params.update(params)

    # Instantiate classification model with the default parameter values
    gbm = lgb.LGBMRegressor(**lgb_params)

    # Dictionary with additional parameters to pass to .fit method.
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

    # Train the model
    logging.info('Training the model...')
    gbm = gbm.fit(training_data[predictors].values,
                  training_data[target].values, **fit_params)

    return gbm


def main():
    args = lf.make_args_parser().parse_args()
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s',
                        level=args.log)

    logging.info('Preprocessing...')

    # Load the training data, i.e. "the 90%"
    train_df = pp.load_train(args.train_file, int(args.number_lines)
        if args.number_lines is not None else None)
    train_df = pp.preprocess_confidence(train_df)

    # Load the validation data, i.e. "the 10%"
    if args.valid_file is not None:
        valid_df = pp.load_train(args.valid_file)
        valid_df = pp.preprocess_confidence(train_df, valid_df)
    else:
        valid_df = None

    # Load the test data set, i.e. the data for which we need to make predictions
    if args.test_file is not None:
        test_df = pp.load_test(args.test_file)
        test_df = pp.preprocess_confidence(train_df, test_df)
    else:
        test_df = None

    # Column we're trying to predict
    target = 'is_attributed'

    # Check if optimal parameter values have been established
    optim_file = path.join(args.job_dir, 'optimal_lgbm_param_values.txt')

    # Provide parameter values
    lgb_params = deepcopy(LGBM_PARAMS)
    if os.path.isfile(optim_file):
        with open(optim_file, "r") as optim_file:
            optim_values = json.load(optim_file)

        # Replace default values
        logging.info('Replacing default parameter values with optimized \
ones...')
        lgb_params.update(optim_values)

    else:
        logging.info('No optimized parameter values found, so using the \
default ones...')

    if args.run == 'optimization':

        # Run cross-validation
        logging.info('Cross-validation part...')
        score = lgb_cv(lgb_params, train_df, pp.predictors, target,
                       categorical_features=pp.categorical, n_splits=5,
                       validation_data=valid_df)
        logging.info('Average score across the folds: {}'.format(score))

    elif args.run == 'submission':

        # Train the final model on all data
        logging.info('Training on all data...')
        gbm = lgb_train(lgb_params, train_df, pp.predictors, target,
                        categorical_features=pp.categorical,
                        validation_data=valid_df)

        # Check if job-dir exists, and if not, create it
        if not path.exists(args.job_dir):
            os.makedirs(args.job_dir)

        # Save model to file
        model_file = path.join(args.job_dir, 'model.txt')
        logging.info('Saving trained model to {!r}...'.format(model_file))
        gbm.booster_.save_model(model_file)

        # Make predictions and save to file
        if args.test_file is not None:
            logging.info('Making predictions...')
            predictions = gbm.predict(test_df[pp.predictors])
            predictions_file = path.join(args.job_dir, 'predictions.csv')
            logging.info('Saving predictions to {!r}...'.format(predictions_file))
            pd.DataFrame({'click_id': test_df['click_id'], 'is_attributed':
                          predictions}).to_csv(predictions_file, index=False)

    # Write parameter values to file
    output_file = path.join(args.job_dir, 'used_param_values.txt')
    logging.info('Saving used parameter values to {!r}...'.format(output_file))
    with pp.open_dispatching(output_file, mode='wb') as f:
        json.dump(lgb_params, f)

    # Make predictions and save to file
    if test_df is not None:
        logging.info('Making predictions...')
        predictions = gbm.predict(test_df[pp.predictors])
        predictions_file = path.join(args.job_dir, 'predictions.csv')
        logging.info('Saving predictions to {!r}...'.format(predictions_file))
        with pp.open_dispatching(predictions_file, mode='wb') as f:
            pd.DataFrame({'click_id': test_df['click_id'], 'is_attributed':
                          predictions}).to_csv(f, index=False)

    # Correlation matrix of data
    #corr = pp.correlation_matrix(train_df[pp.predictors])
    #print(corr)

# Run code
if __name__ == '__main__':
    main()
