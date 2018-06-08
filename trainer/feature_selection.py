from os import path
from copy import deepcopy
import trainer.lightgbm_functions as lf
import trainer.preprocessing as pp
import lightgbm as lgb
import trainer.lightgbm_main as lgbm
from sklearn.feature_selection import SelectFromModel
import numpy as np
import json
import logging
import os
from sklearn.metrics import roc_auc_score
from lightgbm import plot_importance
from matplotlib import pyplot

# Default parameters
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


def feature_importance(model, train, test, target, pred, categ, params):
    # Fit model using each importance as a threshold
    thresholds = np.sort(model.feature_importances_)
    for thresh in thresholds:
        #select predictors
        ind_pred = np.where(model.feature_importances_>=thresh)
        sel_pred = [pred[i] for i in ind_pred[0]]
        # Dictionary with additional parameters to pass to .fit method.
        fit_params = {
            'feature_name': sel_pred,
            'categorical_feature': list(set(sel_pred) & set(categ)),
            # 'callbacks': [lgb.print_evaluation(period=10)]
        }
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        sel_train = selection.transform(train[pred].values)
        sel_test = selection.transform(test[pred].values)
         # train model
        selection_model = lgb.LGBMRegressor(**params)
        selection_model.fit(sel_train, train[target].values, **fit_params)
        # eval model
        y_hat = selection_model.predict(sel_test)
        score = roc_auc_score(test[target].values, y_hat)
        print("Thresh=%.3f, n=%d, auc: %.2f%%" % (thresh, sel_train.shape[1], score))


def main():
    args = lf.make_args_parser().parse_args()
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s',
                        level=args.log)

    logging.info('Preprocessing...')

    # Load training data set, i.e. "the 90%"
    train_df = pp.load_train(args.train_file)

    # Load validation data set, i.e. "the 10%"
    if args.valid_file is not None:
        valid_df = pp.load_train(args.valid_file)
        train_df, valid_df = pp.preprocess_confidence(train_df, valid_df)


    # Column we're trying to predict
    target = 'is_attributed'

    # Check if optimal parameter values have been established
    optim_file = path.join(args.job_dir, 'optimal_lgbm_param_values.txt')

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


    # Train the final model on all data
    logging.info('Training on all data...')
    gbm = lgbm.lgb_train(lgb_params, train_df, pp.predictors, target,
                    categorical_features=pp.categorical,
                    validation_data=valid_df)

    # plot feature importance
    plot_importance(gbm)
    pyplot.show()

    feature_importance(gbm, train_df, valid_df, target, pp.predictors, pp.categorical, lgb_params)


# Run code
if __name__ == '__main__':
    main()