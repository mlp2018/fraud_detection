# Copyright 2018 Alexander Kireev
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


import gc
import logging
import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt

DTYPES = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32',
    'hour'          : 'uint8',
    'day'           : 'uint8',
}


# Columns our predictions are based on
predictors = ['ip', 'app', 'device', 'os', 'channel', 'hour', 'hour_sq',
              'count_ip_day_hour', 'count_ip_hour_os',
              'count_ip_hh_app', 'count_ip_hour_device', 'ip_confRate',
              'app_confRate','device_confRate', 'os_confRate', 'channel_confRate',
              'app_channel_confRate', 'app_os_confRate', 'app_device_confRate',
              'channel_os_confRate', 'channel_device_confRate', 'os_device_confRate']
categorical = ['ip', 'app', 'device', 'os', 'channel', 'hour', 'hour_sq',
               'count_ip_day_hour', 'count_ip_hour_os',
               'count_ip_hh_app', 'count_ip_hour_device']


def reformat_click_time(df):
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype(
        DTYPES['hour'])
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype(
        DTYPES['day'])
    df.drop(['click_time'], axis=1, inplace=True)


def preprocess_common(df):
    """
    Data transformations that should be done to both training and test data.
    """
    logging.info('Modifying variables')
    
    # Get hour and day from clicktime        
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df.drop(['click_time'], axis=1, inplace=True)
    gc.collect()
    #print(df['hour'].value_counts(sort = True, ascending = True))
    
    logging.info('squaring clicks (hours)')
    df['hour_sq'] = df['hour']*df['hour']
    #print( df.info() )    
    
    logging.info('group by : ip_day_hour')
    gp = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day',
             'hour'])[['channel']].count().reset_index().rename(index=str,
             columns={'channel': 'count_ip_day_hour'})
    df = df.merge(gp, on=['ip','day','hour'], how='left')
    del gp
    #print( "count_ip_day_hour max value = ", df.count_ip_day_hour.max() )
    df['count_ip_day_hour'] = df['count_ip_day_hour'].astype('uint16')
    gc.collect()
    #print( df.info() )

    logging.info('group by : ip_hour_os')
    gp = df[['ip', 'day', 'os', 'hour', 'channel']].groupby(by=['ip', 'os', 'day',
             'hour'])[['channel']].count().reset_index().rename(index=str,
             columns={'channel': 'count_ip_hour_os'})
    df = df.merge(gp, on=['ip','os','hour','day'], how='left')
    del gp
    #print( "count_ip_hour_os max value = ", df.count_ip_hour_os.max() )
    df['count_ip_hour_os'] = df['count_ip_hour_os'].astype('uint16')
    gc.collect()
    #print( df.info() )

    logging.info('group by : ip_hh_app')
    gp = df[['ip', 'app', 'hour', 'day', 'channel']].groupby(by=['ip', 'app', 'day',
             'hour'])[['channel']].count().reset_index().rename(index=str,
             columns={'channel': 'count_ip_hh_app'})
    df = df.merge(gp, on=['ip','app','hour','day'], how='left')
    del gp
    #print( "count_ip_hh_app max value = ", df.count_ip_hh_app.max() )
    df['count_ip_hh_app'] = df['count_ip_hh_app'].astype('uint16')
    gc.collect()
    #print( df.info() )

    logging.info('group by : ip_hour_device')
    gp = df[['ip', 'device', 'hour', 'day', 'channel']].groupby(by=['ip', 'device', 'day',
             'hour'])[['channel']].count().reset_index().rename(index=str,
             columns={'channel': 'count_ip_hour_device'})
    df = df.merge(gp, on=['ip','device','day','hour'], how='left')
    del gp
    #print( "count_ip_hour_device max value = ", df.count_ip_hour_device.max() )
    df['count_ip_hour_device'] = df['count_ip_hour_device'].astype('uint32')
    gc.collect()
    #print( df.info() )

    df.drop(['day'], axis=1, inplace=True)
    gc.collect()
    #print( df.info() )
    #print(df.describe())
    return df


def open_dispatching(filename, use_tensorflow=False, **kwargs):
    if filename.startswith('gs://') or use_tensorflow:
        from tensorflow.python.lib.io import file_io
        return file_io.FileIO(filename, **kwargs)
    else:
        return open(filename, **kwargs)


# Aggregation function
def rate_calculation(x):
    """This function is called from within the preprocess_confidence function \
    and calculates the attributed rate and scales it by confidence."""
    log_group = np.log(100000)
    rate = x.sum() / float(x.count())
    conf = np.min([1, np.log(x.count()) / log_group]) # 1000 views -> 60% confidence, 100 views -> 40% confidence
    # if conf <= 0.4: # alternative instead of multiplying with confidence, simply use confidence as threshold
    # rate = np.nan # however this does not yield same performance as the weighting.
    return rate * conf


def preprocess_confidence(train_df, test_df=None, valid_df=None):
    """
    Feature creation that should be done given training data and then merged \
    with test data.
    """
    ATTRIBUTION_CATEGORIES = [
        # V1 Features #
        ###############
        ['ip'], ['app'], ['device'], ['os'], ['channel'],

        # V2 Features #
        ###############
        ['app', 'channel'],
        ['app', 'os'],
        ['app', 'device'],

        # V3 Features #
        ###############
        ['channel', 'os'],
        ['channel', 'device'],
        ['os', 'device']
    ]

    # Find frequency of is_attributed for each unique value in column
    logging.info("Calculating new features: Confidence rates...")
    for cols in ATTRIBUTION_CATEGORIES:
        
        # New feature name
        new_feature = '_'.join(cols) + '_confRate'
        logging.info(new_feature)
        
        # Perform the groupby
        group_object = train_df.groupby(cols)
        
        # Group sizes
        group_sizes = group_object.size()
        
        # Print group size descriptives once
        if test_df is None:
            logging.info(
            "Calculating confidence-weighted rate for: {}.\n   Saving to: {}. \
            Group Max / Mean / Median / Min: {} / {} / {} / {}".format(
                cols, new_feature,
                group_sizes.max(),
                np.round(group_sizes.mean(), 2),
                np.round(group_sizes.median(), 2),
                group_sizes.min()
            ))

        # Merge function
        def merge_new_features(group_object, df):
            df = df.merge(
            group_object['is_attributed']. \
                apply(rate_calculation). \
                reset_index(). \
                rename(
                index=str,
                columns={'is_attributed': new_feature}
            )[cols + [new_feature]],
            on=cols, how='left'
            )
                
            # Replace NaNs by average of column
            df = df.fillna(df.mean())
            
            return df
            
        # Perform the merge
        if test_df is None:
            train_df = merge_new_features(group_object, train_df)
        elif test_df is not None:
            test_df = merge_new_features(group_object, test_df)
            
    # Return the relevant data frame
    if test_df is None:
        return train_df
    elif test_df is not None:
        return test_df


def correlation_matrix(df):
    corr = df.corr()
    sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
    plt.show()


def load_train_raw(filename, number_samples):
    columns = ['ip','app','device','os', 'channel', 'click_time',
               'is_attributed']
    logging.info('Loading labeled data from {!r}...'.format(filename))
    with open_dispatching(filename, mode='rb') as f:
        return pd.read_csv(f, dtype=DTYPES, usecols=columns,
                           nrows=number_samples)


def load_test_raw(filename):
    columns = ['ip','app','device','os', 'channel', 'click_time',
               'click_id']
    logging.info('Loading unlabeled data from {!r}...'.format(filename))
    with open_dispatching(filename, mode='rb') as f:
        return pd.read_csv(f, dtype=DTYPES, usecols=columns)


def load_train(filename, number_samples=None):
    """
    Reads and preprocesses labeled data from `filename`. This method should be
    called for both training and validation data.
    """
    if number_samples < 0:
        number_samples = None
    return preprocess_common(load_train_raw(filename, number_samples))


def load_test(filename):
    """
    Reads and preprocesses unlabeled data from `filename`. This method should be
    called for test data preprocessing.
    """
    return preprocess_common(load_test_raw(filename))
