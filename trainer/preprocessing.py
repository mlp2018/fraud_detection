# Copyright 2018 Aloisio Dourado
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


DTYPES = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32',
}


def _preprocess_common(df):
    """
    Data transformations that should be done to both training and test data.
    """
    logging.info('Modifying variables')
        
    #We have most and least freq hours observed in test data as below
    most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
    least_freq_hours_in_test_data = [6, 11, 15]
    
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df.drop(['click_time'], axis=1, inplace=True)
    gc.collect()
    #print(df['hour'].value_counts(sort = True, ascending = True))
    
    #If hour is in most frequent hours in test data then assign group 1, 
    #If hour is in least frequent hours in test data then assign group 2, 
    #assign group 3 to any remaining hours    
    df['freq_h'] = (   3 
                         - 2*df['hour'].isin(  most_freq_hours_in_test_data ) 
                         - 1*df['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')
    #print( df.info() )
    
    logging.info('squaring clicks (hours)')
    df['hour_sq'] = df['hour']*df['hour']
    #print( df.info() )
    
    
    logging.info('group by : ip_day_freq_h')
    gp = df[['ip', 'day', 'freq_h', 'channel']].groupby(by=['ip', 'day',
             'freq_h'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'count_ip_day_freq_h'})
    df = df.merge(gp, on=['ip','day','freq_h'], how='left')
    del gp
    df.drop(['freq_h'], axis=1, inplace=True)
    #print( "count_ip_day_freq_h max value = ", df.count_ip_day_freq_h.max() )
    df['count_ip_day_freq_h'] = df['count_ip_day_freq_h'].astype('uint32')
    gc.collect()
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

    df.drop( ['ip','day'], axis=1, inplace=True )
    gc.collect()
    #print( df.info() )    
    #print(df.describe())
    return( df )



def load_train_raw(filename):
    columns = ['ip','app','device','os', 'channel', 'click_time',
               'is_attributed']
    logging.info('Loading labeled data from {!r}...'.format(filename))
    return pd.read_csv(filename, dtype=DTYPES, usecols=columns)


def load_test_raw(filename):
    columns = ['ip','app','device','os', 'channel', 'click_time',
               'click_id']
    logging.info('Loading unlabeled data from {!r}...'.format(filename))
    return pd.read_csv(filename, dtype=DTYPES, usecols=columns)


def load_train(filename):
    """
    Reads and preprocesses labeled data from `filename`. This method should be
    called for both training and validation data.
    """
    return _preprocess_common(load_train_raw(filename))


def load_test(filename):
    """
    Reads and preprocesses unlabeled data from `filename`. This method should be
    called for test data preprocessing.
    """
    return _preprocess_common(load_test_raw(filename))
