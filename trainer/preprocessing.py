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


import pandas as pd
import gc
import logging


DTYPES = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32',
}


def _preprocess_common(data_frame):
    logging.info('Replacing `click_time` by `hour`...')
    data_frame['hour'] = pd.to_datetime(
        data_frame['click_time']).dt.hour.astype('uint8')
    del data_frame['click_time']
    gc.collect()
    return data_frame


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
    return _preprocess_common(load_train_raw(filename))


def load_test(filename):
    return _preprocess_common(load_test_raw(filename))


def run(train_file, test_file):

    print('Load test data...')
    test_df = pd.read_csv(test_file, dtype=dtypes, usecols=['ip', 'app',
                                                            'device', 'os', 
                                                            'channel', 
                                                            'click_time', 
                                                            'click_id'])

    len_train = len(train_df)
    train_df=train_df.append(test_df)

    del test_df
    gc.collect()

    print('Preprocessing the data...')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    del train_df['click_time']
    gc.collect()

    train_df.info()

    test_df = train_df[len_train:]
    print(len(test_df))
    val_df = train_df[(len_train-2000):len_train]
    print(len(val_df))
    train_df = train_df[:(len_train-2000)]
    print(len(train_df))


    gc.collect()
    
    return train_df, val_df, test_df
