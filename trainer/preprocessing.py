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

def run(train_file, test_file):
    
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32'
            }

    print('Load train data...')
    train_df = pd.read_csv(train_file, dtype=dtypes,
                           usecols=['ip','app','device','os', 'channel', 
                                    'click_time', 'is_attributed'])
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
    
    train_df['ip_cut'] = pd.cut(train_df.ip,15)
    
    train_df = train_df.drop(['ip','click_id'], axis = 1)
    
    categorical_columns = ['app', 'device', 'os', 'channel', 'ip_cut', 'hour']
    
    train_df = pd.get_dummies(train_df, columns = categorical_columns)

    test_df = train_df[len_train:]
    print(len(test_df))
    val_df = train_df[(len_train-2000):len_train]
    print(len(val_df))
    train_df = train_df[:(len_train-2000)]
    print(len(train_df))

    gc.collect()
    
    return train_df, val_df, test_df
