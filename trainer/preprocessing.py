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

    test_df = train_df[len_train:]
    print(len(test_df))
    val_df = train_df[(len_train-2000):len_train]
    print(len(val_df))
    train_df = train_df[:(len_train-2000)]
    print(len(train_df))


    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    gc.collect()
    
    return train_df, val_df, test_df
