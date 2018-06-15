# -*- coding: utf-8 -*-
# Copyright 2018 Aloisio Dourado
# Copyright 2018 Sophie Arana
# Copyright 2018 Johanna de Vos
# Copyright 2018 Tom Westerhout
# Copyright 2018 André Vargas
#
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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() #make the plots look pretty

path = '~/Documents/fraud_detection/trainer/data/'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

print('load train...')
train_cols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
train_df = pd.read_csv(path+"train_small_10.csv", dtype=dtypes, usecols=train_cols)

import gc

len_train = len(train_df)

gc.collect()

print('data prep...')






def prep_data( df ):
    
    #We have most and least freq hours observed in test data as below
    most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
    least_freq_hours_in_test_data = [6, 11, 15]
    
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df.drop(['click_time'], axis=1, inplace=True)
    gc.collect()
    print(df['hour'].value_counts(sort = True, ascending = True))
#If hour is in most frequent hours in test data then assign group 1, 
#If hour is in least frequent hours in test data then assign group 2, 
#assign group 3 to any remaining hours    
    df['freq_h'] = (   3 
                         - 2*df['hour'].isin(  most_freq_hours_in_test_data ) 
                         - 1*df['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')
    print( df.info() )
    
    print('squaring clicks (hours)')
    df['hour_sq'] = df['hour']*df['hour']
    print( df.info() )
    
    
    print('group by : ip_day_freq_h')
    gp = df[['ip', 'day', 'freq_h', 'channel']].groupby(by=['ip', 'day',
             'freq_h'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'count_ip_day_freq_h'})
    df = df.merge(gp, on=['ip','day','freq_h'], how='left')
    del gp
    df.drop(['freq_h'], axis=1, inplace=True)
    print( "count_ip_day_freq_h max value = ", df.count_ip_day_freq_h.max() )
    df['count_ip_day_freq_h'] = df['count_ip_day_freq_h'].astype('uint32')
    gc.collect()
    print( df.info() )

    print('group by : ip_day_hour')
    gp = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 
             'hour'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'count_ip_day_hour'})
    df = df.merge(gp, on=['ip','day','hour'], how='left')
    del gp
    print( "count_ip_day_hour max value = ", df.count_ip_day_hour.max() )
    df['count_ip_day_hour'] = df['count_ip_day_hour'].astype('uint16')
    gc.collect()
    print( df.info() )

    print('group by : ip_hour_os')
    gp = df[['ip', 'day', 'os', 'hour', 'channel']].groupby(by=['ip', 'os', 'day',
             'hour'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'count_ip_hour_os'})
    df = df.merge(gp, on=['ip','os','hour','day'], how='left')
    del gp
    print( "count_ip_hour_os max value = ", df.count_ip_hour_os.max() )
    df['count_ip_hour_os'] = df['count_ip_hour_os'].astype('uint16')
    gc.collect()
    print( df.info() )

    print('group by : ip_hh_app')
    gp = df[['ip', 'app', 'hour', 'day', 'channel']].groupby(by=['ip', 'app', 'day',
             'hour'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'count_ip_hh_app'})
    df = df.merge(gp, on=['ip','app','hour','day'], how='left')
    del gp
    print( "count_ip_hh_app max value = ", df.count_ip_hh_app.max() )
    df['count_ip_hh_app'] = df['count_ip_hh_app'].astype('uint16')
    gc.collect()
    print( df.info() )

    print('group by : ip_hour_device')
    gp = df[['ip', 'device', 'hour', 'day', 'channel']].groupby(by=['ip', 'device', 'day',
             'hour'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'count_ip_hour_device'})
    df = df.merge(gp, on=['ip','device','day','hour'], how='left')
    del gp
    print( "count_ip_hour_device max value = ", df.count_ip_hour_device.max() )
    df['count_ip_hour_device'] = df['count_ip_hour_device'].astype('uint32')
    gc.collect()
    print( df.info() )

    df.drop( ['day'], axis=1, inplace=True )
    gc.collect()
    print( df.info() )    
    print(df.describe())
    return( df )

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

def preprocess_confidence(train_df, test_df=None):
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
    for cols in ATTRIBUTION_CATEGORIES:
        
        # New feature name
        new_feature = '_'.join(cols) + '_confRate'
        print(new_feature)
        
        # Perform the groupby
        group_object = train_df.groupby(cols)
        
        # Group sizes
        group_sizes = group_object.size()
        
        # Print group size descriptives once
        if test_df is None:
            print(
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

#---------------------------------------------------------------------------------

print( "Train info before: ")
print( train_df.info() )
#train_df = prep_data( train_df )
#train_df = preprocess_confidence (train_df)
#gc.collect()
#print("vars and data type: ")
#train_df.info()
#train_df.describe()

def correlation_matrix(df):
    corr = df.corr()
    sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
    plt.savefig('corr-matrix.png')

#correlation_matrix(train_df)
#plot data

def histo(df):
    df.hist()
    plt.title('Histograms')
    plt.xlabel("Features")
    plt.ylabel("Frequency")
    plt.ioff()
    plt.savefig("histogram.png")
    plt.close()
def histo2(df):
    plt.hist(df, stack = False, bins = "auto")
    plt.title('Mean')
    plt.xlabel("Features")
    plt.ylabel("Frequency")
    plt.ioff()
    plt.savefig("histogram.png")
    plt.close()

def draw_histograms(df, variables, n_rows, n_cols):
    fig=plt.figure()
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[var_name].hist(bins=10,ax=ax)
        ax.set_title(var_name+" Distribution")
    fig.tight_layout()  # Improves appearance a bit.
    plt.ioff()  
    plt.savefig("histogram.png")
    plt.close()  
    
# Import library and dataset
 
# Method 1: on the same Axis
#sns.distplot( df["ip"] , color="skyblue", label="ip")
#sns.distplot( df["os"] , color="red", label="os")
#plt.legend()
 
#sns.plt.show()
# plot
def snsplot(df):
    f, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=True)
    g = sns.distplot( df["os"] , color="olive", ax=axes[0, 0])
    g.set_xlim(0,300)
    sns.distplot( df["app"] , color="gold", ax=axes[0, 1])
    sns.distplot( df["device"] , color="teal", ax=axes[1, 1])
    sns.distplot( df["channel"] , color="teal", ax=axes[1, 0])
#sns.distplot( df["click_time"] , color="teal", ax=axes[2, 1])
    plt.savefig('histogram.png', dpi = 300)
    plt.clf()
    
snsplot(train_df)

def plotip(df):
    sns.distplot(df["ip"], color="skyblue")
#sns.distplot( df["click_time"] , color="teal", ax=axes[2, 1])
    plt.savefig('histogram2.png', dpi = 300)
    plt.clf()
    
def plotattr(df):
    sns.distplot(df["is_attributed"] , color="skyblue")
    plt.savefig('histogram3.png', dpi = 300)
    

plotip(train_df)
plotattr(train_df)
#draw_histograms(train_df, train_df.columns,3,3)


#histo2(train_df)
#plot data against predictor 
#train_df.groupby('is_attributed').hour.value_counts().unstack(0).plot.barh()
#train_df.groupby('is_attributed').os.value_counts().unstack(0).plot.barh()
#train_df.groupby('is_attributed').channel.value_counts().unstack(0).plot.barh()
#train_df.groupby('is_attributed').device.value_counts().unstack(0).plot.barh()
#train_df.groupby('is_attributed').app.value_counts().unstack(0).plot.barh()

#print(train_df.head(n=15))

