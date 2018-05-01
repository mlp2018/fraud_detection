# Based on Alexander Kireev's deep learning model:
# https://www.kaggle.com/alexanderkireev/experiments-with-imbalance-nn-architecture?scriptVersionId=3419429/code


import argparse
import numpy as np
import pandas as pd
import trainer.preprocessing as pp
import gc

from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D, Conv1D
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam

from sklearn.metrics import accuracy_score

def get_keras_data(dataset):
    X = {
        'app': np.array(dataset.app),
        'ch': np.array(dataset.channel),
        'dev': np.array(dataset.device),
        'os': np.array(dataset.os),
        'h': np.array(dataset.hour),
    }
    return X

def main(train_file, test_file, job_dir):
    
    print('Global preprocessing')
    train_df, val_df, test_df = pp.run(train_file, test_file)
    
    print('Neural Network preprocessing')
    y_train = train_df['is_attributed'].values
    train_df.drop(['click_id', 'ip','is_attributed'],axis = 1)
    
    y_val = val_df['is_attributed'].values 
    val_df.drop(['click_id', 'ip','is_attributed'],axis = 1)
    val_df = get_keras_data(val_df)
    
    max_app = np.max([train_df['app'].max(), test_df['app'].max()])+1
    max_ch = np.max([train_df['channel'].max(), test_df['channel'].max()])+1
    max_dev = np.max([train_df['device'].max(), test_df['device'].max()])+1
    max_os = np.max([train_df['os'].max(), test_df['os'].max()])+1
    max_h = np.max([train_df['hour'].max(), test_df['hour'].max()])+1
    
    train_df = get_keras_data(train_df)
    
    print('Model is creating...')
    emb_n = 50
    dense_n = 1000
    
    in_app = Input(shape=[1], name = 'app')
    emb_app = Embedding(max_app, emb_n)(in_app)
    in_ch = Input(shape=[1], name = 'ch')
    emb_ch = Embedding(max_ch, emb_n)(in_ch)
    in_dev = Input(shape=[1], name = 'dev')
    emb_dev = Embedding(max_dev, emb_n)(in_dev)
    in_os = Input(shape=[1], name = 'os')
    emb_os = Embedding(max_os, emb_n)(in_os)
    in_h = Input(shape=[1], name = 'h')
    emb_h = Embedding(max_h, emb_n)(in_h) 
    
    fe = concatenate([(emb_app), (emb_ch), (emb_dev), (emb_os), (emb_h)])
    s_dout = SpatialDropout1D(0.2)(fe)
    fl1 = Flatten()(s_dout)
    conv = Conv1D(100, kernel_size=4, strides=1, padding='same')(s_dout)
    fl2 = Flatten()(conv)
    concat = concatenate([(fl1), (fl2)])
    x = Dropout(0.2)(Dense(dense_n,activation='relu')(concat))
    x = Dropout(0.2)(Dense(dense_n,activation='relu')(x))
    outp = Dense(1,activation='sigmoid')(x)
    
    model = Model(inputs=[in_app,in_ch,in_dev,in_os,in_h], outputs=outp)
    
    print('Model is compiling...')
    batch_size = 50000
    epochs = 2
    exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
    steps = int(len(list(train_df)[0]) / batch_size) * epochs
    lr_init, lr_fin = 0.002, 0.0002
    lr_decay = exp_decay(lr_init, lr_fin, steps)
    optimizer_adam = Adam(lr=0.002, decay=lr_decay)
    
    model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy'])
    
    model.summary()
    
    print('Model is training...')
    class_weight = {0:.01,1:.99} # magic
    model.fit(train_df, y_train, batch_size=batch_size, epochs=2, class_weight=class_weight, shuffle=True, verbose=2)
    del train_df, y_train; gc.collect()
    #model.save_weights('imbalanced_data.h5')
    
    print('Prediction on validation set')
    #Predict on validation set
    predictions_NN_prob = model.predict(val_df, batch_size=batch_size, verbose=2)
    del val_df; gc.collect()
    predictions_NN_prob = predictions_NN_prob[:,0]
    
    predictions_NN = np.where(predictions_NN_prob > 0.5, 1, 0)
    
    #Print accuracy
    acc_NN = accuracy_score(y_val, predictions_NN)
    print('Overall accuracy of Neural Network model:', acc_NN)
    
    print('Prediction on test set')
    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')
    test_df.drop(['click_id', 'ip','is_attributed'], axis=1)
    test_df = get_keras_data(test_df)
    
    sub['is_attributed'] = model.predict(test_df, batch_size=batch_size, verbose=2)
    del test_df; gc.collect()
    print("Writing....")
    sub.to_csv(job_dir + '/imbalanced_data.csv',index=False)
    print("Done...")
    print(sub.info())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Input arguments
    parser.add_argument(
      '--train-file',
      help='GCS or local paths to training data',
      required=True
    )

    parser.add_argument(
      '--test-file',
      help='GCS or local paths to test data',
      required=True
    )

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    
    args = parser.parse_args()
    arguments = args.__dict__
    print('args: {}'.format(arguments))

    main(args.train_file, args.test_file, args.job_dir)
