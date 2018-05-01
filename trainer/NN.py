# Based on Alexander Kireev's deep learning model:
# https://www.kaggle.com/alexanderkireev/experiments-with-imbalance-nn-architecture?scriptVersionId=3419429/code


import argparse
import numpy as np
import pandas as pd
import gc
import logging

from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D, Conv1D
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam

from sklearn.metrics import accuracy_score

from trainer.cross_validation import stratified_kfold, cross_val_score
import trainer.preprocessing as pp

class StoreLoggingLevel(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError('`nargs` is not supported.')
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        level = getattr(logging, value.upper(), None)
        if not isinstance(level, int):
            raise ValueError('Invalid log level: {}'.format(value))
        setattr(namespace, self.dest, level)


def make_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--train-file', help='Path to training data', required=True)
    parser.add_argument(
      '--valid-file', help='Path to validation data', required=False)
    parser.add_argument(
      '--test-file', help='Path to test data', required=False)
    parser.add_argument(
        '--job-dir',
        help='Directory where to store checkpoints and exported models.',
        default='.')
    parser.add_argument(
        '--log', help='Logging level', default=logging.DEBUG,
        action=StoreLoggingLevel)
    return parser

def get_keras_data(dataset):
    X = {
        'app': np.array(dataset.app),
        'ch': np.array(dataset.channel),
        'dev': np.array(dataset.device),
        'os': np.array(dataset.os),
        'h': np.array(dataset.hour),
    }
    return X

def NN(train_df, val_df, test_df):
    logging.info('Neural Network preprocessing')
    y_train = train_df['is_attributed'].values
    train_df.drop(['ip','is_attributed'], axis = 1)
    
    if val_df is not None:
        y_val = val_df['is_attributed'].values 
        val_df.drop(['click_id', 'ip','is_attributed'], axis = 1)
        val_df = get_keras_data(val_df)
    
    if test_df is not None:
        max_app = np.max([train_df['app'].max(), test_df['app'].max()])+1
        max_ch = np.max([train_df['channel'].max(), test_df['channel'].max()])+1
        max_dev = np.max([train_df['device'].max(), test_df['device'].max()])+1
        max_os = np.max([train_df['os'].max(), test_df['os'].max()])+1
        max_h = np.max([train_df['hour'].max(), test_df['hour'].max()])+1
    else:
        max_app = train_df['app'].max()+1
        max_ch = train_df['channel'].max()+1
        max_dev = train_df['device'].max()+1
        max_os = train_df['os'].max()+1
        max_h = train_df['hour'].max()
    
    train_df = get_keras_data(train_df)
    
    logging.info('Model is creating...')
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
    
    logging.info('Model is compiling...')
    batch_size = 50000
    epochs = 2
    exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
    steps = int(len(list(train_df)[0]) / batch_size) * epochs
    lr_init, lr_fin = 0.002, 0.0002
    lr_decay = exp_decay(lr_init, lr_fin, steps)
    optimizer_adam = Adam(lr=0.002, decay=lr_decay)
    
    model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy'])
    
    model.summary()
    
    logging.info('Model is training...')
    class_weight = {0:.01,1:.99} # magic
    model.fit(train_df, y_train, batch_size=batch_size, epochs=2, class_weight=class_weight, shuffle=True, verbose=2)
    del train_df, y_train; gc.collect()
    #model.save_weights('imbalanced_data.h5')
    
    if val_df is not None:
        logging.info('Prediction on validation set')
        #Predict on validation set
        predictions_NN_prob = model.predict(val_df, batch_size=batch_size, verbose=2)
        del val_df; gc.collect()
        predictions_NN_prob = predictions_NN_prob[:,0]
        
        predictions_NN = np.where(predictions_NN_prob > 0.5, 1, 0)
        
        #print accuracy
        acc_NN = accuracy_score(y_val, predictions_NN)
        logging.info('Overall accuracy of Neural Network model:', acc_NN)
    
    if test_df is not None:
        logging.info('Prediction on test set')
        sub = pd.DataFrame()
        sub['click_id'] = test_df['click_id'].astype('int')
        test_df.drop(['click_id', 'ip'], axis=1)
        test_df = get_keras_data(test_df)
        
        sub['is_attributed'] = model.predict(test_df, batch_size=batch_size, verbose=2)
        del test_df; gc.collect()
        logging.info("Writing....")
        sub.to_csv('imbalanced_data.csv',index=False)
        logging.info("Done...")
        logging.info(sub.info())


def main():
    args = make_args_parser().parse_args()
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s',
                        level=args.log)
    
    logging.info('Preprocessing...')
    # Load training data set, i.e. "the 90%"
    train_df = pp.load_train(args.train_file)
    # Load validation data set, i.e. "the 10%"
    val_df = pp.load_train(args.valid_file) if args.valid_file is not None \
        else None
    # Load the test data set, i.e. data for which we need to make predictions.
    test_df = pp.load_test(args.test_file) if args.test_file is not None \
        else None
    
    NN(train_df, val_df, test_df)
    
if __name__ == '__main__':
    main()
