# Based on Alexander Kireev's deep learning model:
# https://www.kaggle.com/alexanderkireev/experiments-with-imbalance-nn-architecture?scriptVersionId=3419429/code


import argparse
import numpy as np
import pandas as pd
import gc
import logging

from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D, Conv1D
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import Adam

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from IPython.display import clear_output

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

class PlotLosses(Callback):
    
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.grid()
        plt.legend()
        plt.show();
        
def get_values(df):
    return df.columns.values.tolist() 

def get_keras_data(dataset):
    
    variables = get_values(dataset)
       
    X = dict([(var, np.array(dataset[var])) for var in variables])
 
    return X

def NN(train_df, val_df, test_df):
    logging.info('Neural Network preprocessing')
    
    '''if val_df is None:
        val_df = train_df[len(train_df)-10000:len(train_df)]
        train_df = train_df[0:len(train_df)-10000]'''
        
    y_train = train_df['is_attributed'].values
    train_df = train_df.drop(['is_attributed'], axis = 1)
    
    if val_df is not None:
        y_val = val_df['is_attributed'].values 
        val_df = val_df.drop(['is_attributed'], axis = 1)
        val_df = get_keras_data(val_df)
        
    list_variables = get_values(train_df)
        
    max_var = []
    if test_df is not None:
        for i, var in enumerate(list_variables):
            max_var.append(np.max([train_df[var].max(), test_df[var].max()])+1)    
        train_df = get_keras_data(train_df)
    else:
        for i, var in enumerate(list_variables):
            max_var.append(train_df[var].max()+1)    
        train_df = get_keras_data(train_df)
    
    logging.info('Model is creating...')
    emb_n = 50
    dense_n = 1000
    
    in_var = []
    emb_var = []    
    for i, var in enumerate(list_variables):
        in_var.append(Input(shape=[1], name = var))
        emb_var.append(Embedding(max_var[i], emb_n)(in_var[i]))
    
    fe = concatenate([emb for emb in emb_var])
    s_dout = SpatialDropout1D(0.2)(fe)
    fl1 = Flatten()(s_dout)
    conv = Conv1D(100, kernel_size=4, strides=1, padding='same')(s_dout)
    fl2 = Flatten()(conv)
    concat = concatenate([(fl1), (fl2)])
    x = Dropout(0.2)(Dense(dense_n,activation='relu')(concat))
    x = Dropout(0.2)(Dense(dense_n,activation='relu')(x))
    outp = Dense(1,activation='sigmoid')(x)
    
    model = Model(inputs=[var for var in in_var], outputs=outp)
    
    logging.info('Model is compiling...')
    #parameters 
    batch_size = 50000
    epochs = 10
    exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
    steps = int(len(list(train_df)[0]) / batch_size) * epochs
    lr_init, lr_fin = 0.002, 0.0002
    lr_decay = exp_decay(lr_init, lr_fin, steps)
    optimizer_adam = Adam(lr=0.002, decay=lr_decay)
    
    model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy'])
    model.summary()
    
    callbacks = [ModelCheckpoint('best_model_NN.h5', save_best_only=True)]#, PlotLosses()]#, ReduceLROnPlateau(patience=1, min_lr=lr_fin), EarlyStopping(patience=1)]
    
    logging.info('Model is training...')
    class_weight = {0:.01,1:.99} # magic
    model.fit(train_df, y_train, validation_split=0.1, batch_size=batch_size, epochs=epochs, class_weight=class_weight, shuffle=True, verbose=2, callbacks=callbacks)
    del train_df, y_train; gc.collect()
    
    if val_df is not None:
        logging.info('Prediction on validation set')
        predictions_NN_prob = model.predict(val_df, batch_size=batch_size, verbose=2)
        del val_df; gc.collect()
        predictions_NN_prob = predictions_NN_prob[:,0]
        
        predictions_NN = np.where(predictions_NN_prob > 0.5, 1, 0)
        
        #print accuracy
        acc_NN = accuracy_score(y_val, predictions_NN)
        print('Overall accuracy of Neural Network model:', acc_NN)
    
    if test_df is not None:
        logging.info('Prediction on test set')
        sub = pd.DataFrame()
        sub['click_id'] = test_df['click_id'].astype('int')
        test_df = test_df.drop(['click_id'], axis=1)
        test_df = get_keras_data(test_df)
        
        sub['is_attributed'] = model.predict(test_df, batch_size=batch_size, verbose=2)
        del test_df; gc.collect()
        logging.info("Writing....")
        sub.to_csv('sub_NN.csv',index=False)
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
