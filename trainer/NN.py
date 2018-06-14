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
'''import matplotlib
import matplotlib.pyplot as plt
from IPython.display import clear_output'''

#from cross_validation import stratified_kfold, cross_val_score
import trainer.preprocessing as pp
from tensorflow.python.lib.io import file_io

def make_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--train-file', help='Path to training data', required=False)
    parser.add_argument(
      '--valid-file', help='Path to validation data', required=False)
    parser.add_argument(
      '--test-file', help='Path to test data', required=False)
    parser.add_argument(
      '--sub-file', help='Path to write the submission in a file', required=False)
    parser.add_argument(
        '--job-dir',
        help='Directory where to store checkpoints and exported models.',
        default='.')
    '''parser.add_argument(
        '--log', help='Logging level', default=logging.DEBUG,
        action=StoreLoggingLevel)'''
    return parser

'''class PlotLosses(Callback):
    """
    Plot the loss and the accuracy of the training set and the validation set
    """
    
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
        plt.show();'''
        
def get_values(df):
    """
    Return a list of the column's name
    """
    return df.columns.values.tolist() 

def get_keras_data(dataset):
    """
    Split the data according to the column into an numpy array 
    """
    
    variables = get_values(dataset)
       
    X = dict([(var, np.array(dataset[var])) for var in variables])
 
    return X

def NN(train_df, val_df, test_df, sub_path):
    """
    Main function of the Neural Network 
    """
    logging.info('Neural Network preprocessing')
    
    '''if val_df is None:
        val_df = train_df[len(train_df)-10000:len(train_df)]
        train_df = train_df[0:len(train_df)-10000]'''
    
    if train_df is not None: 
        y_train = train_df['is_attributed'].values
        train_df = train_df.drop('is_attributed', axis = 1)
        train_df = train_df.drop('attributed_time', axis = 1) 
        #train_df = train_df.drop('click_time', axis = 1) #only if no preprocessing
        gc.collect()
        if val_df is not None:
            y_val = val_df['is_attributed'].values 
            val_df = val_df.drop(['is_attributed'], axis = 1)
            val_df = get_keras_data(val_df)
            
        list_variables = get_values(train_df)
        print(list_variables)
    
        logging.info('Model is creating...')    
        
        max_var = []
        if test_df is not None:
            for i, var in enumerate(list_variables):
                max_var.append(np.max([train_df[var].max(), test_df[var].max()])+1)    
            train_df = get_keras_data(train_df)
        else:
            for i, var in enumerate(list_variables):
                max_var.append(train_df[var].max()+1)    
            train_df = get_keras_data(train_df)
        
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
        #conv = Conv1D(100, kernel_size=4, strides=1, padding='same')(s_dout)
        dl = Dense(100)(s_dout)
        fl2 = Flatten()(dl)
        concat = concatenate([(fl1), (fl2)])
        x = Dropout(0.2)(Dense(dense_n,activation='relu')(concat))
        x = Dropout(0.2)(Dense(dense_n,activation='relu')(x))
        outp = Dense(1,activation='sigmoid')(x)
        
        model = Model(inputs=[var for var in in_var], outputs=outp)
        
        logging.info('Model is compiling...')
        #parameters 
        batch_size = 50000
        epochs = 2 #12 for sample_train
        exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
        steps = int(len(list(train_df)[0]) / batch_size) * epochs
        lr_init, lr_fin = 0.002, 0.0002
        lr_decay = exp_decay(lr_init, lr_fin, steps)
        optimizer_adam = Adam(lr=lr_init, decay=lr_decay)
        
        model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy'])
        model.summary()
        
        logging.info('Model is training...')
        
        model.fit(train_df, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2, validation_split=0.1)
        del train_df, y_train; gc.collect()
        '''model.save_weights('model_NN.h5')
        with file_io.FileIO('model_NN.h5', mode='r') as input_f:
            with file_io.FileIO('gs://bag_of_students/NN/model_NN.h5', mode='w+') as output_f:
                output_f.write(input_f.read())
                print("Saved model_NN.h5 to GCS")'''
    
        #from keras.models import load_weights
        #model.load_weights('C:/Users/Pauline/Documents/GitHub/fraud_detection/trainer/NN_model_NN.h5')
    
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
        with file_io.FileIO(sub_path, mode='wb') as fout: #gs://bag_of_students/NN/
            sub.to_csv(fout,index=False)
        logging.info("Done...")
        logging.info(sub.info())


def main():
    args = make_args_parser().parse_args()
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s')#,level=args.log)
        
    logging.info('Preprocessing...')
    # Load training data set, i.e. "the 90%"
    train_df = pp.load_train(args.train_file) if args.train_file is not None \
        else None
    
    # Load validation data set, i.e. "the 10%"
    val_df = pp.load_train(args.valid_file) if args.valid_file is not None \
        else None
    
    # Load the test data set, i.e. data for which we need to make predictions.
    test_df = pp.load_test(args.test_file) if args.test_file is not None \
        else None
        
    path_sub_file = args.sub_file
    
    NN(train_df, val_df, test_df, path_sub_file)
    
if __name__ == '__main__':
    main()
