import argparse
import gc

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation 
from keras import optimizers
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import preprocessing as pp

def main(train_file, test_file):

    train_df, val_df, test_df = pp.run(train_file, test_file)
    
    '''df = train_df
    df = df.append(val_df)
    
    df['ip_cut'] = pd.cut(df.ip,15)
    
    df = df.drop(['ip','click_id'], axis = 1)
    
    categorical_columns = ['app', 'device', 'os', 'channel', 'ip_cut', 'hour']
    
    df = pd.get_dummies(df, columns = categorical_columns)
    
    train_df = df[0:len(df)-len(val_df)]
    val_df = df[len(df)-len(val_df):len(df)]
    
    del df 
    gc.collect()'''
    
    y_train = train_df.is_attributed
    y_val = val_df.is_attributed
    
    x_train = train_df.drop(['is_attributed'], axis = 1)
    x_val = val_df.drop(['is_attributed'], axis = 1)
    
    x_train = np.array(x_train)
    x_val = np.array(x_val)
    
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    
    x_test = np.array(test_df)
    
    del train_df
    del val_df
    del test_df
    gc.collect()

    print("Training part...")
    print("Create model...")
    
    optimizer = optimizers.adam(lr = 0.0005, decay = 0.000001)

    model = Sequential()
    model.add(Dense(64, input_dim=x_train.shape[1],
                    kernel_initializer='normal',
                    #kernel_regularizer=regularizers.l2(0.02),
                    activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128,
                    #kernel_regularizer=regularizers.l2(0.02),
                    activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    print("Compile model...")
    model.compile(loss="binary_crossentropy", optimizer=optimizer)

    print("Fit model...")
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=3, batch_size=64)
    
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    #Predict on validation set
    predictions_NN_prob = model.predict(x_val)
    predictions_NN_prob = predictions_NN_prob[:,0]
    
    predictions_NN_01 = np.where(predictions_NN_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output

    #Print accuracy
    acc_NN = accuracy_score(y_val, predictions_NN_01)
    print('Overall accuracy of Neural Network model:', acc_NN)

    #Predict on validation set
    predictions_NN_prob = model.predict(x_test)
    result = pd.DataFrame(predictions_NN_prob)
    result.to_csv(job_dir + '/NN.csv',index=False)
    
    
if __name__ == '__main__':
    '''parser = argparse.ArgumentParser()
    
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

    main(args.train_file, args.test_file, args.job_dir)'''
    main("C:/Users/Pauline/Documents/GitHub/fraud_detection/trainer/data/train_sample.csv", "C:/Users/Pauline/Documents/GitHub/fraud_detection/trainer/data/test.csv")
