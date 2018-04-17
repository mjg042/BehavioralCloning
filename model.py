# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 05:21:50 2018

@author: mgriffi3
"""

"""
import pip
packages = ['utils']
pip.main(['install'] + packages + ['--upgrade'])
"""

"""
wd = "C:\\cygwin64\\home\\mgriffi3\\projects\\ImageAnalysis\\udacity\\transferLearning\\CarND-Behavioral-Cloning-P3"
import os
os.chdir(wd)
"""

"""
REFERENCES
    
    https://github.com/naokishibuya/car-behavioral-cloning

"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, BatchGenerator
import argparse
import os

np.random.seed(0)

def GetData(args):
    """
    Load training data and split it into training and validation set
    """
    df = pd.read_csv(os.path.join(args.dataDir, 'driving_log.csv'))
    df.columns = ['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed']

    X = df[['center', 'left', 'right']].values
    y = df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.testSize, random_state=0)

    return X_train, X_valid, y_train, y_valid

def BuildModel(args):
    """
    based on naokishibuya
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 7, 7, activation='elu', subsample=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(48, 3, 3, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 1, 1, activation='elu'))
    model.add(Conv2D(64, 1, 1, activation='elu'))
    model.add(Dropout(args.keepProb))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model

def TrainModel(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model 
    """
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.saveBestModel,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learningRate))

    model.fit_generator(BatchGenerator(args.dataDir, X_train, y_train, args.batchSize, True),
                        args.samplesPerEpoch,
                        args.nbEpoch,
                        max_q_size=1,
                        validation_data=BatchGenerator(args.dataDir, X_valid, y_valid, args.batchSize, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)

def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'

def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    #parser.add_argument('-d', help='data directory',        dest='data Dir',           type=str,   default='data\\Sim\\Lap2')
    parser.add_argument('-d', help='data directory',        dest='dataDir',           type=str,   default='data\\Sim\\Combo')
    #parser.add_argument('-d', help='data directory',        dest='dataDir',           type=str,   default='data\\Sim\\Jungle2')
    parser.add_argument('-t', help='test size fraction',    dest='testSize',          type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keepProb',          type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nbEpoch',           type=int,   default=4)
    parser.add_argument('-s', help='samples per epoch',     dest='samplesPerEpoch',   type=int,   default=1000)
    parser.add_argument('-b', help='batch size',            dest='batchSize',         type=int,   default=100)
    parser.add_argument('-o', help='save best models',      dest='saveBestModel',     type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learningRate',      type=float, default=1.0e-4)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = GetData(args)
    model = BuildModel(args)
    TrainModel(model, args, *data)


if __name__ == '__main__':
    main()