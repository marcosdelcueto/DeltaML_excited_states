# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:45:00 2022
Author: Adam Coxson, PhD student, University of Liverpool
Department of Chemistry, Materials Innovation Factory, Levershulme Research Centre
Project: Delta ML Zindo
Module: FNN.py
Dependancies: Sklearn library, Pandas, Scipy, all other libraries are standard

This is demonstration code to obtain the results in the corresponding paper.
Running this will read and format all training and testing data. The network 
trains on 10506 molecules and is tested on a further 524. It uses the Morgan 
fingerprint and the Radial Distribution Function as inputs.

Note, on a i7 12700KF cpu, the training takes ~520 seconds (10 minutes) to run.
 
"""

import time, os, csv
import numpy as np
import pandas as pd
from math import sqrt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression


def preprocess(df,df_test,xcols,separate_test=True):
    '''
    Parameters
    ----------
    df: pd.dataframe
        training data
    df_test: pd.dataframe
        test data
    xcols: list
        list of features
    separate_test: bool
        whether to use test set or not

    Returns
    -------
    X: np.array
        features of training set
    y: np.array
        target property of training set
    E_lin: np.array
        Linear fit prediction for y from X
    X_test: np.array
        features of test set
    y_test: np.array
        target property of test set
    E_lin_test: np.array
        Linear fit prediction for y_test from X_test
    '''
    # Calculate y as Delta_E
    E_zindo = df['S1_ZINDO'].values.reshape(-1, 1)
    E_tddft = df['S1_TDDFT'].values.reshape(-1, 1)
    linear_regressor = LinearRegression()
    linear_regressor.fit(E_zindo, E_tddft)
    E_lin = linear_regressor.predict(E_zindo)
    y = []
    for i in range(len(E_lin)):
        Delta_E = E_tddft[i][0] - E_lin[i][0]
        y.append(Delta_E)
    y=np.array(y)
    X = df[xcols].values

    X_fp  = []
    X_RDF = []
    for i in range(len(X)):
        # Assign features
        fp  = eval(X[i][0])
        RDF = eval(X[i][1])
        X_fp.append(fp)
        X_RDF.append(RDF)
    X = np.c_[X_RDF,X_fp]

    if separate_test == False:
        X_test=None
        y_test=None
    elif separate_test == True:
        E_zindo_test = df_test['S1_ZINDO'].values.reshape(-1, 1)
        E_tddft_test = df_test['S1_TDDFT'].values.reshape(-1, 1)
        E_lin_test = linear_regressor.predict(E_zindo_test)
        y_test = []

        for i in range(len(E_tddft_test)):
            Delta_E = E_tddft_test[i][0] - E_lin_test[i][0]
            y_test.append(Delta_E)
        y_test=np.array(y_test)
        X_test = df_test[xcols].values
        X_test_fp = []
        X_test_RDF  = []

        for i in range(len(X_test)):
            # Assign features
            fp    =   eval(X_test[i][0])
            x_RDF =   eval(X_test[i][1])
            X_test_fp.append(fp)
            X_test_RDF.append(x_RDF)
        X_test = np.c_[X_test_RDF,X_test_fp]
        
        E_lin = np.squeeze(E_lin)
        E_lin_test = np.squeeze(E_lin_test)

    return X, y, E_lin, X_test, y_test, E_lin_test

    
def train_MLP(cfg, net_input, net_target, y_lin_pred, n_kfold):
    """
    Function to train a Multi-layer Perceptron (Feed-Forward Neural Network).

    Parameters
    ----------
    cfg : List of tuples, strings, and floats.
        Hyperparameters for the MLP architecture.
    net_input : Array, size (N, M)
        Input into network, N data points of M features each.
    net_target : Array, size N
        Target for network validation.
    y_lin_pred : Array, size N
        TDDFT energy prediction from linear fitting
    n_kfold : int
        number of folds in cross validation.

    Returns
    -------
    ML_func : object neural_network.MLPRegressor
        The optimised neural network model, can be used for further test data.
    results : Array of float64 (N,2) 
        For N data points, y_real and y_pred are output.
    metrics : list of floats
        Metrics such as the rms and median from y_real, y_pred.

    """
    
    t1=time.time()
    ML_func = MLPRegressor(hidden_layer_sizes=cfg[0], max_iter=cfg[1], batch_size=cfg[2], learning_rate_init=cfg[3], activation=cfg[4],
                           random_state=1, learning_rate='adaptive',solver='adam',verbose=False, tol=1e-4).fit(net_input, net_target)
    cv = KFold(n_splits=kfold,shuffle=True,random_state=0)
    y_pred = cross_val_predict(estimator=ML_func, X=net_input, y=net_target, cv=cv, n_jobs=kfold)
    t2=time.time()
    
    y_real = net_target
    rms  = sqrt(mean_squared_error(y_real, y_pred))
    rd,_   = pearsonr(y_real, y_pred)
    r,_    = pearsonr(y_lin_pred+y_real, y_lin_pred+y_pred)
    errors = abs(y_real - y_pred)
    median_error = np.median(errors)
    score = ML_func.score(net_input, net_target)
    results = np.array([y_real,y_pred]).T
    metrics = [rms, median_error, r, score]
    print('\nTraining metrics:')
    print('rms',rms)
    print('median_error',median_error)
    print("r",r)
    print("r-\u0394",rd)
    print("Score",score)
    print('Process took %.3f seconds' %(t2-t1))
    
    return ML_func, results, metrics

def test_MLP(ML_func, net_input, net_target, y_lin_pred, n_kfold):
    """
    Function to test a Multi-layer Perceptron on unseen data after training.

    Parameters
    ----------
    ML_func : object neural_network.MLPRegressor
        The optimised neural network model, obtained from previous training.
    net_input : Array, size (N, M)
        Input into network, N data points of M features each.
    net_target : Array, size N
        Target for network validation.
    y_lin_pred : Array, size N
        TDDFT energy prediction from linear fitting
    n_kfold : int
        number of folds in cross validation.

    Returns
    -------
    results : Array of float64 (N,2) 
        For N data points, y_real and y_pred are output.
    metrics : list of floats
        Metrics such as the rms and median from y_real, y_pred.
    """

    t1=time.time()
    y_pred = ML_func.predict(X=net_input)
    t2=time.time()
    
    y_real = net_target
    rms  = sqrt(mean_squared_error(y_real, y_pred))
    rd,_   = pearsonr(y_real, y_pred)
    r,_    = pearsonr(y_lin_pred+y_real, y_lin_pred+y_pred)
    errors = abs(y_real - y_pred)
    median_error = np.median(errors)
    score = ML_func.score(net_input, net_target)
    results = np.array([y_real, y_pred]).T
    metrics = [rms, median_error, r, score]
    print('\nTesting metrics:')
    print('rms',rms)
    print('median_error',median_error)
    print("r",r)
    print("r-\u0394",rd)
    print("Score",score)
    print('Process took %.6f seconds' %(t2-t1))

    return results, metrics
    
def write_to_csv(df, E_ml, filename='data_save'):
    """
    Parameters
    ----------
    df: pd.dataframe
        training or testing data
    E_ml : np.array, 1D
        FNN prediction of TDDFT energy.
    filename : TYPE, optional
        filepath to save data to. The default is 'data_save'.

    Returns
    -------
    None.
    """

    E_zindo = df['S1_ZINDO'].values
    E_tddft = df['S1_TDDFT'].values
    data = [E_zindo, E_tddft, E_ml]
    data = np.asarray(data).T
    writer = csv.writer(open(filename,'w'),lineterminator ="\n")
    writer.writerow(["S1_ZINDO","S1_TDDFT","S1_ML"])
    writer.writerows(data)
    print("Written to",filename,"successfully.")
    return None


################################################################################
if __name__ == '__main__':
    
    # Set file names
    train_csv_file = 'train_data.csv'
    test_csv_file  = 'test_data.csv'
    train_results_csv = 'results_train.csv'
    test_results_csv = 'results_test.csv'
    folder_path = os.getcwd() + "/database/"
    results_path = os.getcwd() + "/reproduce_Fig3/"
    
    separate_test = True
    xcols = ['fingerprint','RDF']
    # [(neurons layer 1, neurons layer 2), iterations, batch size, learning rate, activation]
    cfg = [(703,	312),	1000,	100,	0.001118462,	'relu'] # FNN hyperparameters
    kfold=10
    
    # Preprocess
    print("Data processing")
    df = pd.read_csv(folder_path+train_csv_file)
    df_test = pd.read_csv(folder_path+test_csv_file)
    X,y,E_lin, X_test,y_test,E_lin_test = preprocess(df,df_test,xcols,separate_test)
    print('Processing done\nTraining Network')
    
    # Network training on RDF and FP of 10506 molecules
    optimised_network, results, metrics = train_MLP(cfg, net_input=X, net_target=y, y_lin_pred=E_lin, n_kfold=kfold)
    
    # Optimised network tested on 524 molecules.
    results_test, metrics_test = test_MLP(optimised_network, net_input=X_test,
                                          net_target=y_test, y_lin_pred=E_lin_test,  n_kfold=kfold)
    
    print("\nSaving results to",results_path)
    write_to_csv(df=df, E_ml=E_lin+results[:,1], filename=results_path+train_results_csv)
    write_to_csv(df=df_test, E_ml=E_lin_test+results_test[:,1], filename=results_path+test_results_csv)
  
