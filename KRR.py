#!/usr/bin/env python3
#################################################################################
import sys
import ast
import time
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge

def main():

    # Set up input values
    gamma_fp = 1.0
    gamma_rdf = 0.01
    regularization = 1.0
    separate_test = True
    # set db file names
    train_csv_file = 'train_data.csv'
    test_csv_file  = 'test_data.csv'
    xcols = ['fingerprint','RDF']
    # Preprocess
    df = pd.read_csv(train_csv_file)
    df_test = pd.read_csv(test_csv_file)
    X,y,X_test,y_test = preprocess(df,df_test,xcols,separate_test)
    print('Preprocessing done')
    # ML
    rms = func_ML(X,y,X_test,y_test,separate_test,gamma_fp,gamma_rdf,regularization)
    return

def preprocess(df,df_test,xcols,separate_test):
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
        wether to use test set or not

    Returns
    -------
    X: np.array
        features of training set
    y: np.array
        target property of training set
    X_test: np.array
        features of test set
    y_test: np.array
        target property of test set
    '''
    # Calculate y as Delta_E
    E_zindo = df['S1_ZINDO'].values.reshape(-1, 1)
    E_tddft = df['S1_TDDFT'].values.reshape(-1, 1)
    linear_regressor = LinearRegression()
    linear_regressor.fit(E_zindo, E_tddft)
    E_lin = linear_regressor.predict(E_zindo)
    y = []
    for i in range(len(E_lin)):
        interm = []
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
    X = np.c_[X_fp,X_RDF]

    if separate_test == False:
        X_test=None
        y_test=None
    elif separate_test == True:
        E_zindo_test = df_test['S1_ZINDO'].values.reshape(-1, 1)
        E_tddft_test = df_test['S1_TDDFT'].values.reshape(-1, 1)
        E_lin_test = linear_regressor.predict(E_zindo_test)
        y_test = []

        for i in range(len(E_tddft_test)):
            interm = []
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
        X_test = np.c_[X_test_fp,X_test_RDF]

    return X,y,X_test,y_test

def func_ML(X,y,X_test,y_test,separate_test,gamma_fp,gamma_rdf,regularization):
    '''
    Parameters
    ----------
    X: np.array
        features of training set
    y: np.array
        target property of training set
    X_test: np.array
        features of test set
    y_test: np.array
        target property of test set
    separate_test: bool
        wether to use test set or not
    gamma_fp: float
        weight of fp in kernel
    gamma_rdf: float
        weight of fp in kernel
    regularization: float
        value of regularization parameter

    Returns
    -------
    rms: float
        value of rmse of real and predicted values
    '''

    # Set kernel
    kernel = build_hybrid_kernel(gamma_fp=gamma_fp, gamma_rdf=gamma_rdf)
    ML_algorithm = KernelRidge(alpha=regularization, kernel=kernel)
    # Do 10-fold CV
    if separate_test == False:
        # define CV
        cv = KFold(n_splits=10,shuffle=True,random_state=0)
        # calculate predicted values
        y_predicted = cross_val_predict(ML_algorithm, X, y, cv=cv, n_jobs=1)
        y_real = y
        # calculate prediction metrics
        rms  = sqrt(mean_squared_error(y_real, y_predicted))
        # print predictions
        for i in range(len(y_real)):
            print(y_real[i],y_predicted[i])
    # Predict test set
    elif separate_test == True:
        ML_algorithm.fit(X, y)
        print('Fit done')
        y_predicted = ML_algorithm.predict(X_test)
        print('Prediction done')
        y_real = y_test
        rms  = sqrt(mean_squared_error(y_real, y_predicted))
        # print predictions
        for i in range(len(y_real)):
            print(y_real[i],y_predicted[i])
    return rms

def build_hybrid_kernel(gamma_fp, gamma_rdf):
    '''
    Parameters
    ----------
    gamma_fp: float
        value of gamma_fp parameter
    gamma_rdf: float
        value of gamma_rdf parameter

    Returns
    -------
    hybrid_kernel: callable
        function to compute the hybrid gaussian/Tanimoto kernel given values.
    '''

    def hybrid_kernel(_x1, _x2):
        '''
        Function to compute a hybrid gaussian/Tanimoto (KRR).
        Based on Daniele's function

        Parameters
        ----------
        _x1: np.array.
            data point.
        _x2: np.array.
            data point.

        Returns
        -------
        K: np.float.
            Kernel matrix element.
        '''
        # sanity check
        if _x1.ndim != 1 or _x2.ndim != 1:
            print('ERROR: KRR kernel was expecting 1D vectors!')
            sys.exit()
        if gamma_fp != 0.0:
            # FP
            x1_fp = _x1[0:2048]
            x2_fp = _x2[0:2048]
            D_fp = tanimoto_dist(x1_fp,x2_fp)
            K_fp = np.exp(-gamma_fp*(D_fp**2))
        else:
            K_fp=1.0
        if gamma_rdf != 0.0:
            # RDF
            x1_rdf = _x1[2048:]
            x2_rdf = _x2[2048:]
            D_rdf = np.linalg.norm(x1_rdf-x2_rdf)
            K_rdf = np.exp(-gamma_rdf*(D_rdf**2))
        else:
            K_rdf=1.0
        # Element-wise multiplication
        K = K_fp * K_rdf
        return K

    return hybrid_kernel

def tanimoto_dist(x1,x2):
    '''
    Function to calculate tanimoto distance between two fingerprint vectors

    Parameters
    ----------
    x1: np.array
        data point
    x2: np.array
        data point

    Returns
    -------
    D: float
        Distance using Tanimoto similarity index
    '''
    # calculate T
    T = ( np.dot(x1,x2) ) / ( np.dot(x1,x1) + np.dot(x2,x2) - np.dot(x1,x2) )
    # calculate distance
    D = 1-T
    return D

################################################################################
if __name__ == '__main__':
    main()
