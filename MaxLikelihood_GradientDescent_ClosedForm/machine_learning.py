import numpy as np
import pandas as pd
import sys
from numpy.linalg import inv,pinv

alpha=0.001
lmbda=0.01
tolerance_gd=10**(-20)

def error(X,y,w):
    return (1. / X.shape[0]) * \
           (np.sum((np.dot(X, w) - y) ** 2.))
def cost(X, y, w):
    return (1. / X.shape[0]) * \
           (np.sum((np.dot(X, w) - y) ** 2.) + lmbda * np.dot(w.T, w))
def grad_loss(X, y,w):
    return ((2. * alpha)/ X.shape[0]) * \
            (np.dot(X.T, (X.dot(w) - y)) + lmbda * w)

def gradient_descent(X, y, w):
    L_complete = []
    i=0
    while(True):
        L_complete.append(cost(X,y,w))
        i=i+1
        w = w - ((2. * alpha)/X.shape[0]) * \
            (np.dot(X.T, (X.dot(w) - y)) + lmbda * w)
        if(abs(cost(X,y,w)[0][0]-L_complete[i-1])<tolerance_gd):
            break
    return w, L_complete

def predict(X,w):
    return X.dot(w)

def output(fname,test_data,y):
         test_data['Slowness in traffic (%) Predicted'] = y
         test_data.to_csv(fname,sep=';',index=False,decimal=',')
         print("Predicted values are given in "+fname) 

df = pd.read_csv('Data.csv',sep=';',decimal=',').sample(frac=1)
training_data = df.iloc[:int(0.8*len(df))].reset_index(drop=True)
testing_data = df.iloc[int(0.8*len(df)):].reset_index(drop=True)

x_tr=training_data.drop(columns=['Slowness in traffic (%)']).values
x_te=testing_data.drop(columns=['Slowness in traffic (%)']).values
x_train = np.hstack((np.ones(x_tr.shape[0])[np.newaxis].T, x_tr))
x_test = np.hstack((np.ones(x_te.shape[0])[np.newaxis].T, x_te))
y_train=training_data[['Slowness in traffic (%)']].values
y_test=testing_data[['Slowness in traffic (%)']].values

#Gradient Descent
w_gd = np.zeros((training_data.shape[1], 1))
w_gd,L_complete=gradient_descent(x_train,y_train,w_gd) 
#print(w_gd)
print("Mean Square Error for Gradient Descent is : " + str(error(x_test,y_test,w_gd)))
y_gd=predict(x_test,w_gd)
output("gradient_descent.csv",testing_data,y_gd)

#Standard cost minimization in closed form
idn=np.identity(np.dot(x_train.T,x_train).shape[1])
w_cf=np.asarray(np.dot(np.dot(inv(np.matrix(np.dot(x_train.T,x_train)+np.dot(lmbda,idn))),x_train.T),y_train))
#print(w_cf)
print("Mean Square Error for Closed Form is : "+str(error(x_test,y_test,w_cf)))
y_cf=predict(x_test,w_cf)
output("closed_form.csv",testing_data,y_cf)

#Maximum Likelihood Estimation
w_estimated = np.zeros((training_data.shape[1], 1))
X_mean = np.mean(x_train, axis=0)
x_train = x_train-X_mean
Y_mean = np.mean(y_train, axis=0)
y_train = y_train-Y_mean
X_mean1 = np.mean(x_test, axis=0)
x_test = x_test-X_mean1
Y_mean1 = np.mean(y_test, axis=0)
y_test = y_test-Y_mean1
idn1=np.identity(1)
while(True):
    sigma_square_estimated=(1. / x_train.shape[0]) * \
            np.dot((y_train - np.dot(x_train, w_estimated)).T , (y_train - np.dot(x_train, w_estimated)))
    w_estimated=np.dot(np.dot(pinv(np.dot(2*lmbda,np.dot(sigma_square_estimated,idn1))+np.dot(x_train.T,x_train)),x_train.T),y_train)
    if(abs(sigma_square_estimated-(1. /x_train.shape[0]) * np.dot((y_train - np.dot(x_train, w_estimated)).T , (y_train - np.dot(x_train, w_estimated))))<10**-10):
        break
#print(w_estimated)
print("Mean Square Error for MLE is : "+str(error(x_test,y_test,w_estimated)))    
print("Variance for MLE is : "+str(sigma_square_estimated[0][0]))
y_mle=x_test.dot(w_estimated)+Y_mean1
output("max_likelihood_estimation.csv",testing_data,y_mle)
