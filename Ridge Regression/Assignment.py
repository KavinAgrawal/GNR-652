import numpy as np
import pandas as pd
import sys
import cvxopt as cvxopt
import cvxopt.solvers

def dual(X,Y):
    n_samples, n_features = X.shape
    P = cvxopt.matrix((np.outer(Y,Y) * np.dot(X,X.T)))
    q = cvxopt.matrix(np.ones(n_samples)*-1.,(n_samples,1))
    G = cvxopt.matrix(np.diag(np.ones(n_samples)* -1.)) 
    h = cvxopt.matrix(np.zeros(n_samples))
    A = cvxopt.matrix(Y,(1,n_samples),'d')
    b = cvxopt.matrix(0.0)
    return P,q,G,h,A,b    

def svm(X,Y):
	P,q,G,h,A,b = dual(X,Y)
	alpha = np.array(cvxopt.solvers.qp(P,q,G,h,A,b)['x']).flatten()
	
	n_samples, n_features = X.shape
	sv = alpha > 1e-6

	a = alpha[sv]

	support_v = X[sv]

	support_vy = Y[sv]

	w_dual = np.zeros(n_features)
	for n in range(len(a)):
	    w_dual += a[n] * support_vy[n] * support_v[n]

	class1=np.where(Y==1)
	class0=np.where(Y==-1)
	t=np.dot(w_dual,X.T)
	t1=np.take(t,indices=class1)
	t_1=np.min(t1)
	t2=np.take(t,indices=class0)
	t_2=np.max(t2)
	b_dual=-(1/2)*(t_1+t_2)
	return w_dual,b_dual

def predict(X,w,b):
    return np.sign(np.dot(X, w)+b)

def error(Y,y):
    Y=Y.flatten()
    i=0
    k=0
    for j in range(len(y)):
        if(Y[j]!=y[j]):
            i=i+1
        else:
            k=k+1
    return i*100/len(y)  ,k*100/len(y) 

def svm_data():
    df=pd.read_csv('creditcard.csv').sample(frac=1)
    df0=df[df.Class==0].sample(n=100)
    df0['Class']=df0['Class'].apply(lambda x: x - 1)
    df1=df[df.Class==1].sample(n=100)
    df=pd.concat([df0,df1]).sample(frac=1)
    training_data = df.iloc[:int(0.8*len(df))].reset_index(drop=True)
    testing_data = df.iloc[int(0.8*len(df)):].reset_index(drop=True)
    x_tr=training_data.drop(columns=['Class','Time'],axis=1).values
    x_te=testing_data.drop(columns=['Class','Time'],axis=1).values
    X_train = np.asarray(x_tr)
    X_test = np.asarray(x_te)
    Y_train=training_data[['Class']].values
    Y_test=testing_data[['Class']].values
    
    w,b=svm(X_train,Y_train)

    Y_test_pred=predict(X_test,w,b)

    print(Y_test_pred)
    print(Y_test)
    err,accuracy=error(Y_test,Y_test_pred)
    print("Error is : " + str(err))
    print("Accuracy is : " + str(accuracy))

svm_data()
