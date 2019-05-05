from scipy.io import loadmat
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')

epochs = 5
learningRate = 0.001
batchSize = 10
regStrength = 0.5
momentum = 0
velocity = 0

regStrength=0.5

def SGDWithMomentum(x, y,wt):
    velocity = np.zeros(wt.shape)
    print(wt.shape)
    print(velocity.shape)
    losses = []
    for i in range(0, x.shape[0], batchSize):
        Xbatch = x[i:i+batchSize]
        ybatch = y[i:i+batchSize]
        loss, dw = computeLoss(Xbatch, ybatch,wt)
        velocity = (momentum * velocity) + (learningRate * dw)
        wt = wt-velocity
        losses.append(loss)
    return np.sum(losses) / len(losses)

def softmaxEquation(scores):
    scores -= np.max(scores)
    prob = (np.exp(scores).T / np.sum(np.exp(scores), axis=1)).T
    return prob

def computeLoss(x, yMatrix,wt):
    numOfSamples = x.shape[0]
    scores = np.dot(x,wt)
    prob = softmaxEquation(scores)

    loss = -np.log(np.max(prob)) * yMatrix
    regLoss = (1/2)*regStrength*np.sum(wt*wt)
    totalLoss = (np.sum(loss) / numOfSamples) + regLoss
    grad = ((-1 / numOfSamples) * np.dot(x.T, (yMatrix - prob))) + (regStrength * wt)
    return totalLoss, grad

def oneHotEncoding(y, numOfClasses):
    #y = np.asarray(y, dtype='int32')
  # if len(y) > 1:
  #       y = y.reshape(-1)
  #   if not numOfClasses:
  #       numOfClasses = np.max(y) + 1
  #   yMatrix = np.zeros((len(y), numOfClasses))
  #   yMatrix[np.arange(len(y)), y] = 1
  one_hot_gt=np.zeros((21025,17))
  i=0
  labelcount=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  for x in np.nditer(y.copy(order='C')):
  	one_hot_gt[i][x]=1
  	i=i+1
  	labelcount[x]+=1
  labelcount=np.array(labelcount)
  labelcount.astype(int)
  return labelcount


def predict(x):
    return np.argmax(x.dot(wt), 1)
   
def randomize_data(data,data2):
    (m,n,o)=data.shape
    (p,q)=data2.shape
    random_indices_for_rows=np.random.permutation(m)
    random_indices_for_columns=np.random.permutation(n)
    fifty_percent=int(0.5*m)+1
    train_row_indeces=random_indices_for_rows[0:fifty_percent]
    train_column_indeces=random_indices_for_columns[0:fifty_percent]
    test_row_indeces=random_indices_for_rows[fifty_percent:]
    test_column_indeces=random_indices_for_columns[fifty_percent:]
    X_train=np.matrix(data[train_row_indeces[0],train_column_indeces[0],:])
    X_test=np.matrix(data[test_row_indeces[0],test_column_indeces[0],:])
    Y_train=[data2[train_row_indeces[0],train_column_indeces[0]]]
    Y_test=[data2[test_row_indeces[0],test_column_indeces[0]]]
    for i in range(len(train_row_indeces)):
        for j in range(len(train_column_indeces)):
            if data2[train_row_indeces[i],train_column_indeces[j]]!=0:
                X_train=np.vstack([X_train,np.matrix(data[train_row_indeces[i],train_column_indeces[j],:])])
                Y_train.append(data2[train_row_indeces[i],train_column_indeces[j]])
    
    #X_train=np.hstack((X_train,np.ones((X_train.shape[0],1))))
    #X_train=np.hstack((np.ones(X_train.shape[0],1),X_train))        
    
    for a in range(len(test_row_indeces)):
        for b in range(len(test_column_indeces)):
            if data2[test_row_indeces[a],test_column_indeces[b]]!=0:
                X_test=np.vstack([X_test,np.matrix(data[test_row_indeces[a],test_column_indeces[b],:])])
                Y_test.append(data2[test_row_indeces[a],test_column_indeces[b]])
            
    #X_test=np.hstack((X_test,np.ones((X_test.shape[0],1))))
    #X_test=np.hstack((np.ones(X_test.shape[0],1),X_test))        
            
    return (X_train,np.matrix(Y_train).T,X_test,np.matrix(Y_test).T)           

data3=loadmat('Indian_pines_corrected.mat')
data3
data3=data3['indian_pines_corrected']
data=data3.copy().astype(float)
for j in range(data.shape[2]):
    maxv=np.max(data[:,:,j])
    minv=np.min(data[:,:,j])
    data[:,:,j]=(data[:,:,j]-minv)/(maxv-minv)
data2=loadmat('Indian_pines_gt.mat')    
print(data2)
data2=data2['indian_pines_gt']
print(data.shape)
print(data2.shape)
(X_train,Y_train,X_test,Y_test)=randomize_data(data,data2)
print(X_train.shape)
print(Y_train.shape)


D = X_train.shape[1]  # dimensionality
label = np.unique(Y_train)
numOfClasses = len(label) # number of classes
Y_trainEnc = oneHotEncoding(Y_train, numOfClasses)
Y_testEnc = oneHotEncoding(Y_test, numOfClasses)
wt = 0.001 * np.random.rand(D, numOfClasses)
velocity = np.zeros(wt.shape)
trainLosses = []
testLosses = []
trainAcc = []
testAcc = []
for e in range(epochs): # loop over epochs
    trainLoss = SGDWithMomentum(X_train, Y_trainEnc,wt)
    testLoss, dw = computeLoss(X_test, Y_testEnc,wt)
    trainAcc.append(meanAccuracy(X_train, Y_train))
    testAcc.append(meanAccuracy(X_test, Y_test))
    trainLosses.append(trainLoss)
    testLosses.append(testLoss)
    print("{:d}\t->\tTrainL : {:.7f}\t|\tTestL : {:.7f}\t|\tTrainAcc : {:.7f}\t|\tTestAcc: {:.7f}"
          .format(e, trainLoss, testLoss, trainAcc[-1], testAcc[-1]))
