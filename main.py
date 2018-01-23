

import numpy as np
import pandas as pd
import sklearn.ensemble
dataset=pd.read_csv('/home/parker/watermelonData/watermelon3_0a.csv', delimiter=",")
del dataset['num']
X=dataset.values[:,:-1]
y=dataset.values[:,-1]
m,n=np.shape(X)
for i in range(m):
    X[i, n - 1] = round(X[i, n - 1], 3)
    X[i, n - 2] = round(X[i, n - 2], 3)
print(dataset)

def showConfusionMatrix(trueY,myY):
    confusionMatrix = np.zeros((2, 2))
    for i in range(len(trueY)):
        if myY[i] == trueY[i]:
            if trueY[i] == 0:
                confusionMatrix[0, 0] += 1
            else:
                confusionMatrix[1, 1] += 1
        else:
            if trueY[i] == 0:
                confusionMatrix[0, 1] += 1
            else:
                confusionMatrix[1, 0] += 1
    print(confusionMatrix)

myAdaboost=sklearn.ensemble.AdaBoostClassifier(n_estimators=20,learning_rate=0.1,algorithm='SAMME.R')
myAdaboost.fit(X,y)
predictY=myAdaboost.predict(X)
print(y)
print(predictY)
showConfusionMatrix(y,predictY)

myBagging=sklearn.ensemble.BaggingClassifier(n_estimators=50)
myBagging.fit(X,y)
predictY=myBagging.predict(X)
print(y)
print(predictY)
showConfusionMatrix(y,predictY)

myGradient=sklearn.ensemble.GradientBoostingClassifier(n_estimators=1,learning_rate=0.1,max_depth=3)
myGradient.fit(X,y)
predictY=myGradient.predict(X)
print(y)
print(predictY)
showConfusionMatrix(y,predictY)

myForest=sklearn.ensemble.RandomForestClassifier(n_estimators=10)
myForest.fit(X,y)
predictY=myForest.predict(X)
print(y)
print(predictY)
showConfusionMatrix(y,predictY)