import time
import glob
import os
import math
import numpy as np
import random
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(128)
random.seed(128)


splits=[0.80,0.10]
q=10
horizon=300
num_classes=19
	
##############################################################################################################
def splitData(data, label, splits):
    training_examples = int(data.shape[0] * splits[0])
    x_train, y_train = data[:training_examples], label[:training_examples]
    x_test, y_test = data[training_examples:], label[training_examples:]
    return x_train, y_train, x_test, y_test
##############################################################################################################
def getData(input_size, y_distance):
    data=pd.read_csv("zprof/201507_data.csv", index_col=None, header=0)
    label=pd.read_csv("zprof/201507_label.csv", index_col=None, header=0)
    
    data = data.iloc[:,1:].values.astype(np.float32)
    label = label.iloc[:,1:].values.astype(np.float32)

    data=data.reshape(data.shape[0], 7)
    label=label.reshape(label.shape[0], 1)
   
    encoding_label=[]
    for i in label:
        if i<700:
            encoding_label.extend([0])
        if i<710 and i >=700:
            encoding_label.extend([1])
        if i<720 and i >=710:
            encoding_label.extend([2])
        if i<730 and i >=720:
            encoding_label.extend([3])
        if i<740 and i >=730:
            encoding_label.extend([4])
        if i<750 and i >=740:
            encoding_label.extend([5])
        if i<760 and i >=750:
            encoding_label.extend([6])
        if i<770 and i >=760:
            encoding_label.extend([7])
        if i<780 and i >=770:
            encoding_label.extend([8])
        if i<790 and i >=780:
            encoding_label.extend([9])
        if i<800 and i >=790:
            encoding_label.extend([10])    
        if i<810 and i >=800:
            encoding_label.extend([11])
        if i<820 and i >=810:
            encoding_label.extend([12])
        if i<830 and i >=820:
            encoding_label.extend([13])
        if i<840 and i >=830:
            encoding_label.extend([14])
        if i<850 and i >=840:
            encoding_label.extend([15])
        if i<860 and i >=850:
            encoding_label.extend([16])
        if i<870 and i >=860:
            encoding_label.extend([17])
        if i<890 and i >=870:
            encoding_label.extend([18])
  
    #print(encoding_label.count(0))
    #print(encoding_label.count(1))
    #print(encoding_label.count(2))
    #print(encoding_label.count(3))
    #print(encoding_label.count(4))
    #print(encoding_label.count(5))
    #print(encoding_label.count(6))
    #print(encoding_label.count(7))
    #print(encoding_label.count(8))
    #print(encoding_label.count(9))
    #print(encoding_label.count(10))
    #print(encoding_label.count(11))
    #print(encoding_label.count(12))
    #print(encoding_label.count(13))
    #print(encoding_label.count(14))
    #print(encoding_label.count(15))
    #print(encoding_label.count(16))
    #print(encoding_label.count(17))
    #print(encoding_label.count(18))
    encoding_label=np.array(encoding_label)
    np.savetxt("result/label_power.csv", encoding_label, delimiter=',', fmt='%s')
    return data, encoding_label
	
##############################################################################################################

data, label=getData(q, horizon)
label=label.reshape(len(label), 1)
print("data shape is:",data.shape,"\t","label shape is", label.shape)

##############################################################################################################    
data=(data-np.mean(data))/np.std(data) # standardization   
label=label[0:data.shape[0],:] 
##############################################################################################################

x_train, y_train, x_test, y_test=splitData(data, label, splits)

##############################################################################################################

np.savetxt("result/test_label.csv", y_test, delimiter=',', fmt='%s')
print("train data shape: ",x_train.shape , y_train.shape)
print("Test data shape: " ,x_test.shape,y_test.shape)
##############################################################################################################
 
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)
np.savetxt("result/test_pred.csv", y_pred, delimiter=',', fmt='%s')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
############################################################################################################## 
#ROC calculate and plotting

label = y_test.reshape(len(y_test),1)
pred = y_pred.reshape(len(y_pred),1)
label=label_binarize(label, classes=[0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
pred=label_binarize(pred, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])

## we need to binarize the label and predicted values before using precision, recall, F1 metrics.
############################################################################################################## 
# precision tp / (tp + fp)
precision = precision_score(y_true = y_test, y_pred = y_pred, average='weighted',zero_division=0)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_true = y_test, y_pred = y_pred,  average='weighted')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_true = y_test, y_pred = y_pred,  average='weighted')
print('F1 score: %f' % f1)

############################################################################################################## 

n_classes=19
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(label[:, i], pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(label.ravel(), pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
print("AOC : ", roc_auc)
############################################################################################################## 

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig("ROC.png")

############################################################################################################## 













