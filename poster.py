import sklearn.datasets as sld
import random
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import Imputer 
import sys
import impyute.imputation
import matplotlib.pyplot as plt
import sklearn.preprocessing as slp


sys.setrecursionlimit(100000)

#wine quality dataset chosen from the sklearn library
wine1=sld.load_wine()
wine2=sld.load_wine()
data=wine1.data #full dataset
data_nan=wine2.data 
features=wine1.feature_names

#turn 20% of the alcohol(1st column) feature data to NaN:
r=random.randint(0,177)
data_nan[r,0]=np.NaN
R=[r]
for i in range(35):
    while r in R:
        r=random.randint(0,177)
    R.append(r)
    data_nan[r,0]=np.NaN

#Defining some imputation methods:
def mean_Imputing(X):
    imp_mean=Imputer(missing_values=np.NaN,strategy='mean') #creating a mean based simple imputer
    Y=imp_mean.fit_transform(X)
    return Y
def median_Imputing(X):
    imp_median=Imputer(missing_values=np.NaN,strategy='median') #creating a median based simple imputer
    Y=imp_median.fit_transform(X)
    return Y
def mf_Imputing(X):
    imp_mf=Imputer(missing_values=np.NaN,strategy='most_frequent') #creating an imputer based on the most frequent values
    Y=imp_mf.fit_transform(X)
    return Y
def iter_Imputing(X):
    imp_iter=IterativeImputer(max_iter=10, verbose=0)#creating an iterative imputer
    Y=imp_iter.fit_transform(X)
    return Y
def KNN_Imputing(X):
    Y=impyute.imputation.cs.fast_knn(X, k=5) #imputed data using k nearest neighbours 
    return Y

DS_mean=mean_Imputing(data_nan)
DS_median=median_Imputing(data_nan)
DS_mf=mf_Imputing(data_nan)
DS_iter=iter_Imputing(data_nan)
DS_knn=KNN_Imputing(data_nan)

#extracting the values for alcohol pct(1st column):
def alcohol_pct(X):
    Y=np.array(X[:,0])
    return(Y)

AP_mean=alcohol_pct(DS_mean)
AP_median=alcohol_pct(DS_median)
AP_mf=alcohol_pct(DS_mf)
AP_iter=alcohol_pct(DS_iter)
AP_knn=alcohol_pct(DS_knn)
AP=alcohol_pct(data)

#labeling every sample from 1 to 178:
L=[]
for i in range(1,179):
    L.append('No:'+str(i))

#checking the bias of the imputation methods (using std deviation as a parameter):
def bias(X,Y):
    b=0
    for i in range(len(X)):
        x=abs(X[i]-Y[i])**2
        b=b+x
    return b/len(X)

mean_imp_bias=bias(AP_mean,AP)
median_imp_bias=bias(AP_median,AP)
mf_imp_bias=bias(AP_mf,AP)
iter_imp_bias=bias(AP_iter,AP)
knn_imp_bias=bias(AP_knn,AP)
B=[mean_imp_bias,median_imp_bias,mf_imp_bias,knn_imp_bias,iter_imp_bias]

#writing relative scores for ploting
Bias=slp.normalize(np.array(B).reshape(1,-1))
Scores=np.ones((1,5))-Bias

#ploting the bar charts for all imputations and ploting a bar chart with the efficiency of the method:
wine_labels=L
method_labels=['Mean','Median','Most Frequent','KNN','Iterative MultiV']

plt.style.use('ggplot')
x_pos1 = [i for i, _ in enumerate(wine_labels)]
x_pos2 = [i for i, _ in enumerate(method_labels)]

plt.bar(x_pos1, AP_mean, color='b')
plt.xlabel("Wine number")
plt.ylabel("Alcohol pct")
plt.title("Mean Imputer")
axes=plt.gca()
axes.set_ylim([10,15])
plt.xticks(x_pos1, wine_labels)
plt.show()
plt.bar(x_pos1, AP_median, color='b')
plt.xlabel("Wine number")
plt.ylabel("Alcohol pct")
plt.title("Median Imputer")
axes=plt.gca()
axes.set_ylim([10,15])
plt.xticks(x_pos1, wine_labels)
plt.show()
plt.bar(x_pos1, AP_mf, color='b')
plt.xlabel("Wine number")
plt.ylabel("Alcohol pct")
plt.title("Most Frequent Value Imputer")
axes=plt.gca()
axes.set_ylim([10,15])
plt.xticks(x_pos1, wine_labels)
plt.show()
plt.bar(x_pos1, AP_knn, color='b')
plt.xlabel("Wine number")
plt.ylabel("Alcohol pct")
plt.title("KNN Imputer")
axes=plt.gca()
axes.set_ylim([10,15])
plt.xticks(x_pos1, wine_labels)
plt.show()
plt.bar(x_pos1, AP_iter, color='b')
plt.xlabel("Wine number")
plt.ylabel("Alcohol pct")
plt.title("Iterative Multivariable Imputer")
axes=plt.gca()
axes.set_ylim([10,15])
plt.xticks(x_pos1, wine_labels)
plt.show()
plt.bar(x_pos2,Scores[0], color='b')
plt.ylabel("Method Score")
plt.title("Score(0 to 1 based on relative efficiency)")
axes=plt.gca()
axes.set_ylim([0,1])
plt.xticks(x_pos2, method_labels)
plt.show()






        
    

    
    




    




