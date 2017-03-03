import urllib.request as rq
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
raw_data = rq.urlopen(data_url)
dataset = np.loadtxt(raw_data, delimiter=',')
n = len(dataset)
train_set = dataset[0:int(np.around(0.3*n)), :]
test_set = dataset[int(np.around(0.3*n)):n, :]

def error(predict, origin):
    np.sum()

#------------------KNN----------------------#

def knn_wrap(train_data, test_data, target, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, target)

kf = KFold(n_splits=10)
