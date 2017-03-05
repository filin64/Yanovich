import urllib.request as rq
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn import svm
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
raw_data = rq.urlopen(data_url)
dataset = np.loadtxt(raw_data, delimiter=',')
np.random.shuffle(dataset)
n, m = np.shape(dataset)
train_target = dataset[0:int(np.around(0.7 * n)), 0]
test_target = dataset[int(np.around(0.7 * n)):n, 0]
train_data = dataset[0:int(np.around(0.7 * n)), 1:m]
test_data = dataset[int(np.around(0.7 * n)):n, 1:m]



def knn_wrap(train_data, train_target, test_data, test_target, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_target)
    prdct = knn.predict(test_data)
    err = mse(test_target, prdct)
    return err

def svm_wrap(train_data, train_target, test_data, test_target, kernel, penalty):
    svc = svm.SVC(C=penalty, kernel=kernel)
    svc.fit(train_data, train_target)
    prdct = svc.predict(test_data)
    err = mse(test_target, prdct)
    return err

kf = KFold(n_splits=10)
#------------------KNN----------------------#
# knn_mse = []
# k_start, k_finish = (1, 10)
# for k in range(k_start, k_finish):
#     mse_sum = 0
#     for train_index, test_index in kf.split(train_data):
#         err = knn_wrap(train_data[train_index], train_target[train_index], train_data[test_index], train_target[test_index], k)
#         mse_sum += err
#     knn_mse.append(mse_sum/len(range(k_start, k_finish)))
# plt.title('KNN CV')
# plt.xlabel('k')
# plt.ylabel('MSE')
# plt.plot(range(k_start, k_finish), knn_mse)
# plt.show()

#------------------SVM----------------------#
a_penalty = np.linspace(0.5, 1.5, 10)

for kernel in ['linear', 'rbf', 'sigmoid']:
    plt.title(kernel)
    svm_mse = []
    for penalty in a_penalty:
        mse_sum = 0
        for train_index, test_index in kf.split(train_data):
            err = svm_wrap(train_data[train_index], train_target[train_index], train_data[test_index],
                           train_target[test_index], kernel, penalty)
            mse_sum += err
        svm_mse.append(mse_sum/len(a_penalty))
    plt.xlabel('ERROR PENALTY')
    plt.ylabel('MSE')
    plt.plot(a_penalty, svm_mse)
    plt.show()