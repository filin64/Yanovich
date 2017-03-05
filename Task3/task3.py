import urllib.request as rq
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import pydotplus
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
features = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
            'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
targets = ['1', '2', '3']
fold_num = 10
raw_data = rq.urlopen(data_url)
dataset = np.loadtxt(raw_data, delimiter=',')
np.random.shuffle(dataset)
n, m = np.shape(dataset)
train_target = dataset[0:int(np.around(0.7 * n)), 0]
test_target = dataset[int(np.around(0.7 * n)):n, 0]
train_data = dataset[0:int(np.around(0.7 * n)), 1:m]
test_data = dataset[int(np.around(0.7 * n)):n, 1:m]

def plot_data (x, y, xlbl, ylbl, title):
    plt.title(title)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.plot(x, y)
    plt.show()


def knn_wrap(train_data, train_target, test_data, test_target, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_target)
    prdct = knn.predict(test_data)
    err = mse(test_target, prdct)
    return err

def svm_wrap(train_data, train_target, test_data, test_target, kernel, penalty):
    svc = svm.SVC(C=penalty, kernel=kernel, decision_function_shape='ovo')
    svc.fit(train_data, train_target)
    prdct = svc.predict(test_data)
    err = mse(test_target, prdct)
    return err
def tree_wrap(train_data, train_target, test_data, test_target):
    dtree = tree.DecisionTreeClassifier()
    dtree.fit(train_data, train_target)
    prdct = dtree.predict(test_data)
    err = mse(test_target, prdct)
    # dot_data = tree.export_graphviz(dtree, out_file=None,
    #                                 feature_names=features,
    #                                 class_names=targets,
    #                                 filled=True, rounded=True,
    #                                 special_characters=True)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_pdf("tree.pdf")
    return err
def nn_wrap(train_data, train_target, test_data, test_target, layers, func):
    nn = MLPClassifier(layers, func, 'lbfgs')
    nn.fit(train_data, train_target)
    prdct = nn.predict(test_data)
    err = mse (test_target, prdct)
    return err
def rf_wrap(train_data, train_target, test_data, test_target, tree_num):
    rf = RandomForestClassifier(tree_num)
    

kf = KFold(n_splits=fold_num)
#------------------KNN----------------------#
# knn_mse = []
# k_start, k_finish = (1, 10)
# for k in range(k_start, k_finish):
#     mse_sum = 0
#     for train_index, test_index in kf.split(train_data):
#         err = knn_wrap(train_data[train_index], train_target[train_index], train_data[test_index], train_target[test_index], k)
#         mse_sum += err
#     knn_mse.append(mse_sum/fold_num)
# plot_data(range(k_start, k_finish), knn_mse, 'k', 'MSE', 'KNN CV')

#------------------SVM----------------------#
# a_penalty = np.linspace(0.5, 1.5, 10)
#
# for kernel in ['linear', 'rbf', 'sigmoid', 'poly']:
#     svm_mse = []
#     for penalty in a_penalty:
#         mse_sum = 0
#         for train_index, test_index in kf.split(train_data):
#             err = svm_wrap(train_data[train_index], train_target[train_index], train_data[test_index],
#                            train_target[test_index], kernel, penalty)
#             mse_sum += err
#         svm_mse.append(mse_sum/fold_num)
#     plot_data(a_penalty, svm_mse, 'ERROR PENALTY', 'MSE', kernel)

#------------------TREE----------------------#

# mse_sum = 0
# for train_index, test_index in kf.split(train_data):
#     err = tree_wrap(train_data[train_index], train_target[train_index], train_data[test_index], train_target[test_index])
#     mse_sum += err

#------------------NN----------------------#
n_num = range(100, 150, 10)
for func in ['identity', 'logistic', 'tanh', 'relu']:
    nn_mse =[]
    for k in n_num:
        mse_sum = 0
        for train_index, test_index in kf.split(train_data):
            err = nn_wrap(train_data[train_index], train_target[train_index], train_data[test_index], train_target[test_index], (k), func)
            mse_sum += err
        nn_mse.append(mse_sum/fold_num)
    plot_data(n_num, nn_mse, 'neurons num', 'MSE', 'NN with func ' + func)



