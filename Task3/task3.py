import urllib.request as rq
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib.legend_handler import  HandlerLine2D
import pydotplus
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
features = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
            'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
targets = ['1', '2', '3']
fold_num = 10
raw_data = rq.urlopen(data_url)
#raw_data = open('wine.data.txt')
dataset = np.loadtxt(raw_data, delimiter=',')
np.random.seed(2)
np.random.shuffle(dataset)
n, m = np.shape(dataset)
train_target = dataset[0:int(np.around(0.7 * n)), 0]
test_target = dataset[int(np.around(0.7 * n)):n, 0]
train_data = dataset[0:int(np.around(0.7 * n)), 1:m]
test_data = dataset[int(np.around(0.7 * n)):n, 1:m]

def sim(origin, predict):
    return sum(origin != predict)/float(len(origin))

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
    err = sim(test_target, prdct)
    return err

def svm_wrap(train_data, train_target, test_data, test_target, kernel, penalty):
    svc = svm.SVC(C=penalty, kernel=kernel, decision_function_shape='ovo')
    svc.fit(train_data, train_target)
    prdct = svc.predict(test_data)
    err = sim(test_target, prdct)
    return err, svc.n_support_
def tree_wrap(train_data, train_target, test_data, test_target, k, crt):
    dtree = tree.DecisionTreeClassifier(criterion=crt, min_samples_split=k)
    dtree.fit(train_data, train_target)
    prdct = dtree.predict(test_data)
    err = sim(test_target, prdct)
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
    err = sim (test_target, prdct)
    return err
def rf_wrap(train_data, train_target, test_data, test_target, tree_num, crt):
    rf = RandomForestClassifier(tree_num, criterion=crt)
    rf.fit(train_data, train_target)
    prdct = rf.predict(test_data)
    err = sim(test_target, prdct)
    return err


kf = KFold(n_splits=fold_num)
#------------------KNN----------------------#
print ("KNN statistics")
knn_err_tst = []
knn_err_tr = []
k_start, k_finish = (3, 25)
best_knn_param = 0
for k in range(k_start, k_finish):
    knn_err_sum_cv_tst = 0
    knn_err_sum_cv_tr = 0
    for train_index, test_index in kf.split(train_data):
        err = knn_wrap(train_data[train_index], train_target[train_index], train_data[test_index], train_target[test_index], k)
        knn_err_sum_cv_tst += err
        err = knn_wrap(train_data[train_index], train_target[train_index], train_data[train_index], train_target[train_index], k)
        knn_err_sum_cv_tr += err
    knn_err_tst.append((k, knn_err_sum_cv_tst / fold_num))
    knn_err_tr.append((k, knn_err_sum_cv_tr / fold_num))
ans = min(knn_err_tst, key=lambda x: x[1])
print ("Best k for KNN =", ans[0], "Error =", ans[1])
best_knn_param = ans
plt.title("KNN error depends on k")
plt.xlabel("k")
plt.ylabel("Error")
line1, = plt.plot(range(k_start, k_finish), np.array(knn_err_tst)[:,1], 'r', label='Test Set')
line2, = plt.plot(range(k_start, k_finish), np.array(knn_err_tr)[:,1], 'b', label='Train Set')
plt.legend(handles=[line1, line2])
plt.show()


#------------------SVM----------------------#
print ('SVM statistics')
a_penalty = np.linspace(0.1, 1.5, 20)
sv_num_vals = 0
sv_num_data = []
svm_err_cv_tst = []
for kernel in ['linear', 'rbf', 'sigmoid', 'poly']:
    for penalty in a_penalty:
        sv_num_vals = 0
        svm_err_sum_cv_tst = 0
        for train_index, test_index in kf.split(train_data):
            err, sv_num = svm_wrap(train_data[train_index], train_target[train_index], train_data[test_index],
                           train_target[test_index], kernel, penalty)
            svm_err_sum_cv_tst += err
            if kernel == 'linear':
                sv_num_vals += np.array(sv_num)
        if kernel == 'linear':
            sv_num_data.append(sv_num_vals / fold_num)
        svm_err_cv_tst.append((kernel, penalty, svm_err_sum_cv_tst / fold_num))
ans = min(svm_err_cv_tst, key=lambda x: x[2])
print ("Train set:", "Kernel:", ans[0], "Penalty =", ans[1], "Error =", ans[2])
best_svm_param = ans
plt.title("SV number depends on C-parameter")
plt.xlabel("C")
plt.ylabel("SV number")
line1, = plt.plot(a_penalty, np.array(sv_num_data)[:,0], 'r', label='First class')
line2, = plt.plot(a_penalty, np.array(sv_num_data)[:,1], 'b', label='Second class')
line3, = plt.plot(a_penalty, np.array(sv_num_data)[:,2], 'g', label='Third class')
plt.legend(handles=[line1, line2, line3])
plt.show()

#------------------TREE----------------------#
print ("Tree statistics")
tree_err_cv_tst = []
for crt in ['gini', 'entropy']:
    for k in range (5, 50):
        tree_err_sum_cv_tst = 0
        for train_index, test_index in kf.split(train_data):
            err = tree_wrap(train_data[train_index], train_target[train_index], train_data[test_index],
                            train_target[test_index], k, crt)
            tree_err_sum_cv_tst += err
        tree_err_cv_tst.append((k, crt, tree_err_sum_cv_tst / fold_num))
ans = min(tree_err_cv_tst, key=lambda x: x[1])
print ("Test Set: Min split =", ans[0], "Criterion:", ans[1], "Error =", ans[2])
best_tree_param = ans

#------------------NN----------------------#
print ("Neural Network statistics")
n_num1 = range(10, 21, 10)
n_num2 = range(0, 21, 10)
n_num3 = range(0, 41, 10)
mn = 1e9
ans = 0
nn_err_cv_tst = []
for func in ['identity', 'logistic', 'tanh', 'relu']:
    for i in n_num3:
        for j in n_num2:
            for k in n_num1:
                layers = [k]
                if j != 0:
                    layers.append(j)
                if i != 0:
                    layers.append(i)
                nn_err_sum_cv_tst = 0
                for train_index, test_index in kf.split(train_data):
                    err = nn_wrap(train_data[train_index], train_target[train_index], train_data[test_index], train_target[test_index],
                                  tuple(layers), func)
                    nn_err_sum_cv_tst += err
                nn_err_cv_tst.append((func, layers, nn_err_sum_cv_tst / fold_num))
ans = min(nn_err_cv_tst, key=lambda x: x[2])
print ("Train Set: Function:", ans[0], "Layers:", ans[1], "Error =", ans[2])
best_nn_param = ans

#------------------RandomForest----------------------#
print ("Random forest statistics")
rf_err_cv_tst = []
for crt in ['gini', 'entropy']:
    for k in range (10, 50):
        rf_err_sum_cv_tst = 0
        for train_index, test_index in kf.split(train_data):
            err = rf_wrap(train_data[train_index], train_target[train_index], train_data[test_index],
                          train_target[test_index], k, crt)
            rf_err_sum_cv_tst += err
            # err = rf_wrap(train_data[train_index], train_target[train_index], train_data[train_index], train_target[train_index], k)
        rf_err_cv_tst.append((k, crt, rf_err_sum_cv_tst / fold_num))
ans = min(rf_err_cv_tst, key=lambda x: x[1])
print ("Number of trees =", ans[0], "Criterion:", ans[1], "Error =", ans[2])
best_rf_param = ans

err_sum = 0
#----------------------------TEST/TRAIN ERRORS-----------------------#
print ("ERRORS")
for train_index, test_index in kf.split(train_data):
    err = tree_wrap(train_data[train_index], train_target[train_index], train_data[train_index], train_target[train_index],
                    best_tree_param[0], best_tree_param[1])
    err_sum += err
print ('Decision Tree')
print ('Train Set Error', err_sum / fold_num)
print ('Test Set Error', best_tree_param[2])
err_sum = 0
for train_index, test_index in kf.split(train_data):
    err = nn_wrap(train_data[train_index], train_target[train_index], train_data[train_index], train_target[train_index],
              best_nn_param[1], best_nn_param[0])
    err_sum += err
print ('Neural Network')
print ('Train Set Error', err_sum / fold_num)
print('Test Set Error', best_nn_param[2])
err_sum = 0
for train_index, test_index in kf.split(train_data):
    err = svm_wrap(train_data[train_index], train_target[train_index], train_data[train_index],
                           train_target[train_index], best_svm_param[0], best_svm_param[1])[0]
    err_sum += err
print ('SVM')
print ('Train set error:', err_sum / fold_num)
print ('Test set error', best_svm_param[2])

#----------------------------FULL-----------------------------------#
print ("Full Data Errors:")
err = knn_wrap(train_data, train_target, test_data, test_target, best_knn_param[0])
print ("Knn:", err)
err = tree_wrap(train_data, train_target, test_data, test_target, best_tree_param[0], best_tree_param[1])
print ("Tree:", err)
err = svm_wrap(train_data, train_target, test_data, test_target, best_svm_param[0], best_svm_param[1])[0]
print ("SVM", err)
err = nn_wrap(train_data, train_target, test_data, test_target, best_nn_param[1], best_nn_param[0])
print ("Neural Network:", err)
err = rf_wrap(train_data, train_target, test_data, test_target, best_rf_param[0], best_rf_param[1])
print ("Random Forest:", err)

