import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
import os
# 使用PyOD工具
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.pca import PCA
from pyod.models.iforest import IForest
from pyod.utils.data import evaluate_print
from pyod.utils.utility import precision_n_scores

from sklearn.metrics import roc_auc_score
from sklearn.utils import column_or_1d
from sklearn.utils import check_consistent_length


# 提取两个列表的共同元素
def extra_element(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    same_set = set1.intersection(set2)
    return list(same_set)


# 提取文件
def load_file(file_path):
    file_list = []
    i = 0
    for _ in os.listdir(file_path):
        file_list.append(file_path + '/' + _)
    return file_list


# 提取各文件公共列
def extra_column(file_list):
    list1 = pd.read_csv(file_list[0], index_col=0).columns
    print("提取所有csv文件中的公共列：")
    for file in tqdm(file_list[1:]):
        list2 = pd.read_csv(file, index_col=0).columns
        list1 = extra_element(list1, list2)
    print("公共列为：", list1)
    return list1


# 划分数据集
def split_data(file_list, columns):
    for file in file_list:
        print(file)
        data = pd.read_csv(file, index_col=0)
        #data = data[data['ground.truth'] == 'nominal']
        #data = data[columns]
        data_len = len(data)
        data.loc[data['ground.truth'] == 'anomaly','ground.truth'] = 1
        data.loc[data['ground.truth'] == 'nominal','ground.truth'] = 0

        x = data[columns]
        y = data['ground.truth']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    
    return x_train, x_test, y_train, y_test


def calculate(method, total_roc, total_prn, x_train, x_test, y_train, y_test):
    if method == 'KNN':
        clf = KNN()
    elif method == 'CBLOF':
        clf = CBLOF()
    elif method == 'PCA':
        clf = PCA()
    else:
        clf = IForest()
    clf.fit(x_train)  # 使用x_train训练检测器clf

    # 返回训练数据x_train上的异常标签和异常分值
    y_train_pred = clf.labels_  # 返回训练数据上的分类标签 (0: 正常值, 1: 异常值)
    y_train_scores = clf.decision_scores_  # 返回训练数据上的异常值 (分值越大越异常)
    print("On train Data:")
    evaluate_print(method, y_train, y_train_scores)

    # 用训练好的clf来预测未知数据中的异常值
    y_test_pred = clf.predict(x_test)  # 返回未知数据上的分类标签 (0: 正常值, 1: 异常值)
    y_test_scores = clf.decision_function(x_test)  # 返回未知数据上的异常值 (分值越大越异常)
    print("On Test Data:")
    evaluate_print(method, y_test, y_test_scores)

    y_true = column_or_1d(y_test)
    y_pred = column_or_1d(y_test_scores)
    check_consistent_length(y_true, y_pred)

    roc = np.round(roc_auc_score(y_true, y_pred), decimals=4),
    prn = np.round(precision_n_scores(y_true, y_pred), decimals=4)
    
    total_roc.append(roc)
    total_prn.append(prn)


# 结果可视化
def visualisation(roc_all, prn_all, names):
    plt.figure(figsize=(10, 10), dpi=80)
    # 柱子总数
    N = 4
    # 包含每个柱子对应值的序列
    values1 = roc_all
    values2 = prn_all
    # 包含每个柱子下标的序列
    index = np.arange(N)
    # 绘制柱状图
    p1 = plt.bar(index, values1, label="ROC")
    p2 = plt.bar(index, values2, label="PRN")
    plt.xlabel('algorithm')
    plt.ylabel('ROC')
    plt.title('')
    plt.xticks(index, ('KNN', 'CBLOF', 'PCA', 'IForest'))
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":

    # 提取公共列 
    file_path = 'homework4\\data\\abalone\\benchmarks'
    file_list = load_file(file_path)
    # columns = extra_column(file_list)
    # 为节约运行时间，直接将上面得到的公共列付给columns变量
    columns = ['V2', 'diff.score', 'original.label', 'V1', 'V4', 'V3', 'V7', 'V5', 'V6']

    # 划分数据集
    x_train, x_test, y_train, y_test = split_data(file_list, columns)

    # 保存实验结果
    knn_roc, knn_prn = [], []
    cblof_roc, cblof_prn = [], []
    pca_roc, pca_prn = [], []
    iforest_roc, iforest_prn = [], []

    # 计算
    calculate('knn', knn_roc, knn_prn, x_train, x_test, y_train, y_test)
    calculate('cblof', cblof_roc, cblof_prn, x_train, x_test, y_train, y_test)
    calculate('pca', pca_roc, pca_prn, x_train, x_test, y_train, y_test)
    calculate('iforest', iforest_roc, iforest_prn, x_train, x_test, y_train, y_test)

    # 输出平均结果
    print('KNN average ROC:', np.average(knn_roc))
    print('KNN average PRN:', np.average(knn_prn))
    print('CBLOF average ROC:', np.average(cblof_roc))
    print('CBLOF average PRN:', np.average(cblof_prn))
    print('PCA average ROC:', np.average(pca_roc))
    print('PCA average PRN:', np.average(pca_prn))
    print('IForest average ROC:', np.average(iforest_roc))
    print('IForest average PRN:', np.average(iforest_prn))

    # 结果可视化
    roc_all = [np.average(knn_roc), np.average(cblof_roc), np.average(pca_roc), np.average(iforest_roc)]
    prn_all = [np.average(knn_prn), np.average(cblof_prn), np.average(pca_prn), np.average(iforest_prn)]
    names = ['KNN', 'CBLOF', 'PCA', 'IForest']
    visualisation(roc_all, prn_all, names)
