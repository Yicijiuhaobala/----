
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn import datasets, model_selection

# 构建SPP层（空间金字塔池化层）
print(datasets.load_iris()['DESCR'])
# Load the Iris dataset and store it in the variable 'iris'
iris = datasets.load_iris()
# Load the Iris dataset and store it in the variable 'X'
X = iris.data
# Load the Iris dataset and store it in the variable 'y'
y = iris.target

x_train,x_test, y_train, y_test = model_selection.train_test_split(X=X, y=y, test_size=0.2, shuffle=0.1)
class SPPLayer(nn.Module):
    def __init__(self, num_layers, pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.num_layers = num_layers
        self.pool_type = pool_type
        
    def forward(self, x):
        batch_size, c, h, w = x.size()  # 样本的batch_size,维度，高，宽
        for i in range(self.num_layers):
            level = i + 1
            kernel_size = (math.ceil(h/level), math.ceil(w/level))
            stride = (math.ceil(h/level), math.ceil(w/level))
            paddding = (math.floor(
                (kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))

            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(
                    x, kernel_size=kernel_size, stride=stride, paddding=paddding).view(batch_size, -1)
            else:
                tensor = F.avg_pool2d(
                    x, kernel_size=kernel_size, stride=stride, paddding=paddding).view(batch_size, -1)
            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(batch_size, -1)
            else:
                x_flatten = torch.cat(
                    (x_flatten, tensor.view(batch_size, -1)), 1)
        return x_flatten


import numpy as np
from sklearn.metrics import roc_auc_score

# python sklearn包计算auc
def get_auc(y_labels, y_scores):
    auc = roc_auc_score(y_labels, y_scores)
    print('AUC calculated by sklearn tool is {}'.format(auc))
    return auc

# 方法1计算auc
def calculate_auc_func1(y_labels, y_scores):
    pos_sample_ids = [i for i in range(len(y_labels)) if y_labels[i] == 1]
    neg_sample_ids = [i for i in range(len(y_labels)) if y_labels[i] == 0]

    sum_indicator_value = 0
    for i in pos_sample_ids:
        for j in neg_sample_ids:
            if y_scores[i] > y_scores[j]:
                sum_indicator_value += 1
            elif y_scores[i] == y_scores[j]:
                sum_indicator_value += 0.5

    auc = sum_indicator_value/(len(pos_sample_ids) * len(neg_sample_ids))
    print('AUC calculated by function1 is {:.2f}'.format(auc))
    return auc

# 方法2计算auc, 当预测分相同时，未按照定义使用排序值的均值，而是直接使用排序值，当数据量大时，对auc影响小
def calculate_auc_func2(y_labels, y_scores):
    samples = list(zip(y_scores, y_labels))
    rank = [(values2, values1) for values1, values2 in sorted(samples, key=lambda x:x[0])]
    pos_rank = [i+1 for i in range(len(rank)) if rank[i][0] == 1]
    pos_cnt = np.sum(y_labels == 1)
    neg_cnt = np.sum(y_labels == 0)
    auc = (np.sum(pos_rank) - pos_cnt*(pos_cnt+1)/2) / (pos_cnt*neg_cnt)
    print('AUC calculated by function2 is {:.2f}'.format(auc))
    return auc


    

if __name__ == '__main__':
    # import torch.nn.functional as F
    # a = torch.randn([3,4,3,5])
    # print(a)
    # b = F.max_pool2d(a, 2 , 1, 0).view(3,-1)
    # print(b)
    a = torch.randn([1, 3, 4, 5])
    b = torch.randn([1, 3, 1, 1])
    print(a)
    print(b)
    print(a*b)
    # spp = SPPLayer(2)
    print(SPPLayer.__mro__)

# https://www.cnblogs.com/marsggbo/p/8572846.html SPP网络详解
# https://zhuanlan.zhihu.com/p/60919662
# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2)
#         x = x.view(x.size(0),  x = F.relu(self.fc1(x)))
#         F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# lenet = LeNet()
    y_labels = np.array([1, 1, 0, 0, 0])
    y_scores = np.array([0.4, 0.8, 0.2, 0.4, 0.5])
    get_auc(y_labels, y_scores)
    calculate_auc_func1(y_labels, y_scores)
    calculate_auc_func2(y_labels, y_scores)