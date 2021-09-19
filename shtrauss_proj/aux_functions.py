import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import heapq
import numpy as np
from sklearn.metrics import roc_auc_score


class Dataset:
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.label[i]


# read to dataframes, drop NaN, and do normalization
def process_data(df_X, df_Y):
    # df_x = pd.read_csv(df_X, header=None)  # when there are no headers
    df_x = pd.read_csv(df_X)  # when there are headers
    df_y = pd.read_csv(df_Y)

    df = pd.concat([df_x, df_y], axis=1)
    df.dropna(inplace=True)

    df_y = df['label']
    df_x = df.drop(['label'], axis=1)

    df_x = df_x.values
    df_y = df_y.values

    # scaler = StandardScaler()
    # df_x = scaler.fit_transform(df_x)

    return df_x, df_y


def process_y_only(df_Y):
    df_y = pd.read_csv(df_Y)
    df_y = df_y.values
    return df_y


# load data to pytorch tensors
def loading_data(x_train, y_train, x_validation, y_validation, batch_size):
    train_data = Dataset(x_train.to_numpy(dtype=np.float32), y_train.to_numpy(dtype=np.float32))
    validation_data = Dataset(x_validation.to_numpy(dtype=np.float32), y_validation.to_numpy(dtype=np.float32))

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader


def loading_data_only_test(x_test, y_test, batch_size):
    test_data = testData(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    return test_loader



# for weighted in loss function (in binary classification)
def calculate_pos_neg_in_train(y_train):
    neg = list(y_train).count(0)
    pos = list(y_train).count(1)

    return [1/neg, 1/pos]


# for weighted in loss function (in multi classification)
def calculate_weighted_in_train(y_train):
    class0 = list(y_train).count(0)
    class1 = list(y_train).count(1)
    class2 = list(y_train).count(2)
    class3 = list(y_train).count(3)

    return torch.tensor([1/class0, 1/class1, 1/class2, 1/class3])  # check


# get data distribution
def get_dist(obj):
    count_dict = {'c0': 0, 'c1': 0, 'c2': 0, 'c3': 0}
    for i in obj:
        if i == 0:
            count_dict['c0'] += 1
        elif i == 1:
            count_dict['c1'] += 1
        elif i == 2:
            count_dict['c2'] += 1
        elif i == 3:
            count_dict['c3'] += 1
    return count_dict


# acc calculation (for classification)
def binary_acc(y_pred, y_test):
    y_pred_tag = (y_pred > 0.5).float()
    # y_pred_tag = (y_pred.cpu().numpy() > 0.5).float()

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc


def early_stopping(lst, n):
    last_items = lst[-n:]  # get the last n items in list
    for i in range(len(last_items) - 1):
        if last_items[i] < last_items[i + 1]:  # if there is increase between the item i and item i+1, not stop
            return False
    return True  # if there is not any increase in the last n element, so stop


# def calculate_recall_top_10_percents(y_true, y_pred):  # 1: 860, 0: 29577
#     y_pred, y_true = list(y_pred), list(y_true)
#     N = len(y_pred) // 3  # N: amount of top 10% elements in y_pred and y_true
#     idx_N_largest_pred = heapq.nlargest(N, range(len(y_pred)), y_pred.__getitem__)
#
#     # take the N elements in y_true that in same indexes of the N largest element in y_pred
#     y_true_idx_of_pred_largest = []
#     for i in range(len(y_true)):
#         if i in idx_N_largest_pred:
#             y_true_idx_of_pred_largest.append(y_true[i])
#
#     success = np.array([y_true_idx_of_pred_largest[j] == 1.0 for j in range(len(y_true_idx_of_pred_largest))]).sum()
#     # recall = success / len(y_true_idx_of_pred_largest)
#     recall = success / 860
#
#     return recall


# def auc_in_multiclass(y_true, y_pred):
#     class0, class1, class2, class3 = [], [], [], []
#     for true_pred in y_true:
#         if true_pred == 0:
#             class0.append(1)
#             class1.append(0)
#             class2.append(0)
#             class3.append(0)
#         elif true_pred == 1:
#             class0.append(0)
#             class1.append(1)
#             class2.append(0)
#             class3.append(0)
#         elif true_pred == 2:
#             class0.append(0)
#             class1.append(0)
#             class2.append(1)
#             class3.append(0)
#         elif true_pred == 3:
#             class0.append(0)
#             class1.append(0)
#             class2.append(0)
#             class3.append(1)
#     auc_class0 = roc_auc_score(class0, y_pred)
#     auc_class1 = roc_auc_score(class1, y_pred)
#     auc_class2 = roc_auc_score(class2, y_pred)
#     auc_class3 = roc_auc_score(class3, y_pred)
#
#     return auc_class0, auc_class1, auc_class2, auc_class3










