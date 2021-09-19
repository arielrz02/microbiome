import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

import sys

from prep_data import *

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, PoissonRegressor

def log_w0(x):
    if x == 0:
        return 0
    else:
        return np.log(x)

def linear_reg(data, tag, groupid, alpha, i=0):
    tag = np.log10(tag)
    data_train, tag_train, data_test, tag_test = gss(data, tag, groupid, train_prec=0.8)
    reg = Ridge(alpha=alpha)
    reg.fit(data_train, tag_train)
    pred = reg.predict(data_test)
    # if i==0:
    #     plt.scatter(x=pred, y=tag_test)
    #     plt.xlabel("prediction")
    #     plt.ylabel("real")
    #     # plt.xlim(2.9, 5)
    #     # plt.ylim(2.9, 5)
    #     plt.show()
    return r2_score(y_true=tag_test, y_pred=pred), spearmanr(a=tag_test, b=pred)[0]


if __name__ == "__main__":
    if not sys.warnoptions:
        import warnings

        warnings.simplefilter("ignore")
    _, _, groupid = create_organized_data("Merged_Shtrauss.csv", "Merged_Res.csv")
    data = pd.read_csv("OTU_merged_Shtrauss.csv", index_col=0)
    for bacnum in range(5, 6):
        data = remove_redun_bac(data, bacnum)
        tag = pd.read_csv("new_shtrauss_tag.csv", index_col=0)
        # plt.hist(x=tag, bins=np.logspace(2, 5, 30))
        # plt.xscale('log')
        # plt.xlabel("number of bacteria")
        # plt.ylabel("amount of tags")
        # plt.show()

        data = take_max(data=data, groupid=groupid)
        corr_arr = np.zeros((1000, 2))
        for alpha in np.linspace(5000, 5000, 1):
            for i in range(1000):
                corr_arr[i, :] = linear_reg(data, tag, groupid, alpha)
            res = corr_arr.mean(axis=0)
            print(res)
            # with open("res.txt", 'a') as file:
            #     file.write(f"{bacnum}, {alpha}: {res[1]}\n")
