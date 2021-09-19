import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

def create_organized_data(data_file, tag_file):
    data = pd.read_csv(data_file, index_col=0).dropna(axis=1)
    tagdf = pd.read_csv(tag_file).T
    tagdf.columns = data.columns
    tagdf.drop("Unnamed: 6", inplace=True)
    full_data = data.append(tagdf).T
    full_data = full_data[full_data["Flag"] != -1]
    full_data.drop("Flag", axis=1, inplace=True)
    data = full_data.iloc[:, :-5]
    tags = full_data.iloc[:, -5:]
    groupid = pd.Series([date.split(".")[0] for date in tags.index])
    return data, tags, groupid

def gss(data, tags, group_id, train_prec):
    gss = GroupShuffleSplit(n_splits=1, train_size=train_prec)
    train_idx, val_idx = next(gss.split(data, groups=group_id))
    train_data, train_tag = data.iloc[train_idx, :], tags.iloc[train_idx]
    val_data, val_tag = data.iloc[val_idx, :], tags.iloc[val_idx]
    return train_data, train_tag,\
           val_data, val_tag

def col_normalization(df):
    sum = df.sum(axis=0).replace(0, 1)
    df = df.divide(sum, axis=1).fillna(0)
    return df

def take_max(data, groupid):
    groupid.index = data.index
    new_data = pd.concat([data, groupid], axis=1)
    max_data = new_data.groupby(0).max()
    for ind, date in zip(new_data.index, new_data[0]):
        data.loc[ind] = max_data.loc[date]
    return data

def remove_redun_bac(data, cutoff=5):
    data = data.T
    data = data[data.replace(-1, pd.NA).count(axis=1) > cutoff]
    return data.T.replace(-1,0)


if __name__ == "__main__":
    remove_redun_bac(pd.read_csv("OTU_merged_Shtrauss.csv", index_col=0))