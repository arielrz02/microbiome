import numpy as np
from scipy.stats import spearmanr

import nni
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

#from models import Model_to_nni, Model_to_nni_reg
from aux_functions import process_data, loading_data, calculate_pos_neg_in_train, binary_acc
from prep_data import gss, create_organized_data, col_normalization

class Model_to_nni_reg(nn.Module):
    def __init__(self, input_size, hid1_size, hid2_size, hid3_size, activation_fun, dropout, batch_norm):
        super(Model_to_nni_reg, self).__init__()

        self.layer_1 = nn.Linear(input_size, hid1_size)
        self.layer_2 = nn.Linear(hid1_size, hid2_size)
        self.layer_3 = nn.Linear(hid2_size, hid3_size)
        # self.layer_4 = nn.Linear(hid3_size, hid3_size)
        self.layer_out = nn.Linear(hid3_size, 1)

        self.activation = activation_fun
        self.dropout = nn.Dropout(p=dropout)

        self.isbatchnorm = batch_norm
        self.batchnorm1 = nn.BatchNorm1d(hid1_size)
        self.batchnorm2 = nn.BatchNorm1d(hid2_size)
        self.batchnorm3 = nn.BatchNorm1d(hid3_size)

    def forward(self, inputs):

        x = self.activation(self.layer_1(inputs))
        if self.isbatchnorm == "true":
            x = self.batchnorm1(x)
        x = self.activation(self.layer_2(x))
        if self.isbatchnorm == "true":
            x = self.batchnorm2(x)
        x = self.activation(self.layer_3(x))
        if self.isbatchnorm == "true":
            x = self.batchnorm3(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x


def train(model, train_loader, optimizer, criterion, device):
    loss_train_total, r2_train_total, spear_train_total = 0, 0, 0
    collect_preds = []
    collect_y_batch = []

    model.train()
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(x_batch)

        loss = criterion(y_pred, y_batch.unsqueeze(1))/1000000
        # r2_train = r2_score(y_pred, y_batch.unsqueeze(1))
        r2_train = r2_score(y_pred.detach().cpu().numpy(), y_batch.detach().cpu().numpy())
        spear_train = spearmanr(y_pred.detach().cpu().numpy(), y_batch.detach().cpu().numpy())[0]

        loss.backward()
        optimizer.step()

        loss_train_total += loss.item()
        r2_train_total += r2_train
        spear_train_total += spear_train

        collect_preds.extend(np.array([a.squeeze().tolist() for a in y_pred.detach().cpu().numpy()]))
        collect_y_batch.extend(y_batch.detach().cpu().numpy())

    return loss_train_total / len(train_loader), r2_train_total / len(train_loader), spear_train_total / len(train_loader)


def validation(model, validation_loader, criterion, device):
    loss_val_total, r2_val_total, spear_val_total = 0, 0, 0
    collect_preds = []
    collet_y_batch = []

    model.eval()
    with torch.no_grad():
        for x_batch_val, y_batch_val in validation_loader:
            x_batch_val, y_batch_val = x_batch_val.to(device), y_batch_val.to(device)
            y_pred_val = model(x_batch_val)

            loss_val = criterion(y_pred_val, y_batch_val.unsqueeze(1))/1000000
            r2_val = r2_score(y_pred_val.cpu(), y_batch_val.unsqueeze(1).cpu())
            spear_val = spearmanr(y_pred_val.cpu(), y_batch_val.unsqueeze(1).cpu())[0]

            loss_val_total += loss_val.item()
            r2_val_total += r2_val
            spear_val_total += spear_val

            collect_preds.extend(y_pred_val.squeeze(1).cpu().numpy().tolist())
            collet_y_batch.extend(y_batch_val.cpu().numpy().tolist())



    nni.report_intermediate_result(spear_val_total / len(validation_loader))

    return loss_val_total / len(validation_loader), r2_val_total / len(validation_loader), spear_val_total / len(validation_loader)


def main(X, y, group_id, epoch, batch_size, learning_rate, model, input_size,
         hid1_size, hid2_size, hid3_size, activation_fun, dropout, batch_norm, weight_decay):
    x_train, y_train, x_validation, y_validation = gss(X, y, group_id, train_prec=0.8)
    train_loader, validation_loader = loading_data(x_train, y_train, x_validation, y_validation, batch_size)

    # device = 'cpu'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: " + str(device))

    Model = model(input_size=input_size, hid1_size=hid1_size,hid2_size=hid2_size, hid3_size=hid3_size,
                  activation_fun=activation_fun, dropout=dropout, batch_norm=batch_norm).to(device)
    print(Model)

    criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    optimizer = optim.Adam(Model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # (params, lr=0.001, betas=(0.9, 0.999), eps = 1e-08, weight_decay = 0, amsgrad = False)

    best_spear_val = -1000

    for e in range(1, epoch + 1):
        loss_train, r2_train, spear_train = train(Model, train_loader, optimizer, criterion, device)
        loss_val, r2_val, spear_val = validation(Model, validation_loader, criterion, device)

        print(f'Epoch {e + 0:03}: | Loss Train: {loss_train:.5f} | '
              f'Loss Val: {loss_val:.5f} | R2 Train: {r2_train:.5f} | R2 Val: {r2_val:.5f} |'
              f'spearman Train: {spear_train:.5f} | spearman Val: {spear_val:.5f} | ')

        if spear_val > best_spear_val:
            best_spear_val = spear_val

        if r2_train > -1:
            optimizer.param_groups[0]['lr'] = learning_rate / 10
    return spear_train, spear_val


def running_nni():
    params = nni.get_next_parameter()

    epoch = 40
    batch_size = params["batch_size"]
    learning_rate = params["learning_rate"]
    dropout = params["dropout"]
    hidden_layer1_size = params["hidden_layer1_size"]
    hidden_layer2_size = params["hidden_layer2_size"]
    hidden_layer3_size = params["hidden_layer3_size"]
    batch_norm = params["batch_norm"]
    weight_decay = params["weight_decay"]
    if params["activation_function"] == "relu":
        activation_function = nn.ReLU()
    elif params["activation_function"] == "tanh":
        activation_function = nn.Tanh()
    elif params["activation_function"] == 'elu':
        activation_function = nn.ELU()
    else:
        print("not an activation func")
        return

    data, tag, groupid = create_organized_data("Merged_Shtrauss.csv", "Merged_Res.csv")
    tag = tag["Total Counts"]
    data = col_normalization(data)
    # running_nni()
    cv = 10
    train_lst, val_lst = np.zeros(cv), np.zeros(cv)
    for i in range(cv):
        r2_train, r2_val = main(X=data, y=tag, group_id=groupid, epoch=epoch,batch_size=batch_size,
                                learning_rate=learning_rate,model=Model_to_nni_reg,
                                input_size=len(data.columns), hid1_size=hidden_layer1_size,
                                hid2_size=hidden_layer2_size, hid3_size=hidden_layer3_size,
                                activation_fun=activation_function, dropout=dropout,
                                batch_norm=batch_norm, weight_decay=weight_decay)
        train_lst[i], val_lst[i] = r2_train, r2_val
    nni.report_final_result(val_lst.mean())


if __name__ == '__main__':
    running_nni()
    # data, tag, groupid = create_organized_data("Merged_Shtrauss.csv", "Merged_Res.csv")
    # tag = tag["Total Counts"]
    # data = col_normalization(data)
    # cv = 5
    # train_lst, val_lst = np.zeros(cv), np.zeros(cv)
    # for i in range(cv):
    #     spear_train, spear_val = main(X=data, y=tag, group_id=groupid, epoch=20, batch_size=64, batch_norm="true",
    #                            learning_rate=0.2, model=Model_to_nni_reg, input_size=len(data.columns),
    #                            hid1_size=400, hid2_size=200, activation_fun=nn.ELU(), dropout=0.1, weight_decay=0.1)
    #     train_lst[i], val_lst[i] = spear_train, spear_val
    # print(f"train mean: {train_lst.mean()},\ntrain std: {train_lst.std()},\n"
    #       f"val mean:  {val_lst.mean()},\nval std: {val_lst.std()}")
