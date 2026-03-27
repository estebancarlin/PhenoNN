#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hyperparameter tuning for lstm models

@author: gliu
"""

import csv
import os
import pandas as pd
import numpy as np
from . import dataloader_phenodata as dl
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
import pickle
from tqdm import tqdm
import argparse
from .lstm import LSTM

# Set CUDA device if specified
if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    train_loader,
    valid_loader,
    model,
    loss_function,
    optimizer,
    scheduler,
    epochs,
    patience,
    nr_features,
    target="gcc",
):
    patience_counter = 0
    best_loss = 10e10
    train_loss_epochs = []
    valid_loss_epochs = []

    tepoch = tqdm(range(epochs))
    for epoch in tepoch:
        model.train()
        train_total_losses = 0

        for batch_number, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()  # caluclate the gradient, manually setting to 0
            if nr_features == 6:
                X_train = batch["features"][:, :, :6]
            elif nr_features == 8:
                X_train = torch.cat(
                    (batch["features"][:, :, :6], batch["features"][:, :, 8:10]),
                    axis=-1,
                )
            elif nr_features == 9:
                X_train = torch.cat(
                    (batch["features"][:, :, :7], batch["features"][:, :, 8:10]),
                    axis=-1,
                )
            else:
                X_train = torch.cat(
                    (batch["features"][:, :, :7], batch["features"][:, :, 8:]), axis=-1
                )

            if target == "gcc":
                y_train = batch["target"][:, :, 0]
            elif target == "rcc":
                y_train = batch["target"][:, :, 1]
            elif target == "gcc_lowess":
                y_train = batch["target"][:, :, 2]
            else:
                y_train = batch["target"][:, :, 3]

            outputs = model(X_train)
            outputs = outputs[:, 365 : 365 + y_train.shape[-1]]
            loss = loss_function(outputs, y_train)

            loss.backward()
            optimizer.step()

            train_total_losses += loss.item()

        train_loss = train_total_losses / len(train_loader)
        valid_loss = valid_model(
            valid_loader, model, loss_function, nr_features, target
        )
        train_loss_epochs.append(train_loss)
        valid_loss_epochs.append(valid_loss)

        scheduler.step()

        # Early stopping
        tepoch.set_postfix(train_loss=train_loss, validation_loss=valid_loss)
        if valid_loss > best_loss:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!\nStart to test process.")
                print(epoch)
                return model, train_loss_epochs, valid_loss_epochs, epoch
        else:
            patience_counter = 0
            best_loss = valid_loss

    return model, train_loss_epochs, valid_loss_epochs, epoch


def valid_model(valid_loader, model, loss_function, nr_features, target):
    valid_total_losses = 0
    model.eval()

    with torch.no_grad():
        for batch_number, batch in enumerate(valid_loader, 1):
            if nr_features == 6:
                X_valid = batch["features"][:, :, :6]
            elif nr_features == 8:
                X_valid = torch.cat(
                    (batch["features"][:, :, :6], batch["features"][:, :, 8:10]),
                    axis=-1,
                )
            elif nr_features == 9:
                X_valid = torch.cat(
                    (batch["features"][:, :, :7], batch["features"][:, :, 8:10]),
                    axis=-1,
                )
            else:
                X_valid = torch.cat(
                    (batch["features"][:, :, :7], batch["features"][:, :, 8:]), axis=-1
                )

            if target == "gcc":
                y_valid = batch["target"][:, :, 0]
            elif target == "rcc":
                y_valid = batch["target"][:, :, 1]
            elif target == "gcc_lowess":
                y_valid = batch["target"][:, :, 2]
            else:
                y_valid = batch["target"][:, :, 3]

            outputs = model(X_valid)
            outputs = outputs[:, 365 : 365 + y_valid.shape[-1]]
            loss = loss_function(outputs, y_valid)

            # valid loss
            valid_total_losses += loss.item()
        valid_loss = valid_total_losses / len(valid_loader)
    return valid_loss


def test_model(test_loader, model, loss_function, nr_features, target):
    test_total_losses = 0
    model.eval()

    with torch.no_grad():
        for batch_number, batch in enumerate(test_loader, 1):
            if nr_features == 6:
                X_test = batch["features"][:, :, :6]
            elif nr_features == 8:
                X_test = torch.cat(
                    (batch["features"][:, :, :6], batch["features"][:, :, 8:10]),
                    axis=-1,
                )
            elif nr_features == 9:
                X_test = torch.cat(
                    (batch["features"][:, :, :7], batch["features"][:, :, 8:10]),
                    axis=-1,
                )
            else:
                X_test = torch.cat(
                    (batch["features"][:, :, :7], batch["features"][:, :, 8:]), axis=-1
                )

            if target == "gcc":
                y_test = batch["target"][:, :, 0]
            elif target == "rcc":
                y_test = batch["target"][:, :, 1]
            elif target == "gcc_lowess":
                y_test = batch["target"][:, :, 2]
            else:
                y_test = batch["target"][:, :, 3]

            outputs = model(X_test)
            outputs = outputs[:, 365 : 365 + y_test.shape[-1]]
            loss = loss_function(outputs, y_test)

            # test loss
            test_total_losses += loss.item()
        test_loss = test_total_losses / len(test_loader)
    return test_loss


def model_predict(data_loader, model, nr_features, target, denormalize=None):
    pred = torch.tensor([])
    obs = torch.tensor([])
    site = []
    year = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            if nr_features == 6:
                X = batch["features"][:, :, :6]
            elif nr_features == 8:
                X = torch.cat(
                    (batch["features"][:, :, :6], batch["features"][:, :, 8:10]),
                    axis=-1,
                )
            elif nr_features == 9:
                X = torch.cat(
                    (batch["features"][:, :, :7], batch["features"][:, :, 8:10]),
                    axis=-1,
                )
            else:
                X = torch.cat(
                    (batch["features"][:, :, :7], batch["features"][:, :, 8:]), axis=-1
                )

            if target == "gcc":
                y = batch["target"][:, :, 0]
            elif target == "rcc":
                y = batch["target"][:, :, 1]
            elif target == "gcc_lowess":
                y = batch["target"][:, :, 2]
            else:
                y = batch["target"][:, :, 3]

            outputs = model(X)
            y_star = outputs[:, 365:]

            if denormalize is not None:
                X, outputs = denormalize(X, outputs)
                X, y = denormalize(X, y)

            pred = torch.cat((pred, y_star), 0)
            obs = torch.cat((obs, y), 0)

            site.extend(batch["sitename"])
            year = torch.cat((year, batch["year"]), 0)

    return pred, obs, site, year


def run_lstm_hp_tuning(
    batch_size=8,
    learning_rate=0.01,
    hidden_size=64,
    dropout_rate=0,
    weight_decay=0,
    patience=30,
    m=0,
    pft="DB",
    nr_features=8,
    target="gcc",
    plot_learning_curve=True,
):
    num_layers = 1
    epochs = 150
    target_size = 1
    input_size = nr_features
    # k_folds = 10
    train_dataset0 = dl.PhenoDataset(
        "../data/datasetm{}/".format(m), pft, preprocessing="standardize"
    )

    with open("../data/train_dataset_fixed_4hptuning_DB.pickle", "rb") as data:
        train_dataset = pickle.load(data)

    with open("../data/valid_dataset_fixed_4hptuning_DB.pickle", "rb") as data:
        valid_dataset = pickle.load(data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    # training the LSTM model
    lstm = LSTM(target_size, input_size, hidden_size, num_layers, dropout_rate)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        lstm.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    lstm, train_losses, valid_losses, epoch_estop = train_model(
        train_loader,
        valid_loader,
        lstm,
        loss_function,
        optimizer,
        scheduler,
        epochs,
        patience=patience,
        nr_features=nr_features,
        target=target,
    )

    if plot_learning_curve:
        # plot learning curve
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(train_losses, label="train loss", color="b", lw=2)
        plt.plot(valid_losses, label="valid loss", color="r", lw=2)

        plt.xlabel("Epoch", fontsize=15)
        plt.ylabel("Loss (MSE)", fontsize=15)
        plt.legend(loc="best", fontsize=13)
        plt.title("Learning curve (M{}-{}-{}f)".format(m, pft, nr_features))
        plt.savefig(
            "../plots/Fig_LearningCurve_M{}_{}_{}f_32layers_earlystop_patience30.png".format(
                m, pft, nr_features
            )
        )
        plt.show()

    # test the lstm model
    test_dataset = dl.PhenoDataset(
        "../data/testdatasetm{}/".format(m),
        pft,
        preprocessing=train_dataset0.preprocessor.standardize,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_loss = test_model(test_loader, lstm, loss_function, nr_features, target)

    # pred gcc
    train_pred, train_obs, train_site, train_year = model_predict(
        train_loader, lstm, nr_features, target, denormalize=None
    )
    valid_pred, valid_obs, valid_site, valid_year = model_predict(
        valid_loader, lstm, nr_features, target, denormalize=None
    )
    test_pred, test_obs, test_site, test_year = model_predict(
        test_loader, lstm, nr_features, target, denormalize=None
    )

    # destandardize
    path_to_train = "../data/TrainingData_gcc_rcc_{}.csv".format(pft)
    df_train = pd.read_csv(path_to_train, delimiter=",")
    df_train = df_train.set_index(df_train.columns[0])
    df_train.index = pd.to_datetime(df_train.index)

    train_mean = df_train[target].mean()
    train_std = df_train[target].std()

    train_pred = (train_pred * train_std + train_mean).numpy()
    train_obs = (train_obs * train_std + train_mean).numpy()
    test_pred = (test_pred * train_std + train_mean).numpy()
    test_obs = (test_obs * train_std + train_mean).numpy()

    # R2 & RMSE
    RMSE_train = np.sqrt(
        metrics.mean_squared_error(train_obs.reshape(-1), train_pred.reshape(-1))
    )
    R2_train = metrics.r2_score(train_obs.reshape(-1), train_pred.reshape(-1))
    RMSE_valid = np.sqrt(
        metrics.mean_squared_error(valid_obs.reshape(-1), valid_pred.reshape(-1))
    )
    R2_valid = metrics.r2_score(valid_obs.reshape(-1), valid_pred.reshape(-1))
    RMSE_test = np.sqrt(
        metrics.mean_squared_error(test_obs.reshape(-1), test_pred.reshape(-1))
    )
    R2_test = metrics.r2_score(test_obs.reshape(-1), test_pred.reshape(-1))

    with open(
        "../lstm_outputs/results_lstm_hp_tuning_fixeddataset.csv", "a", newline=""
    ) as csvfile:
        fieldnames = [
            "PFT",
            "LSTM Model",
            "Feature",
            "Learning rate",
            "batch_size",
            "hidden_size",
            "patience",
            "dropout_rate",
            "weight_decay",
            "epoch_estop",
            "train_loss",
            "valid_loss",
            "test_loss",
            "R2_train",
            "R2_valid",
            "R2_test",
            "RMSE_train",
            "RMSE_valid",
            "RMSE_test",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # writer.writeheader()
        writer.writerow(
            {
                "PFT": pft,
                "LSTM Model": "M{}".format(m),
                "Feature": nr_features,
                "Learning rate": learning_rate,
                "batch_size": batch_size,
                "hidden_size": hidden_size,
                "patience": patience,
                "dropout_rate": dropout_rate,
                "weight_decay": weight_decay,
                "epoch_estop": epoch_estop,
                "train_loss": round(train_losses[-1], 4),
                "valid_loss": round(valid_losses[-1], 4),
                "test_loss": round(test_loss, 4),
                "R2_train": round(R2_train, 4),
                "R2_valid": round(R2_valid, 4),
                "R2_test": round(R2_test, 4),
                "RMSE_train": round(RMSE_train, 4),
                "RMSE_valid": round(RMSE_valid, 4),
                "RMSE_test": round(RMSE_test, 4),
            }
        )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an LSTM using shuffled data for one PFT with some of features"
    )
    parser.add_argument(
        "m", help='Put the length of blocks to shuffle or "full" to not shuffle'
    )
    parser.add_argument("pft", help="The PFT (can be DB, EN or GR)")
    parser.add_argument(
        "nr_features",
        type=int,
        help="6: dynamic variables; 8: dynamic + static climate; 9: dynamic (+ snow) + static climate; 14: dynamic + static climate & soil",
    )
    parser.add_argument(
        "target", help="target can be: gcc; rcc; gcc_lowess; rcc_lowess"
    )
    parser.add_argument(
        "batch_size",
        type=int,
        help="the hyperparameters tuning, lr: 0.01, 0.005, 0.001, 0.0001; dropout: 0, 0.1, 0.2, 0.5; batch_size: 4, 8, 16, 32",
    )
    args = parser.parse_args()

    run_lstm_hp_tuning(
        m=args.m,
        pft=args.pft,
        nr_features=args.nr_features,
        target=args.target,
        batch_size=args.batch_size,
    )
