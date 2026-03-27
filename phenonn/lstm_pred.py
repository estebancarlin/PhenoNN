#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict gcc using lstm models

"""

import numpy as np
import pandas as pd
from . import dataloader_phenodata as dl
import torch
import os
from torch.utils.data import DataLoader
import argparse
from .lstm import LSTM

# Set CUDA device if specified
if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_predict(data_loader, model, denormalize=None):
    pred = torch.tensor([]).to(device)
    obs = torch.tensor([]).to(device)
    site = []
    year = torch.tensor([]).to(device)
    model.eval()
    model.to(device)
    with torch.no_grad():
        for batch in data_loader:
            X = torch.cat(
                (batch["features"][:, :, :6], batch["features"][:, :, 8:10]), axis=-1
            )
            y = batch["target"][:, :, 2]

            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            y_star = outputs[:, 365:].to(device)

            if denormalize is not None:
                X, outputs = denormalize(X, outputs)
                X, y = denormalize(X, y)

            pred = torch.cat((pred, y_star), 0)
            obs = torch.cat((obs, y), 0)

            site.extend(batch["sitename"])
            year = torch.cat((year, batch["year"].to(device)), 0)

    return pred, obs, site, year


def run_lstm_pred(
    batch_size=8,
    learning_rate=0.01,
    hidden_size=64,
    dropout_rate=0,
    weight_decay=0,
    patience=30,
    m=0,
    pft="DB",
    nr_features=8,
    target="gcc_lowess",
    input_path="../example/",
):
    num_layers = 1
    target_size = 1
    input_size = nr_features
    input_path = input_path

    train_min = 0.2876257959148408
    train_max = 0.4868239180735729

    # load the mininum gcc/rcc data for each site and each pft
    gccmins = pd.read_csv(os.path.join(input_path, "gcc_rcc_mins_site_veg.csv"))
    gccmins = gccmins.set_index(gccmins.columns[0])
    gccmins_pft = gccmins[gccmins["1"] == pft]

    normalizer = dl.Normalizer(torch.zeros((1,)), torch.ones((1,)))
    if pft == "DB":
        normalizer.min = torch.Tensor(
            [
                -3.5210e01,
                -2.6880e01,
                -1.0229e00,
                0.0000e00,
                1.7730e01,
                8.4446e00,
                0.0000e00,
                7.9700e02,
            ]
        )
        normalizer.max = torch.Tensor(
            [
                2.6040e01,
                4.1570e01,
                3.0722e03,
                1.6573e01,
                7.3497e02,
                1.5555e01,
                2.4811e02,
                1.3410e03,
            ]
        )

        model_nums = 25

    elif pft == "EN":
        normalizer.min = torch.Tensor(
            [
                -3.9570e01,
                -3.1940e01,
                -1.0229e00,
                0.0000e00,
                1.3570e01,
                7.1131e00,
                0.0000e00,
                4.2400e02,
            ]
        )
        normalizer.max = torch.Tensor(
            [
                2.6040e01,
                4.1570e01,
                3.0722e03,
                1.0632e01,
                7.0257e02,
                1.6887e01,
                1.2408e02,
                2.5650e03,
            ]
        )

        model_nums = 12

    else:
        normalizer.min = torch.Tensor(
            [
                -3.7050e01,
                -2.6760e01,
                2.8549e-01,
                0.0000e00,
                2.2400e01,
                7.8985e00,
                0.0000e00,
                3.1100e02,
            ]
        )
        normalizer.max = torch.Tensor(
            [
                2.9880e01,
                4.5070e01,
                4.1225e03,
                8.2590e00,
                6.4160e02,
                1.6102e01,
                1.1260e02,
                9.7400e02,
            ]
        )

        model_nums = 8

    test_dataset = dl.PhenoDataset(
        os.path.join(input_path, "testdata/"),
        pft,
        preprocessing=normalizer,
        device=device,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # LSTM model predict
    test_pred_allmodels = []
    lstm = LSTM(target_size, input_size, hidden_size, num_layers, dropout_rate, device)
    lstm = lstm.to(device)

    for mod_num in range(model_nums):
        # load the LSTM model
        lstm.load_state_dict(
            torch.load(
                os.path.join(
                    input_path,
                    'lstm_models/m{}_{}_8f_{}".format(m, pft)' + str(mod_num),
                )
            )
        )
        lstm.eval()

        # pred gcc
        test_pred, test_obs, test_site, test_year = model_predict(
            test_loader, lstm, denormalize=None
        )

        # denormalize
        test_pred = (
            (test_pred * (train_max - train_min) + train_min).detach().cpu().numpy()
        )
        test_obs = (
            (test_obs * (train_max - train_min) + train_min).detach().cpu().numpy()
        )

        # test: add site info
        test_site = np.array(test_site).reshape(len(test_site), 1)
        test_year = (
            test_year.detach().cpu().numpy().reshape(len(test_year), 1).astype(int)
        )
        test_siteyear = np.stack((test_site, test_year), axis=1).squeeze()
        test_pred_allsites = []
        for num_siteyr, siteyear in enumerate(test_siteyear):
            test_pred_siteyear = test_pred[num_siteyr]
            test_pred_siteyear = test_pred_siteyear.reshape(365, 1)
            # add min
            gccmin_site = gccmins_pft[gccmins_pft["0"] == siteyear[0]]["4"]
            test_pred_siteyear = test_pred_siteyear + gccmin_site.values

            test_pred_allsites.append(test_pred_siteyear)

        test_pred_allsites = np.concatenate(test_pred_allsites, axis=0)
        test_pred_allmodels.append(test_pred_allsites)

        df_pred_test = pd.DataFrame(test_pred_allsites)

    test_pred_allmodels = np.concatenate(test_pred_allmodels, axis=1)
    df_pred_test = pd.DataFrame(test_pred_allmodels)

    df_pred_test.to_csv("gcc_pred_test_{}_m{}.csv".format(pft, m))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an LSTM using shuffled data for one PFT with some of features"
    )
    parser.add_argument(
        "--m",
        help='Put the length of blocks to shuffle or "full" to not shuffle',
        default="full",
    )
    parser.add_argument("pft", help="The PFT (can be DB, EN or GR)")
    parser.add_argument(
        "batch_size",
        type=int,
        help="DB: batch_size = 8; EN: batch_size = 6; GR: batch_size = 4",
    )
    parser.add_argument("input_path", help='input data in the folder "example"')
    args = parser.parse_args()

    run_lstm_pred(
        m=args.m, pft=args.pft, batch_size=args.batch_size, input_path=args.input_path
    )
