# part of the code blocks are modified from: https://github.com/litian96/FedProx/blob/master/flearn/models/shakespeare/stacked_lstm.py
# credit goes to: Tian Li (litian96 @ GitHub)

import json
import logging
import numpy as np
import time
import math
import pickle
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from language_utils import *
from language_fedma import layerwise_fedma
from language_fedma import patch_weights

import language_model

from Transformer import merge, transformer, data
import torch
import sys
import argparse

from original_fedma.language_modeling import language_utils

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


BATCH_SIZE = 50
TRAIN_DATA_DIR = "./datum/shakespeare/data/train/"
TEST_DATA_DIR = "./datum/shakespeare/data/test/"

# all_data_niid_0_keep_10000_train_9.json
TRAIN_DATA_NAME = "all_data_niid_0_keep_10000_train_9.json"
TEST_DATA_NAME = "all_data_niid_0_keep_10000_test_9.json"

TRIAL_EPOCH=10

# since we used a relatively "fixed" model for shakespeare dataset
# we thus hardcode it here
#NUM_LAYERS=3 # we start from 1-layer LSTM now (so the 3 layers now is encoder|hidden LSTM|decoder)

NUM_LAYERS=3 # 2-layer LSTM (4 layers: encoder|hidden LSTM1|hidden LSTM2|decoder)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu")
    logger.info("Experiment running on device: {}".format(device))

    with open(TRAIN_DATA_DIR+TRAIN_DATA_NAME) as json_file:
        train_data = json.load(json_file)

    with open(TEST_DATA_DIR+TEST_DATA_NAME) as json_file:
        test_data = json.load(json_file)

    lr = 0.25
    clip = 0.25
    n_clients = 2
    retrain_flag = False
    communication_rounds = 10

    TRIAL_USER_NAME = train_data["users"][0:n_clients]  # this can be of length in range from 1 to 132
    start_time = time.time()
    total_loss = 0.0

    logger.info("Learning rate: {}".format(lr))
    logger.info("number of clients: {}, len trial user name: {}".format(n_clients, len(TRIAL_USER_NAME)))

    global_matched_model = transformer.get_model()

    for cr in range(communication_rounds):
        print("communication round: {}".format(cr))
        logger.info("Start to work on communication round-{}".format(cr))

        """with open("lstm_matching_assignments", "rb") as assignment_file:
            assignments_list = pickle.load(assignment_file)
        with open("lstm_matching_shapes", "rb") as ms_file:
            matching_shapes = pickle.load(ms_file)
        with open("matched_global_weights", "rb") as matched_weight_file:
            global_matched_model = pickle.load(matched_weight_file)"""

        # we will need to construct a new global test set based on all test data on each of the clients
        global_test_data = []
        global_test_label = []
        global_num_samples_test = 0
        for client_index in range(n_clients):
            client_user_name = TRIAL_USER_NAME[client_index]
            global_num_samples_test += len(test_data["user_data"][client_user_name]['x'])
            global_test_data += test_data["user_data"][client_user_name]['x']
            global_test_label += test_data["user_data"][client_user_name]['y']
        global_eval_batch_size = 10

        total_val_loss, global_correct_prediction, global_matched_model = transformer.eval_shakespeare(global_num_samples_test, global_eval_batch_size, global_test_data, global_test_label, device, global_matched_model)

        logger.info('*' * 89)
        logger.info('| Matched model on Global Testset | valid loss {:5.2f} | pred: {}/{} | acc: {:.4f}%'.format(total_val_loss, global_correct_prediction, global_num_samples_test, global_correct_prediction/global_num_samples_test*100.0))
        logger.info('*' * 89)
        #exit()

        ##########################################
        # stage of reconstruct each local model
        ##########################################
        models = []
        for client_index in range(n_clients):
            weights_global = global_matched_model.state_dict()
            model = transformer.get_model()
            model.load_state_dict(weights_global)
            models.append(model)

        ###############################################
        # local retraining process
        ###############################################
        # TODO: is that local retraining process only retraining after a certain layer? Or is this fine because it is at the beginning of the communication round?
        # TODO: regular training
        for model in models:
            model.to(device)
            model.train()
        #exit()

        #######################################################
        # start to conduct FedMA process
        #######################################################
        gamma = 1e-3
        sigma = 1.0
        sigma0 = 1.0
        it=5

        matching_shapes = []
        assignments_list = []

        order_of_layers = [
            ['encoder.weight'],
            [
                'transformer_encoder.layers.0.self_attn.in_proj_weight',
                'transformer_encoder.layers.0.self_attn.in_proj_bias',
            ],
            [
                'transformer_encoder.layers.0.self_attn.out_proj.weight',
                'transformer_encoder.layers.0.self_attn.out_proj.bias',
            ],
            [
                'transformer_encoder.layers.0.linear1.weight',
                'transformer_encoder.layers.0.linear1.bias',
            ],
            [
                'transformer_encoder.layers.0.linear2.weight',
                'transformer_encoder.layers.0.linear2.bias',
            ],
            [
                'decoder.weight',
                'decoder.bias'
            ]
        ]

        for i in range(len(order_of_layers)):
            print("layer: {}".format(i))
            # Let's do two clients to begin with

            # TODO: do I have to worry about the norm and positional encoder?

            for client_index in range(n_clients):
                client_user_name = TRIAL_USER_NAME[client_index]
                num_samples_train = len(train_data["user_data"][client_user_name]['x'])
                user_train_data = train_data["user_data"][client_user_name]
                print("test data size: {}".format(num_samples_train))

                model = models[client_index]
                total_loss, model = transformer.train_shakespeare(device, model, logger, num_samples_train, user_train_data, BATCH_SIZE)

            global_matched_model = merge.average_model(models[0], models[1])

            total_val_loss, global_correct_prediction, _ = transformer.eval_shakespeare(
                global_num_samples_test, global_eval_batch_size, global_test_data, global_test_label, device,
                global_matched_model)

            logger.info('| Matched model on Global Testset | valid loss {:5.2f} | pred: {}/{} | acc: {:.4f}%'.format(
                total_val_loss, global_correct_prediction, global_num_samples_test,
                global_correct_prediction / global_num_samples_test * 100.0))

        total_val_loss, global_correct_prediction, global_matched_model = transformer.eval_shakespeare(global_num_samples_test, global_eval_batch_size, global_test_data,
                                      global_test_label, device, global_matched_model)

        logger.info('*' * 89)
        logger.info('| Matched model on Global Testset | valid loss {:5.2f} | pred: {}/{} | acc: {:.4f}%'.format(total_val_loss, global_correct_prediction, global_num_samples_test, global_correct_prediction/global_num_samples_test*100.0))
        logger.info('*' * 89)
