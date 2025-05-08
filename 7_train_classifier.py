#! /usr/bin/env python3

import os
from datetime import datetime
import numpy as np
import cv2
import multiprocessing as mp
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from dataset_loader import ClipVectorDataset
import rnn

L_VECTOR_DIR = './Data/Vectors/Sabre/Left/'
R_VECTOR_DIR = './Data/Vectors/Sabre/Right/'
MODEL_WEIGHT_DIR = './Data/ModelWeights/Sabre/'
PLOT_DIR = './Data/Plots/Sabre/'

N_CLASSES = 2

def bce_accuracy(
    outputs, expected
):
    actual = outputs >= 0
    return sum(actual == expected) / len(expected)

def evaluate(
    model: torch.nn.Module, device,
    val_set: ClipVectorDataset, collate_fn,
    criterion,
):
    confusion = torch.zeros(N_CLASSES, N_CLASSES)
    num_correct = 0
    total = 0

    # just one big batch to evaluate
    eval_loader = DataLoader(
        dataset=val_set,
        batch_size=256,
        shuffle=True,
        collate_fn=collate_fn
    )

    model.eval()
    losses = []
    accuracies = []
    with torch.no_grad():
        for idx, batch in enumerate(eval_loader):
            inputs, labels = batch
            actual_batch_len = len(inputs.sorted_indices)

            outputs = model.forward(inputs.to(device)).cpu() # shape: (N, 2), N = batch size
            # logits = torch.nn.functional.softmax(outputs)
            loss = criterion(outputs.flatten(), labels)
            losses.append(loss.item())
            accuracies.append(bce_accuracy(outputs.flatten(), labels))

    return np.mean(losses), np.mean(accuracies)

def train(
    model: torch.nn.Module, device,
    train_set: ClipVectorDataset, val_set: ClipVectorDataset, collate_fn,
    criterion=None, optimizer=None,
    n_epoch=10, batch_size=32, learning_rate=0.2, report_every=5
):

    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    total_inputs = len(train_set)
    batches_trained = 0

    epoch_losses = []
    batch_losses = []
    # train_losses = []
    val_losses = []
    val_accs = []

    for epoch_no in range(n_epoch):
        print(f'epoch #{epoch_no}: ')
        epoch_losses.append(0)

        inputs_trained = 0
        past_progress = 0

        model.train()
        model.zero_grad()
        for idx, batch in enumerate(train_loader):
            inputs, labels = batch
            actual_batch_len = len(inputs.sorted_indices)

            outputs = model.forward(inputs.to(device)).cpu() # shape: (N, 2), N = batch size
            # logits = torch.nn.functional.softmax(outputs)
            torch.nn.functional.softsign

            loss = criterion(outputs.flatten(), labels)
            batch_losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.)
            optimizer.step()
            optimizer.zero_grad()

            inputs_trained += actual_batch_len
            epoch_losses[-1] += loss.item()

            # progress = np.floor(100 * inputs_trained / total_inputs)
            # if progress > past_progress:
            #     print('⬛')
            #     past_progress = progress
            # print(f'epoch {epoch_no} (batch {idx}, {100 * inputs_trained / total_inputs:.2f}%): train loss is {loss:g}')
            batches_trained += 1

            if (batches_trained) % report_every == 0:
                loss, acc = evaluate(model, device, val_set, collate_fn, criterion)
                val_losses.append(loss)
                val_accs.append(acc)
                print(f'\t({100 * inputs_trained / total_inputs:.2f}%): val loss is {loss:g}, acc is {acc:g}')
                model.train()

    timestamp = datetime.now().strftime("%Y-%m-%d-%Hh-%Mm-%Ss")
    save_name = f'ne{n_epoch}-bs{batch_size}-lr{learning_rate}-{timestamp}'
    torch.save(model, MODEL_WEIGHT_DIR + save_name + '.pth')

    plt.plot(batch_losses)
    plt.plot(np.array([[vl] * report_every for vl in val_losses]).flatten())
    plt.savefig(PLOT_DIR + save_name + '_loss.png')
    plt.cla()

    plt.plot(np.array([[acc] * report_every for acc in val_accs]).flatten())
    plt.savefig(PLOT_DIR + save_name + '_acc.png')

    plt.plot()

    return epoch_losses, batch_losses, val_losses


if __name__ == '__main__':
    if torch.cuda.is_available():
        print('Using GPU')
        device = torch.device('cuda:0')
    else:
        print('Using CPU !!!')
        device = torch.device('cpu')

    all_data = ClipVectorDataset(L_VECTOR_DIR, R_VECTOR_DIR)
    collate_fn = all_data.collate

    # train/test split of 90/10
    train_set, val_set = random_split(all_data, [.9, .1])

    # input_size = train_set[0][0].shape[1]
    # num_rnn_layers = 2
    # lstm_hidden_size = 500
    # rnn_hidden_size = 250
    # dropout=.2

    # model = rnn.LSTM_MultiRNN(
    #     input_size,
    #     num_rnn_layers,
    #     lstm_hidden_size,
    #     rnn_hidden_size,
    #     dropout
    # ).to(device)

    input_size = train_set[0][0].shape[1]
    num_lstm_layers = 2
    lstm_hidden_size = 500
    dropout=.2

    model = rnn.MultiLSTM(
        input_size,
        num_lstm_layers,
        lstm_hidden_size,
        dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=1)

    epoch_losses, batch_losses, val_losses = train(
        model, device,
        train_set, val_set, collate_fn,
        n_epoch=10, batch_size=32, optimizer=optimizer,
        report_every=30
    )

    plt.plot(batch_losses)
    plt.title('batch losses')
    plt.show()

    plt.plot(val_losses)
    plt.title('val losses')
    plt.show()
