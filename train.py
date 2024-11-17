from datetime import datetime
from argparse import ArgumentParser
from datetime import time

from data_loader import load_data

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import *

import matplotlib.pyplot as plt

import os
import warnings
warnings.filterwarnings("ignore")


def evaluate(g, features, labels, mask, model):
    """Evaluate model accuracy"""
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, train_labels, val_labels, train_mask, val_mask, model, epochs, es_iters=None):

    # define train/val samples, loss function and optimizer
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-2, weight_decay=5e-4)
    training_process = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }

    # If early stopping criteria, initialize relevant parameters
    if es_iters:
        print("Early stopping monitoring on")
        loss_min = 1e8
        es_i = 0

    # training loop
    for epoch in range(epochs):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = evaluate(g, features, val_labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )

        val_loss = loss_fcn(logits[val_mask], val_labels).item()
        if es_iters:
            if val_loss < loss_min:
                loss_min = val_loss
                es_i = 0
            else:
                es_i += 1

            if es_i >= es_iters:
                print(f"Early stopping at epoch={epoch+1}")
                break

        training_process['epoch'].append(epoch)
        training_process['train_loss'].append(loss.item())
        training_process['val_loss'].append(val_loss)
        training_process['val_acc'].append(acc)

    return training_process


def plot_training_process(training_process, save_path=None):
    # plot training process
    # plot loss and accuracy in same figure and save the figure
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(training_process['epoch'], training_process['train_loss'], 'g-')
    ax1.plot(training_process['epoch'], training_process['val_loss'], 'r-')
    ax2.plot(training_process['epoch'], training_process['val_acc'], 'b-')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='r')
    ax2.set_ylabel('Accuracy', color='b')

    if save_path:
        plt.savefig(save_path)

    plt.show()


if __name__ == '__main__':

    parser = ArgumentParser()
    # you can add your arguments if needed
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--es_iters', type=int,
                        help='num of iters to trigger early stopping')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)

    # Load data
    features, graph, num_classes, \
        train_labels, val_labels, test_labels, \
        train_mask, val_mask, test_mask = load_data()

    train_mask = torch.tensor(train_mask).to(device)
    train_labels = torch.tensor(train_labels).to(device)
    val_mask = torch.tensor(val_mask).to(device)
    val_labels = torch.tensor(val_labels).to(device)
    test_mask = torch.tensor(test_mask).to(device)
    test_labels = torch.tensor(test_labels).to(device)
    graph = graph.to(device)
    features = features.to(device)

    # Initialize the model (Baseline Model: GCN)
    in_size = features.shape[1]
    out_size = num_classes

    # model = GCNSample(in_size, 16, out_size).to(device)
    # model = GCN(in_size, 8, out_size).to(device)
    # model = GAT(graph, in_size, 4, out_size, 4).to(device)
    # model = GATv2(in_size, 16, out_size, 3).to(device)
    model = SAGE(in_size, 3, out_size, 'gcn').to(device)
    # model = SAGEOneLayer(in_size, out_size, 'gcn').to(device)

    # model training
    print("Training...")
    training_process = train(graph, features, train_labels, val_labels, train_mask,
                             val_mask, model, args.epochs, args.es_iters)

    print("Testing...")
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[test_mask]
        _, indices = torch.max(logits, dim=1)

    # Export predictions as csv file
    print("Export predictions as csv file.")
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    with open(f'./output/output_{timestamp}.csv', 'w') as f:
        f.write('ID,Predict\n')
        for idx, pred in enumerate(indices):
            f.write(f'{idx},{int(pred)}\n')
    # Please remember to upload your output.csv file to Kaggle for scoring

    # Plot training process
    plot_training_process(
        training_process, save_path=f'./output/plot_{timestamp}.png')
