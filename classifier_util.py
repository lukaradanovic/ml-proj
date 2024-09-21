#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
import torch.nn.functional as F
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
from matplotlib import pyplot as plt

BATCH_SIZE = 32
NCLASSES = 11
N_EPOCHS = 30

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def bind_gpu(data):
    device = get_device()
    if isinstance(data, (list, tuple)):
        return [bind_gpu(data_elem) for data_elem in data]
    else:
        return data.to(device, non_blocking=True)


def train_classification(model, criterion, optimizer, number_of_epochs, train_loader):
    model.train()
    losses = []
    accuracies = []
    device = get_device()

    for epoch in range(number_of_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            correct += sum([int(int(labels[ind][val])==1) for ind, val in enumerate(predicted)])
            total += labels.size(0)
            
                
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        print(f"Epoch [{epoch + 1}/{number_of_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    return losses, accuracies


def evaluate_classification(model, criterion, loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    predicted_labels, true_labels = [], []
    device = get_device()

    with torch.no_grad():  # No gradient computation during evaluation
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs.squeeze(), labels).item()
            
            predicted = torch.argmax(outputs, dim=1)
            
            predicted_labels.extend(predicted.squeeze().tolist())
            
            true_labels.extend(torch.argmax(labels, dim=1).tolist())

            total_samples += labels.size(0)
            
            total_correct += sum([int(int(labels[ind][val])==1) for ind, val in enumerate(predicted)])
            

    # compute metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
    print(f'Model evaluation on: {loader}')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    # plot
    cm = confusion_matrix(true_labels, predicted_labels)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def evaluate(model, criterion, loader, train=True):
    loss, accuracy = evaluate_classification(model, criterion, loader)
    print(f'{"Train" if train else "Validation"} Set: Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')


def plot_classification(train_loss, train_accuracy, val_loss, val_accuracy):
    number_of_epochs = len(train_loss)
    epochs = range(1, number_of_epochs + 1)

    plt.figure(figsize=(12, 5))

    # Plotting Training Loss
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, train_loss, label='Training Loss')
    if val_loss:
        plt.plot(epochs, val_loss, label='Validation Loss')

    plt.legend()

    # Plotting Training Accuracy
    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accuracy, label='Training Accuracy')
    if val_accuracy:
        plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def load_weights_from_dict(model, path):
    pretrained_dict = torch.load(path)
    model.load_state_dict(pretrained_dict, strict=False) 
    for name, layer in model.named_children():
        if name+".weight" in pretrained_dict:
            for p in layer.parameters():
                p.requires_grad = False

