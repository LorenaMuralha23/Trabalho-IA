import numpy as np
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import models

# Classe para a rede neural


@ray.remote
class CNN:
    def __init__(self, train_data, validation_data, test_data, batch_size):
        self.train_loader = data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True)
        self.validation_loader = data.DataLoader(
            validation_data, batch_size=batch_size, shuffle=False)
        self.test_loader = data.DataLoader(
            test_data, batch_size=batch_size, shuffle=False)
        self.device = torch.device("cpu")

    def create_and_train_cnn(self, model_name, num_epochs, learning_rate, weight_decay, replicacoes):
        soma = 0
        acc_max = 0
        for i in range(replicacoes):
            model = self.create_model(model_name)
            optimizerSGD = self.create_optimizer(
                model, learning_rate, weight_decay)
            criterionCEL = self.create_criterion()
            self.train_model(model, self.train_loader, optimizerSGD, criterionCEL,
                             model_name, num_epochs, learning_rate, weight_decay, i)
            acc = self.evaluate_model(model, self.validation_loader)
            soma += acc
            if acc > acc_max:
                acc_max = acc
                iter_acc_max = i
        return soma / replicacoes, iter_acc_max

    def create_model(self, model_name):
        if model_name == 'VGG11':
            model = models.vgg11(weights='DEFAULT')
        elif model_name == 'Alexnet':
            model = models.alexnet(weights='DEFAULT')
        else:
            model = models.mobilenet_v3_large(weights='DEFAULT')

        # Freeze layers
        for param in model.parameters():
            param.requires_grad = False

        # Modify classifier for binary classification
        if hasattr(model, "classifier"):
            num_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_features, 2)
        else:  # For models like ResNet
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 2)

        return model

    def create_optimizer(self, model, learning_rate, weight_decay):
        params_to_update = [
            param for param in model.parameters() if param.requires_grad]
        optimizerSGD = optim.SGD(
            params_to_update, lr=learning_rate, weight_decay=weight_decay)
        return optimizerSGD

    def create_criterion(self):
        return nn.CrossEntropyLoss()

    def train_model(self, model, train_loader, optimizer, criterion, model_name, num_epochs, learning_rate, weight_decay, replicacao):
        model.to(self.device)
        min_loss = float("inf")
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(
                model, train_loader, optimizer, criterion)
            if train_loss < min_loss:
                min_loss = train_loss
                model_path = f"models/{model_name}_{num_epochs}_{learning_rate}_{weight_decay}_{replicacao}.pth"
                torch.save(model.state_dict(), model_path)

    def train_epoch(self, model, train_loader, optimizer, criterion):
        model.train()
        losses = []
        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return np.mean(losses)

    def evaluate_model(self, model, loader):
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                output = model(X)
                _, y_pred = torch.max(output, 1)
                total += len(y)
                correct += (y_pred == y).sum().item()
        return correct / total
