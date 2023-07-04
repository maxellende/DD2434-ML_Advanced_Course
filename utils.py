import logging
import shutil
import os
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from VAE import VAE
from AE import AE
from torchsummary import summary

"""
@author: Harry
"""

def set_logger(log_path):
    # Credits: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, 'w+')
        file_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def init_weights_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.1)


def delete_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)


def create_optimizer(model, learning_rate, type):
    wd_temp = torch.normal(mean=0, std=0.001, size=(1, 1)).item()
    weight_decay = wd_temp * -1 if wd_temp < 0 else wd_temp
    if type.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return optimizer


def create_model(mnist_shape, hidden_dimen, latent_space, name, device):
    if name.lower() == 'vae':
        model = VAE(mnist_shape, hidden_dim=hidden_dimen, z_dim=latent_space)
    else:
        model = AE(input_shape=mnist_shape[0] * mnist_shape[1], hidden_dimension=hidden_dimen,
                   latent_shape=latent_space)
    # normal weight initializations of the network, mean=0, std=0.1
    model.apply(init_weights_normal)
    model.to(device)

    return model


def create_criterion(type):
    return {
        'l1': torch.nn.L1Loss(reduction='sum'),
        'mse': torch.nn.MSELoss(reduction='sum'),
        'bce': torch.nn.BCELoss(reduction='sum'),
    }[type.lower()]


def create_dataset(batch_size):
    data_path = "data_mnist"
    # note that ToTensor automatically converts value range into [0,1]
    # to keep the original range use PilToTensor
    transform = transforms.Compose([transforms.ToTensor()])
    training_set = torchvision.datasets.MNIST(data_path,
                                              download=True,
                                              train=True,
                                              transform=transform)
    test_set = torchvision.datasets.MNIST(data_path,
                                          download=True,
                                          train=False,
                                          transform=transform)

    train_loader = torch.utils.data.DataLoader(training_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0)

    return train_loader, test_loader


def print_model_summary(model, input_shape):
    summary(model, input_shape)


def learning_rate_hyperparam_search(parameters_to_optimize, parser, train_fn, mnist_shape, train_loader, test_loader,
                                    state, device):
    params_cand = list(map(lambda key: parameters_to_optimize[key], parameters_to_optimize))
    params_cand = np.array(np.meshgrid(params_cand[0], params_cand[1])).T.reshape(-1, 2)
    losses = []
    logging.info("Start hyperparameter search...")
    logging.info(f"Number of candidates: {len(params_cand)}")
    for candidate in params_cand:
        learning_rate = candidate[0]
        latent_space = int(candidate[1])
        logging.info(f"lr: {learning_rate}, ls:{latent_space}")
        model = create_model(mnist_shape, parser.hd, latent_space, device)
        optimizer = create_optimizer(model, learning_rate, parser.optimizer)
        criterion = create_criterion(parser.criterion)
        train_losses, _, _, _ = train_fn(model=model, optimizer=optimizer, criterion=criterion,
                                         train_loader=train_loader, test_loader=test_loader, state=state)
        train_loss_last_iter = train_losses[-1]
        losses.append((train_loss_last_iter, learning_rate, latent_space))
        logging.info(
            f"learning rate: {learning_rate}, latent space: {latent_space}, train loss: {train_loss_last_iter}")

    return {"best_lr_latent_space_3": min(list(filter(lambda x: x[2] == 3, losses)))[1],
            "best_lr_latent_space_5": min(list(filter(lambda x: x[2] == 5, losses)))[1],
            "best_lr_latent_space_10": min(list(filter(lambda x: x[2] == 10, losses)))[1],
            "best_lr_latent_space_20": min(list(filter(lambda x: x[2] == 20, losses)))[1],
            "best_lr_latent_space_200": min(list(filter(lambda x: x[2] == 200, losses)))[1]
            }
