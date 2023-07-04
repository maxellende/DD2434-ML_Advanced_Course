import os.path

import utils
from time import time
import logging
import random as random
from setup_parser import create_argparser
from plot_figures import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
parser = create_argparser().parse_args()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

# root paths for experiments
ROOT_EXPERIMENT_PATHS = os.path.join("experiments", parser.model, f"ls{str(parser.ls)}")

"""
@author: Harry
@author: Maxellende
"""


def train(model, optimizer, criterion, train_loader, test_loader, name='vae', state=None):
    # number of steps or epoch not given in the paper so have to experiment with
    # 60k training data from MNIST, 100 mini batches
    training_losses = []
    test_losses = []

    # where training epochs start, if loaded from checkpoint then it will be higher than 1
    start_epoch = 1
    if parser.load_checkpoint:
        training_losses = state['training_losses']
        test_losses = state['test_losses']
        start_epoch = state['epoch']

    logging.info("Start training...")
    for epoch in range(start_epoch, parser.epochs + 1):
        model.train()
        running_loss = 0

        for i, (x, _) in enumerate(train_loader):
            # Forward and back prop
            x = x.to(DEVICE)
            # if vae then need to do gradient ascent by multiplying with -1 and do gradient descent
            # but for ae don't need to do anything
            loss = -model.loss(x, criterion) if name.lower() == 'vae' else model.loss(x, criterion)
            optimizer.zero_grad()
            loss.backward()

            # if vae need to revert to NLL (negative log likelihood) otherwise no need for actions
            loss = -loss if name.lower() == "vae" else loss
            # update model parameters
            optimizer.step()
            running_loss += loss.item()
            # revert loss back to its original form (we expect it to be negative in its original form)
            avg_loss = loss.item() / len(x)

            if epoch % parser.save_rate_epoch == 0 and i % parser.save_rate_iter == 0 \
                    and parser.save_checkpoint:
                save_model_checkpoint(epoch, parser.epochs, i, model, optimizer, test_losses, training_losses)

            if i % parser.log_track_rate == 0:
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(x), len(train_loader.dataset),
                           100. * i / len(train_loader),
                    avg_loss))

        training_loss = running_loss / len(train_loader.dataset)
        training_losses.append(training_loss)
        logging.info('====> Epoch: {} Average training loss: {:.4f}'.format(
            epoch, training_loss))

        test_loss = eval(model, test_loader, criterion, None)
        test_losses.append(test_loss)
        logging.info('====> Epoch: {} Average testing loss: {:.4f}'.format(
            epoch, test_loss))

        # make sure to save at the end of epoch according to given save epoch rate
        if (epoch % parser.save_rate_epoch == 0 or epoch == parser.epochs) and parser.save_checkpoint:
            save_model_checkpoint(epoch, parser.epochs, 600, model, optimizer, test_losses, training_losses)

    logging.info(f"settings: {vars(parser)}")

    return training_losses, test_losses


def save_model_checkpoint(epoch, num_epoch, iteration, model, optimizer,
                          test_losses,
                          training_losses):
    # save model checkpoints
    state = {
        'epoch': epoch,
        'total_epochs': num_epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'training_losses': training_losses,
        'test_losses': test_losses,
        # the loss may not be needed since we start over from a new epoch when resuming
    }
    checkpoint_path = os.path.join(ROOT_EXPERIMENT_PATHS, "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    torch.save(state, os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch}_iter_{iteration}.pt'))


@torch.no_grad()
def eval(model, test_loader, criterion, stop=None):
    """
    Train on the test dataset
    Can be used to test model after one epoch
    """
    model.eval()
    running_loss = 0

    if not stop:  # evaluate model on whole test dataset
        for i, (x, _) in enumerate(test_loader):
            # Forward and back prop
            x = x.to(DEVICE)
            loss = model.loss(x, criterion)
            running_loss += loss.item()

        return running_loss / len(test_loader.dataset)

    if stop:  # evaluate model on only one batch
        it = iter(test_loader)
        x, c = next(it)
        x = x.to(DEVICE)
        loss = model.loss(x, criterion)

        return loss.item() / len(x)


def test_image(model, x, parser, plot_path, stop=None, criterion=None):
    if not criterion:
        criterion = torch.nn.MSELoss(reduction='sum')
    x = x.to(DEVICE)
    if parser.model.lower() == 'vae':
        _, _, _, _, _, reconstructed_x = model.forward(x)
    else:
        reconstructed_x = model.forward(x)
    loss = model.loss(x, criterion)
    x = x[0].detach().cpu().numpy()
    reconstructed_x = reconstructed_x.detach().cpu().numpy().reshape(x.shape)
    save_test_image(x, reconstructed_x, loss.item(), parser, plot_path)


def resume_training(model, optimizer):
    load_checkpoint_path = os.path.join(ROOT_EXPERIMENT_PATHS, "checkpoints", parser.load_checkpoint_file)
    if torch.cuda.is_available():
        state = torch.load(load_checkpoint_path)
    else:
        state = torch.load(load_checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    model.to(DEVICE)
    epoch = state['epoch']
    # logging.info(f"Loading model checkpoint")
    # logging.info(f"state_dict: {model.state_dict()}, optimizer: {optimizer.state_dict()}, epoch: {epoch}")
    return epoch, optimizer, model, state


def main():
    random.seed(parser.seed)
    np.random.seed(parser.seed)
    torch.manual_seed(parser.seed)
    torch.cuda.manual_seed_all(parser.seed)

    # paths
    generate_sample_path = os.path.join("generated_samples", parser.model)
    plot_path = os.path.join("figures", parser.model)
    if not os.path.exists(ROOT_EXPERIMENT_PATHS):
        os.makedirs(ROOT_EXPERIMENT_PATHS)
    if not os.path.exists(generate_sample_path):
        os.makedirs(generate_sample_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # setup logger so can save logging.infos, call logging.info
    if parser.hs or parser.train:
        # logger filehandler seems to be able to create a directory if it doesn't exist
        utils.set_logger(os.path.join(ROOT_EXPERIMENT_PATHS, "train.log"))

    train_loader, test_loader = utils.create_dataset(batch_size=parser.batch_size)
    MNIST_shape = train_loader.dataset.data.shape[1:]  # exclude batch dimension
    print(len(test_loader))
    print(len(train_loader))

    model = utils.create_model(mnist_shape=MNIST_shape, hidden_dimen=parser.hd,
                               latent_space=parser.ls, name=parser.model, device=DEVICE)
    criterion = utils.create_criterion(parser.criterion)
    optimizer = utils.create_optimizer(model=model, learning_rate=parser.lr, type=parser.optimizer)

    # load model checkpoint
    state = {}
    epochs = parser.epochs
    if parser.load_checkpoint:
        epochs, optimizer, model, state = resume_training(model, optimizer)
        # training_losses = list(map(lambda x: -x, state['training_losses']))
        # test_losses = list(map(lambda x: -x, state['test_losses']))
        training_losses = state['training_losses']
        test_losses = state['test_losses']

    # hyperparameter search
    if parser.hs:
        params_to_optimize = {'lr': [0.01, 0.02, 0.1], 'ls': [3, 5, 10, 20, 200]}
        best_learning_rates = utils.learning_rate_hyperparam_search(parameters_to_optimize=params_to_optimize,
                                                                    parser=parser,
                                                                    train_fn=train, mnist_shape=MNIST_shape,
                                                                    train_loader=train_loader, test_loader=test_loader,
                                                                    state=state, device=DEVICE)
        logging.info(f"Best learning rates for different latent spaces: {str(best_learning_rates)}")

    # train
    if parser.train:
        tic = time()
        training_losses, test_losses = \
            train(model=model, optimizer=optimizer, criterion=criterion, train_loader=train_loader,
                  test_loader=test_loader, name=parser.model, state=state)
        final_time = time() - tic
        logging.info('Done (t={:0.2f}m)'.format(final_time / 60))
        epochs = parser.epochs

    # this can appear weird, and there are better solutions, but because of the control flow this is
    # the assignment for the correctly displayed epochs after the action of loading and training has been performed
    parser.epochs = epochs
    # plot evolution of loss along epochs/datapoint
    if parser.plot:
        plot_epoch_losses(training_losses, test_losses, parser, plot_path)

    if parser.generate:
        # run model on one image to test
        inputs, classes = next(iter(test_loader))
        x = inputs[np.random.randint(len(inputs))]
        test_image(model, x, parser, plot_path, stop=None, criterion=criterion)

        # # generate an image x_hat from noise as input into the trained decoder
        generate_images(model, DEVICE, parser, generate_sample_path)

    if parser.manifold and parser.ls == 2:
        # plot manifold for latent dimension of 2
        plot_manifold(model, DEVICE, parser, plot_path, n=12)

    utils.print_model_summary(model, MNIST_shape)

if __name__ == '__main__':
    main()
