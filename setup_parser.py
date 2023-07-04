import argparse
from pathlib import Path

"""
@author: Harry
"""

def create_argparser():
    parser = argparse.ArgumentParser(description="DD2434 VAE")
    # model
    parser.add_argument('--model', default='vae', type=str,
                        help="Model to use, either vae or ae")
    # params
    parser.add_argument('--batch_size', default=100, type=int,
                        help="minibatch size")
    parser.add_argument('--lr', default=2e-2, type=float,
                        help="learning rate, (3,10,20) has learning rate 2e-2 and (5,200) has 1e-2")
    parser.add_argument('--epochs', default=1670, type=int,
                        help="number of epochs")
    parser.add_argument('--hd', default=500, type=int,
                        help="hidden dimensions for the encoder/decoder")
    parser.add_argument('--ls', default=3, type=int,
                        help="latent space, 3,5,10,20,200")
    parser.add_argument('--seed', default=123, type=int,
                        help="random seed")

    # optimizer & loss function
    parser.add_argument('--optimizer', default="adagrad", type=str,
                        help="optimizer for the training")
    parser.add_argument('--criterion', default="bce", type=str,
                        help="loss function, choose between bce (binary data) or mse (continuous data)")

    # actions
    parser.add_argument('--hs', default=False, type=strtobool,
                        help="hyperparameter search for learning rate for over all latent spaces")
    parser.add_argument('--train', default=False, type=strtobool,
                        help="train")
    parser.add_argument('--plot', default=False, type=strtobool,
                        help="plot training and testing losses")
    parser.add_argument('--generate', default=False, type=strtobool,
                        help="generate samples")
    parser.add_argument('--n_samples', default=10, type=int,
                        help="number of samples to generate")
    parser.add_argument('--manifold', default=False, type=strtobool,
                        help="plot manifold for 2D latent space")

    # save related
    parser.add_argument('--save_checkpoint', default=False, type=strtobool,
                        help="to save model checkpoint when training")

    # frequency at which to save and log
    parser.add_argument('--save_rate_epoch', default=10, type=int,
                        help="epoch frequency saving checkpoint")
    parser.add_argument('--save_rate_iter', default=300, type=int,
                        help="iteration frequency saving checkpoint, inside the dataloader loop")
    parser.add_argument('--log_track_rate', default=300, type=int,
                        help="iteration frequency logging")

    # load related
    parser.add_argument('--load_checkpoint', default=False, type=strtobool,
                        help="to load model checkpoint")
    parser.add_argument('--load_checkpoint_file', default="checkpoints", type=format_path,
                        help="filename of model checkpoint to load")

    return parser


# source code for distutils.util.strtobool(), that is deprecated in python 3.12
def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


def format_path(path):
    # need to resolve otherwise will return a path object
    return Path(path).resolve()
