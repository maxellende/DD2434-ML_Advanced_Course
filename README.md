Reimplmentation of [VAE](https://arxiv.org/pdf/1312.6114.pdf), a variational autoencoder, in Pytorch.

To run first install dependencies

    pip install -r requirements.txt

Python version 3.7+ is used.

All hyperparameters can be found in the setup_parser.py.\
Layout of directories:
- figures contain all loss plots structured in subfolders vae and ae for the different models
- experiments contain training logs and checkpoints for the different experiments, structured in subfolders vae and ae 
and each specific experiment is contained within its own subdirectory
- generated_samples contain samples from the trained models, structured in subfolders vae and ae

Note that actions such as training, plotting, hyperparameter searching and generating samples are off by default.
Meaning that even if a file path is provided for the actions it will not be used unless the action is set to true.

# Experiments with VAE
### latent space 2 with binary cross-entropy (for plotting manifold)

    py main.py --model "vae" --ls 2 --epochs 1670 --train true --plot true --manifold true --generate true --n_samples 10 --save_checkpoint true --save_rate_epoch 100

### latent space 3 with binary cross-entropy 

    py main.py --model "vae" --ls 3 --epochs 1670 --train true --plot true --generate true --n_samples 10 --save_checkpoint true --save_rate_epoch 100

### latent space 5 with binary cross-entropy

    py main.py --model "vae" --ls 5 --lr 1e-2 --epochs 1670 --train true --plot true --generate true --n_samples 10 --save_checkpoint true --save_rate_epoch 100

### latent space 10 with binary cross-entropy

    py main.py --model "vae" --ls 10 --epochs 1670 --train true --plot true --generate true --n_samples 10 --save_checkpoint true --save_rate_epoch 100

### latent space 20 with binary cross-entropy

    py main.py --model "vae" --ls 20 --epochs 1670 --train true --plot true --generate true --n_samples 10 --save_checkpoint true --save_rate_epoch 100

### latent space 200 with binary cross-entropy

    py main.py --model "vae" --ls 200 --lr 1e-2 --epochs 1670 --train true --plot true --generate true --n_samples 10 --save_checkpoint true --save_rate_epoch 100

# Experiments with vanilla AE
### latent space 2, bce (for plotting manifold)

    py main.py --model "ae" --criterion "bce" --ls 2 --epochs 1670 --train true --plot true --manifold true --generate true --n_samples 10 --save_checkpoint true --save_rate_epoch 100

### latent space 3, bce

    py main.py --model "ae" --criterion "bce" --ls 3 --epochs 1670 --train true --plot true --generate true --n_samples 10 --save_checkpoint true --save_rate_epoch 100

### latent space 5, bce

    py main.py --model "ae" --criterion "bce" --ls 5 --lr 1e-2 --epochs 1670 --train true --plot true --generate true --n_samples 10 --save_checkpoint true --save_rate_epoch 100

### latent space 10, bce

    py main.py --model "ae" --criterion "bce" --ls 10 --epochs 1670 --train true --plot true --generate true --n_samples 10 --save_checkpoint true --save_rate_epoch 100

### latent space 20, bce

    py main.py --model "ae" --criterion "bce" --ls 20 --epochs 1670 --train true --plot true --generate true --n_samples 10 --save_checkpoint true --save_rate_epoch 100

### latent space 200, bce

    py main.py --model "ae" --criterion "bce" --ls 200 --lr 1e-2 --epochs 1670 --train true --plot true --generate true --n_samples 10 --save_checkpoint true --save_rate_epoch 100


## Loading example
For loading use for instance these arguments

    --load_checkpoint true --load_checkpoint_file "checkpoint_epoch_1670_iter_600.pt"

## Other helpful flags
Including but not limited to
```
--hs                    # hyperparameter search for learning rate for over all latent spaces
--seed                  # set RNG seed. Default is 123, which is what the experiments used
--optimizer             # adagrad or adam optimizer. Default is adagrad
--criterion             # loss function, choose between bce (binary data) or mse (continuous data)
```

For full message of all the arguments type

    py main.py -h
