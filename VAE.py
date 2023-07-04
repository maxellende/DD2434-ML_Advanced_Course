import torch
import torch.nn as nn

"""
@author: Harry
"""

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=500, z_dim=20):
        super().__init__()
        self.encoder_h = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim[0] * input_dim[1], hidden_dim),
            nn.Tanh()
        )

        self.z_mean = nn.Linear(hidden_dim, z_dim)
        self.z_log_var = nn.Linear(hidden_dim, z_dim)

        self.decoder_h = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim[0] * input_dim[1]),
            nn.Sigmoid()  # needed if we use binary cross entropy loss since they expect range [0,1]
        )

        self.x_mean = nn.Linear(hidden_dim, input_dim[0] * input_dim[1])
        self.x_log_var = nn.Linear(hidden_dim, input_dim[0] * input_dim[1])

    # f(x) = p(z|x)
    def encode(self, x):
        h = self.encoder_h(x)
        mu_z = self.z_mean(h)
        log_var_z = self.z_log_var(h)
        # p_z_x = torch.log(torch.normal(mu_z, log_var_z))
        return mu_z, log_var_z

    # f(z) = p(x|z)
    def decode(self, z):
        h = self.decoder_h(z)
        mu_x = self.z_mean(h)
        log_var_x = self.z_log_var(h)
        reconstructed_x = self.decoder(h)
        # p_x_z = torch.log(torch.normal(mu_x, log_var_x))
        return mu_x, log_var_x, reconstructed_x

    # z = mu_z + epsilon * (sigma_z)^2
    def reparametrize(self, mu, log_var):
        epsilon = torch.randn_like(log_var)
        return mu + epsilon * torch.exp(log_var * 0.5)

    def forward(self, x):
        mu_z, log_var_z = self.encode(x)
        z = self.reparametrize(mu_z, log_var_z)
        mu_x, log_var_x, reconstructed_x = self.decode(z)
        # x = torch.normal(mean=mu_x, std=torch.sqrt(torch.exp(0.5*log_var_x)))
        return z, mu_z, log_var_z, mu_x, log_var_x, reconstructed_x

    def loss(self, x, criterion):
        _, z_mean, z_log_var, x_mean, x_log_var, reconstructed_x = self.forward(x)
        # analytical form of -KL(q_fi(z|x) || p_theta(z))
        kl_div = 0.5 * torch.sum(1 + z_log_var
                                 - z_mean ** 2
                                 - torch.exp(z_log_var))

        # There are some motivations as to why we use MSE:
        # https://stats.stackexchange.com/questions/347378/variational-autoencoder-why-reconstruction-term-is-same-to-square-loss
        # if data is continuous then the decoder and encoder are gaussian according to the paper, we set p(x|z) to gaussian and get the following
        # log(P(x | z)) \propto log[e^(-|x-x'|^2)] = - |x-x'|^2
        # others use binary cross-entropy which seems to give results closer to the paper
        loss_log_likelihood = -criterion(reconstructed_x, torch.flatten(x, start_dim=1))

        # perform gradient ascent because the lower variational bound (ELBO) should be maximized
        # this is done by taking the negative loss (multiply with -1) in the train function and doing a descent
        # can be interpreted in two ways which are both equivalent:
        # 1. minimize -ELBO = KL(q_fi(z|x) || p_theta(z)) - log(p(x|z) <--- this is the one we are doing
        # 2. maximize ELBO = -KL(q_fi(z|x) || p_theta(z)) + log(p(x|z))
        return kl_div + loss_log_likelihood
