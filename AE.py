import torch.nn

class AE(torch.nn.Module):
    """
        This is an implementation of an Auto Encoder, adapted from Medium [https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1]
        @author: Erik Hamberg
    """

    def __init__(self, **kwargs):
        """
        :param kwargs:
        """
        super().__init__()
        self.z = kwargs["latent_shape"]
        self.encoder_hidden_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(
            in_features=kwargs["input_shape"], out_features=kwargs["hidden_dimension"]
        ))
        self.encoder_output_layer = torch.nn.Linear(
            in_features=kwargs["hidden_dimension"], out_features=kwargs["latent_shape"]
        )
        self.decoder_hidden_layer = torch.nn.Linear(
            in_features=kwargs["latent_shape"], out_features=kwargs["hidden_dimension"]
        )
        self.decoder_output_layer = torch.nn.Linear(
            in_features=kwargs["hidden_dimension"], out_features=kwargs["input_shape"])

    def forward(self, x):
        """
        :param x: image of a number
        :return: the recondstruction of the image
        """
        activation = self.encoder_hidden_layer(x)
        activation = torch.tanh(activation)
        code = self.encoder_output_layer(activation)
        code = torch.tanh(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.tanh(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.sigmoid(activation)
        return reconstructed

    def decode(self, code):
        """
        :param code: a tensor with an encoded input
        :return:
        """
        activation = self.decoder_hidden_layer(code)
        activation = torch.tanh(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.sigmoid(activation)
        return reconstructed

    def encode(self, x):
        """
        :param x: An image to encode
        :return: An encoding of the image
        """
        activation = self.encoder_hidden_layer(x)
        activation = torch.tanh(activation)
        code = self.encoder_output_layer(activation)
        code = torch.tanh(code)
        return code

    def generate_random(self):
        """
        :return: A randomly generated image.
        """
        return self.decode(torch.rand(self.z))

    def loss(self, x, criterion):
        return criterion(self.forward(x), torch.flatten(x, start_dim=1))

