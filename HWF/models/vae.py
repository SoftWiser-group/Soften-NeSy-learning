import torch
from torch import nn
from torch.nn import functional as F

class VanillaVAE(nn.Module):

    def __init__(self, z_dim=128, n_chan=1, im_shape=(45, 45)):
        super(VanillaVAE, self).__init__()
        self.z_dim = z_dim
        self.im_shape = im_shape
        self.n_chan = n_chan
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=n_chan, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 256, 3, 1),
            nn.ReLU(),
            nn.Conv2d(256, 2*z_dim, 1),
            nn.Flatten(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(z_dim, 256, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, n_chan, 3, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        self.mean = h[:,:self.z_dim]
        self.logvar = h[:,self.z_dim:]
        self.var = torch.exp(self.logvar)
        gaussian_noise = torch.randn_like(self.mean, device=x.device)
        z = self.mean + gaussian_noise * torch.sqrt(self.var)
        return z

    def inference(self, x):
        h = self.encoder(x)
        mean = h[:,:self.z_dim]
        return mean

    def decode(self, z):
        z = z.view(-1,self.z_dim,1,1)
        x_hat = self.decoder(z)
        return x_hat

    def calculate_kl(self):
        # kl
        return -0.5 * (1+self.logvar-self.mean**2-torch.exp(self.logvar)).sum() / self.mean.shape[0]
    
    def calculate_re(self, x_hat, x):
        # loss(x_hat, x)
        return F.binary_cross_entropy(x_hat, x, reduction='sum') / x.shape[0]

