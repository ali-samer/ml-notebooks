from swadl import Losses, utils
from swadl.Models import Module
import swadl.Optim
import torch

class LinearRegressionScratch(Module):
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        utils.save_hparams()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        return torch.matmul(X, self.w) + self.b

    def loss(self, y_hat, y):
        return Losses.MSE()(y_hat, y)

    def configure_optimizers(self):
        return swadl.Optim.SGD([self.w, self.b], self.lr)