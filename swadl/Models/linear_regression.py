import swadl
import torch

class LinearRegression(swadl.Models.Module):
    def __init__(self, lr):
        super().__init__()
        swadl.utils.save_hparams()
        self.net = torch.nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        fn = torch.nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)