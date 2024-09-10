from swadl import utils

class SGD:
    def __init__(self, params, lr):
        utils.save_hparams()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

