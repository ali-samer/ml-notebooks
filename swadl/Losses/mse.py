from swadl.Losses import LossModule


class MSE(LossModule):
    def __init__(self):
        super().__init__(cb=lambda *args, **kwargs: self.loss_func(*args, **kwargs))

    def loss_func(self, *args, **kwargs):
        if len(args) == 2:
            y_hat, y = args
            return self.__loss_func(y_hat, y)
        else:
            raise ValueError("Invalid number of arguments. Expected 2 arguments")

    def __loss_func(self, y_hat, y):
        l = (y_hat - y)**2 / 2
        return l.mean()