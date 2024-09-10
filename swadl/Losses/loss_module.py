class LossModule:
    def __init__(self, cb):
        self.__loss = cb if cb is not None else self.default_loss

    def __call__(self, *args, **kwargs):
        return self.__loss(*args, **kwargs)

    def default_loss(self, *args, **kwargs):
        raise NotImplementedError("Please define a default loss function.")
