import swadl


class DataModule:
    def __init__(self, root='../data', num_workers=4):
        swadl.utils.save_hparams(set_hparams=True)

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
