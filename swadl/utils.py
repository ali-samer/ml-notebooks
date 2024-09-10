import inspect


def save_hparams(ignore=[], set_hparams=False):
    frame = inspect.currentframe().f_back
    _, _, _, local_vars = inspect.getargvalues(frame)
    obj = local_vars.get('self', None)
    hparams = {k: v for k, v in local_vars.items()
               if k not in set(ignore + ['self']) and not k.startswith('_')}
    for k, v in hparams.items():
        setattr(obj, k, v)
    if set_hparams:
        setattr(obj, "hparams", hparams)


reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)
numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
