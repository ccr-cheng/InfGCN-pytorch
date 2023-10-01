_DATASET_DICT = {}


def register_dataset(name):
    def decorator(cls):
        _DATASET_DICT[name] = cls
        return cls

    return decorator


def get_dataset(cfg):
    d_cfg = cfg.copy()
    d_type = d_cfg.pop('type')
    train_cfg = d_cfg.pop('train', {})
    val_cfg = d_cfg.pop('validation', {})
    test_cfg = d_cfg.pop('test', {})
    return (
        _DATASET_DICT[d_type](split='train', **train_cfg, **d_cfg),
        _DATASET_DICT[d_type](split='validation', **val_cfg, **d_cfg),
        _DATASET_DICT[d_type](split='test', **test_cfg, **d_cfg),
    )
