import hydra

from torch.utils.data import Dataset, DataLoader

from .cater import CaterDataset
from .moma import MomaDataset

from omegaconf import DictConfig

def get_empty_loader():
    return (dict(), dict())

def get_dataset(
        dataset_cfg: DictConfig, 
        mode: str, 
    ):
    if dataset_cfg.choice == 'cater': return CaterDataset(dataset_cfg=dataset_cfg, mode=mode)
    elif dataset_cfg.choice == 'moma': return MomaDataset(dataset_cfg=dataset_cfg, mode=mode)
    else: NotImplementedError("Unsupported Dataset")

def get_loader(
        dataset_cfg : DictConfig,
        mode : str, 
        batch_size : int, 
        num_workers : int,
        shuffle: bool,
    ):
    dtset = get_dataset(dataset_cfg, mode=mode)
    if mode == 'train':
        dataloader = DataLoader(
            dataset=dtset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=dtset.custom_collate_fn,
        )
    else:
        dataloader = DataLoader(
            dataset=dtset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=dtset.custom_collate_fn,
        )
    return dataloader

def get_loaders(cfg : DictConfig, trainer_mode : str):
    """
        NOTE
    """
    if trainer_mode == 'train_eval':
        return [
            hydra.utils.instantiate(cfg.dataset_train, dataset_cfg=cfg.dataset),    # train dataloader
            hydra.utils.instantiate(cfg.dataset_val, dataset_cfg=cfg.dataset),      # val dataloader
            get_empty_loader(),                                                     # test dataloader
        ]
    elif trainer_mode == 'test':
        return [
            get_empty_loader(),                                                     # train dataloader
            get_empty_loader(),                                                     # val dataloader
            hydra.utils.instantiate(cfg.dataset_test, dataset_cfg=cfg.dataset),     # test dataloader
        ]
    else:
        NotImplementedError("Unsupported Trainer mode")

