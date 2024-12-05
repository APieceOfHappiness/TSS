from itertools import repeat
import pandas as pd
from hydra.utils import instantiate
from pathlib import Path
import os, shutil

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed

def prepare_lj_speech_dataset(root_path, train_size=0.8, random_state=1):
    root_path = Path(root_path)
    metadf = pd.read_csv(root_path / 'metadata.csv', delimiter='|', header=None)
    
    # fix some data bugs
    mask = metadf[2].isna()
    bugs = metadf.loc[mask, 1].str.split('|', expand=True)
    metadf.loc[mask, 2] = bugs.loc[:, 1] 

    train_path = root_path / 'train'
    val_path = root_path / 'val'
    
    def create_folder(path):
        if not path.exists():
            os.mkdir(str(path))

    for path in [train_path, val_path]:
        create_folder(path)
        create_folder(path / 'wavs')
        create_folder(path / 'transcriptions')

    metadf = metadf.sample(frac=1, random_state=random_state).reset_index(drop=True)
    train_len = int(len(metadf) * train_size)
    train_df = metadf.iloc[:train_len]
    val_df = metadf.iloc[train_len:]

    def save_df(df, path):
        for _, (audio_name, _, transcription) in df.iterrows():
            with open(path / 'transcriptions' / f'{audio_name}.txt', 'w', encoding='utf-8') as f:
                f.write(transcription)

            shutil.move(path.parent / 'wavs' / f'{audio_name}.wav', 
                        path / 'wavs' / f'{audio_name}.wav')

    save_df(train_df, train_path)
    save_df(val_df, val_path)
    shutil.rmtree(root_path / 'wavs')
    os.remove(root_path / 'metadata.csv')


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, device):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    # dataset partitions init
    datasets = instantiate(config.datasets)  # instance transforms are defined inside

    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]

        assert config.dataloader.batch_size <= len(dataset), (
            f"The batch size ({config.dataloader.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms
