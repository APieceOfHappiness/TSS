import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result_batch = {}

    result_batch['mel'] = pad_sequence([item['mel'] for item in dataset_items], batch_first=True)
    result_batch['audio'] = [item['audio'] for item in dataset_items]
    result_batch['transcription'] = [item['transcription'] for item in dataset_items]

    print(result_batch)
    return result_batch
