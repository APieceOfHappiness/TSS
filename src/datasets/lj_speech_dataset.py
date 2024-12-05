import numpy as np
import torch
from tqdm.auto import tqdm

from pathlib import Path
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import read_json, write_json

# Не вижу смысла делать его параметром, так как пользователю он не нужен
# Оставлю для читабельности и гибкости редактирования
INDEX_NAME = 'index.json'

class LJSpeechDataset(BaseDataset):
    def __init__(
        self, dataset_path, split_name, *args, **kwargs
    ):
        print('constructor')
        dataset_path = Path(dataset_path)
        root_path = dataset_path / split_name
        index_path = root_path / INDEX_NAME

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(root_path, *args, **kwargs)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, root_path, *args, **kwargs):
        print('index')
        # index = []
        # data_path = ROOT_PATH / "data" / "example" / name
        # data_path.mkdir(exist_ok=True, parents=True)

        # # to get pretty object names
        # number_of_zeros = int(np.log10(dataset_length)) + 1

        # # In this example, we create a synthesized dataset. However, in real
        # # tasks, you should process dataset metadata and append it
        # # to index. See other branches.
        # print("Creating Example Dataset")
        # for i in tqdm(range(dataset_length)):
        #     # create dataset
        #     example_path = data_path / f"{i:0{number_of_zeros}d}.pt"
        #     example_data = torch.randn(input_length)
        #     example_label = torch.randint(n_classes, size=(1,)).item()
        #     torch.save(example_data, example_path)

        #     # parse dataset metadata and append it to index
        #     index.append({"path": str(example_path), "label": example_label})

        # # write index to disk
        # write_json(index, str(data_path / "index.json"))

        # return index
        return ['EEEEEE']
