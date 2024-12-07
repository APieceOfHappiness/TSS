import numpy as np
import torch
from tqdm.auto import tqdm

import torchaudio

from pathlib import Path
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import read_json, write_json

from src.utils.mel_spec_utils import MelSpectrogram, MelSpectrogramConfig
# Не вижу смысла делать его параметром, так как пользователю он не нужен
# Оставлю для читабельности и гибкости редактирования
INDEX_NAME = 'index.json'

class LJSpeechDataset(BaseDataset):
    def __init__(
        self, dataset_path, split_name, force_recreate, *args, **kwargs
    ):
        dataset_path = Path(dataset_path)
        split_path = dataset_path / split_name
        index_path = split_path / INDEX_NAME
        if index_path.exists() and not force_recreate:
            index = read_json(str(index_path))
        else:
            index = self._create_index(split_path, *args, **kwargs)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, split_path, transcription, audio, temp_mels, mel_creation_type,
                      *args, **kwargs):

        index = []
        assert not (transcription is None and split_path is None), 'At least one of the audio/transcription parameters must not be None'
        assert temp_mels is not None, 'temp_mels must be provided'

        data = audio if audio is not None else transcription
        data_names = [file.stem for file in (split_path / data).iterdir()]
        if not (split_path / temp_mels).exists():
            (split_path / temp_mels).mkdir(exist_ok=True)

        for data_name in tqdm(data_names, desc='Index'):
            audio_full = str(split_path / audio / f'{data_name}.wav') if audio is not None else None
            transcription_full = str(split_path / transcription / f'{data_name}.txt') if transcription is not None else None
            mel_full = str(split_path / temp_mels / f'{data_name}.pt')
            sample = {}

            sample['audio'] = audio_full

            sample['mel'] = mel_full
            mel = self.create_mel(audio_path=sample['audio'], 
                                  creation_type=mel_creation_type)
            torch.save(mel.squeeze(0).permute(1, 0), mel_full)  # [T, Batch, Bins]

            if transcription_full is not None:
                with open(transcription_full) as f:
                    sample['transcription'] = f.read()
            else:
                sample['transcription'] = None
            index.append(sample)

        write_json(index, str(split_path / "index.json"))
        return index
        
    def create_mel(self, audio_path, creation_type):        
        if creation_type == 'audio_to_mel':
            wav = self.load_audio(audio_path)
            mel_config = MelSpectrogramConfig()
            mel_model = MelSpectrogram(mel_config)
            mel = mel_model.forward(wav)
        elif creation_type == 'transcription_to_mel':
            mel = None
            #  TODO: for inference
        elif creation_type == 'load_from_folder':
            mel = None
            #  TODO: for convenience
        else:
            mel = None
        return mel


