from src.datasets.data_utils import prepare_lj_speech_dataset

if __name__ == '__main__':
    prepare_lj_speech_dataset(root_path='./datasets/LJSpeech', train_size=0.8, random_state=1)