from pathlib import Path
import config
import random
import copy

def shuffle(arr):
    res = copy.deepcopy(arr)
    random.shuffle(res)
    return res

def split(arr, split_point: float):
    return arr[:int(len(arr)*split_point)], arr[int(len(arr)*split_point):]

def build_exp_path(filename):
    exp_root = config.exp_root / Path(filename).stem
    if not exp_root.exists():
        exp_root.mkdir(parents=True)
    return exp_root


def random_split(data_root: Path, exp_root: Path, split_point: float = 0.8):
    categories = {cat_path.name for cat_path in data_root.glob('*')}
    train_data_root = exp_root / 'train_test_split' / 'train'
    test_data_root =  exp_root / 'train_test_split' / 'test'

    for category in categories:
        train, test = split(shuffle(list((data_root / category).glob('*'))), split_point)

        (train_data_root / category).mkdir(parents=True)
        for img_path in train:
            (train_data_root / category / img_path.name).symlink_to(img_path)

        (test_data_root / category).mkdir(parents=True)
        for img_path in test:
            (test_data_root / category / img_path.name).symlink_to(img_path)

    return train_data_root, test_data_root