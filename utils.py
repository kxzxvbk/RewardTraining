import pickle
import os
import random
import string

import torch
import datasets
import pandas as pd


def load_pair_dataset(path):
    train_dataset_path = os.path.join(path, 'train.pkl')
    test_dataset_path = os.path.join(path, 'test.pkl')
    return {
        'train': datasets.Dataset.from_pandas(pd.DataFrame(_load_pair_dataset(train_dataset_path))),
        'test': datasets.Dataset.from_pandas(pd.DataFrame(_load_pair_dataset(test_dataset_path)))
    }


def load_abs_dataset(path):
    train_dataset_path = os.path.join(path, 'train.pkl')
    test_dataset_path = os.path.join(path, 'test.pkl')
    return {
        'train': datasets.Dataset.from_pandas(pd.DataFrame(_load_abs_dataset(train_dataset_path))),
        'test': datasets.Dataset.from_pandas(pd.DataFrame(_load_abs_dataset(test_dataset_path)))
    }


def calculate_mse(tensor1, tensor2):
    assert tensor1.shape == tensor2.shape and len(tensor1.shape) == 2
    return torch.mean((tensor1 - tensor2) ** 2)


def _load_pair_dataset(path):
    # Data is a pkl file.
    # Example: data = [{'chosen': 'xxx', 'rejected': 'yyy'}, {'chosen': 'aaa', 'rejected': 'bbb'}, ...]
    with open(path, 'rb') as f:
        data = pickle.load(f)
    keys = data[0].keys()
    assert 'chosen' in keys and 'rejected' in keys and len(keys) == 2, f"Bad data format, got: {keys}"
    return data


def _load_abs_dataset(path):
    # Data is a pkl file.
    # Example: data = [{'content': 'xxx', 'score': 0.01}, {'content': 'aaa', 'score': 0.02}, ...]
    with open(path, 'rb') as f:
        data = pickle.load(f)
    keys = data[0].keys()
    assert 'content' in keys and 'score' in keys and len(keys) == 2, f"Bad data format, got: {keys}"
    return data


def _gen_random_str(length=100):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def gen_pseudo_dataset():
    # Generate pseudo paired dataset.
    paired_dataset = []
    for _ in range(200):
        paired_dataset.append({'chosen': _gen_random_str(), 'rejected': _gen_random_str()})
    with open('./data/pair_dataset/train.pkl', 'wb') as f:
        pickle.dump(paired_dataset[:150], f)
    with open('./data/pair_dataset/test.pkl', 'wb') as f:
        pickle.dump(paired_dataset[150:], f)

    # Generate pseudo score dataset.
    paired_dataset = []
    for _ in range(200):
        paired_dataset.append({'content': _gen_random_str(), 'score': random.random()})
    with open('./data/abs_dataset/train.pkl', 'wb') as f:
        pickle.dump(paired_dataset[:150], f)
    with open('./data/abs_dataset/test.pkl', 'wb') as f:
        pickle.dump(paired_dataset[150:], f)


if __name__ == '__main__':
    gen_pseudo_dataset()
