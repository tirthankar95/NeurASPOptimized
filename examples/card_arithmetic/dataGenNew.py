import os
import re
from os.path import join

import clingo
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import io, transforms

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)


class CardArithmetic(Dataset):

    def __init__(self, data_dir, downstream_labels, transform=None, latent_labels_file=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = downstream_labels
        if latent_labels_file:
            # Store latent labels if they are available
            latent_data = pd.read_csv(latent_labels_file)
            suits = np.array(['h', 'c', 's', 'd'])
            ranks = np.array(['2', '3', '4', '5', '6', '7', '8', '9', '10', 'j', 'q', 'k', 'a'])
            # Create an index to map semantic labels to numerical labels
            semantic_to_num = {}
            for s_idx, s in enumerate(suits):
                for r_idx, r in enumerate(ranks):
                    card_string = f"{r}{s}"
                    semantic_to_num[card_string] = r_idx + (s_idx * len(ranks))
            # Merge latent labels to downstream labels
            for i, player in enumerate(downstream_labels.columns[:-1], 1):
                self.data = pd.merge(self.data, latent_data, how='left', left_on=player, right_on='img')
                # Map semantic labels to numerical labels
                latent_col_name = f'latent_{i}'
                self.data[latent_col_name] = self.data['label'].map(semantic_to_num)
                # Drop redundant columns
                self.data = self.data.drop(columns=['img', 'label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        imgs = []
        latent_labels = []
        img_idxs = self.data.iloc[index]
        for name, value in img_idxs.items():
            if name.startswith('card_'):
                img_path = join(self.data_dir, f"{value}.jpg")
                img = self.transform(io.read_image(img_path))
                imgs.append(img)
            if name.startswith('latent'):
                latent_labels.append(value)
            if name == 'result':
                label = value
        if not latent_labels:
            return {'p': torch.stack(imgs)}, f':- not result({label}).'
        else:
            return {'p': (torch.stack(imgs), {'card': torch.Tensor(latent_labels)})}, f':- not result({label}).'


def read_dataset(data_file):
    train_file, val_file = data_file + '_train.csv', data_file + '_val.csv'
    train_data, val_data = pd.read_csv(train_file), pd.read_csv(val_file)
    return train_data, val_data


def get_dataset(task_name, image_folder, train_size = 10000):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((274, 174)),
        transforms.ToTensor(),
    ])
    train_data, val_data = read_dataset(dir_path + f'/data/{task_name}')
    trainDataset = CardArithmetic(f'{image_folder}/train',
                                train_data[:train_size], transform,
                                None)
    valDataset = CardArithmetic(f'{image_folder}/test',
                                val_data, transform,
                                None)
    with open(dir_path + '/data/playing_card_facts.lp') as file:
        facts = file.read()
    with open(dir_path + f'/data/{task_name}.lp') as file:
        task_rules = file.read()
    return trainDataset, valDataset, facts + '\n\n' + task_rules


def generate_dataset_from_asp(task_name, image_folder, sample_size=15000):
    """ Generate a dataset from an ASP task specification."""
    with open(dir_path + '/data/playing_card_facts.lp') as file:
        facts = file.read()
    with open(dir_path + f'/data/{task_name}.lp') as file:
        task_rules = ""
        line = file.readline()
        while line:
            if line.startswith('nn('):
                # Determine number of players from nn atom and don't add the atom to the program
                match = re.search(r'card\(([0-9]+),p\)', line)
                num_players = int(match[1])
            else:
                task_rules += line
            line = file.readline()
    # Generate all choices for suits and ranks
    players = " ".join([f'player({i}).' for i in range(num_players)])
    choices = "\n\n{suit(P,S): suit(S)}=1 :- player(P). {rank(P,R): rank(R)}=1 :- player(P)."
    program = facts + '\n\n' + task_rules + players + choices
    # Create a dataset with an entry for each player and a result entry
    dataset = {f'player_{i + 1}': [] for i in range(num_players)}
    dataset['result'] = []
    def on_model(m):
        atoms = str(m)
        for i in range(num_players):
            rank_match = re.search(fr'rank\({i},(\w+)\)', atoms)
            suit_match = re.search(fr'suit\({i},(\w)\)', atoms)
            dataset[f'player_{i + 1}'].append(rank_match.groups()[0] + suit_match.groups()[0])
        result_match = re.search(r'result\((\d+)\)', atoms)
        dataset['result'].append(f'{result_match.groups()[0]}')
    showers = "#show suit/2. #show rank/2. #show result/1."
    ctl = clingo.Control(['0'])
    ctl.add("base", [], program + showers)
    ctl.ground([("base", [])])
    ctl.solve(on_model=on_model)
    print(f"There are {len(set(dataset['result']))} unique labels.")
    # Take at most 15,000 rows
    if len(dataset) > sample_size:
        semantic_dataset = pd.DataFrame(dataset).sample(n=sample_size)
    else:
        semantic_dataset = pd.DataFrame(dataset)
    convert_semantic_to_numeric(semantic_dataset, task_name, image_folder, sample_size)


def generate_dataset_from_fun(fun, num_players, task_name, image_folder, sample_size=15000):
    """ Generate a dataset using a function that calculates the result given concepts."""
    suits = ['h', 'c', 's', 'd']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'j', 'q', 'k', 'a']
    concepts = []
    for suit in suits:
        for rank in ranks:
            concepts.append(rank + suit)
    data = np.random.choice(concepts, size=(sample_size, num_players))
    results = np.apply_along_axis(func1d=fun, axis=1, arr=data)
    results = results.reshape(-1, 1)
    data = np.hstack([data, results])
    column_names = [f'player_{i + 1}' for i in range(num_players)] + ['result']
    semantic_dataset = pd.DataFrame(data, columns=column_names)
    print(f"There are {len(set(semantic_dataset['result']))} unique labels.")
    convert_semantic_to_numeric(semantic_dataset, task_name, image_folder, sample_size)


def card_arithmetic_unique(data_row):
    suit_values = {'h': 39, 'c': 13, 's': 26, 'd': 0}
    rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
                'j': 11, 'q': 12, 'k': 13, 'a': 14}
    rankuit_to_num = np.vectorize(lambda rankuit: rank_values[rankuit[:-1]] + suit_values[rankuit[-1]])
    return np.sum(rankuit_to_num(data_row))

def card_arithmetic(data_row):
    suit_values = {'h': 4, 'c': 2, 's': 3, 'd': 1}
    rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
                'j': 11, 'q': 12, 'k': 13, 'a': 14}
    rankuit_to_num = np.vectorize(lambda rankuit: rank_values[rankuit[:-1]] * suit_values[rankuit[-1]])
    return np.sum(rankuit_to_num(data_row))


def convert_semantic_to_numeric(semantic_dataset, task_name, image_folder, sample_size):
    """Take a dataset with semantic entries (e.g. 5d) and replace them with random image ids of that card."""
    image_names = pd.read_csv(f'{image_folder}/playing_card_labels.csv')
    image_labels = pd.DataFrame()
    final_labels = pd.DataFrame()

    while len(final_labels) < sample_size:
        for column in semantic_dataset.columns:
            if column == 'result':
                image_labels[column] = semantic_dataset[column]
            else:
                # For each card, replace its label with a random image idx of a card with that label
                image_labels[column] = \
                    semantic_dataset[column].apply(lambda rankuit: image_names.loc[image_names['label'] == rankuit]
                    ['img'].sample().item())
        final_labels = pd.concat([final_labels, image_labels])

    final_labels = final_labels.sample(n=sample_size)
    output_path = join(dir_path, 'data', f'{task_name}_labels.csv')
    final_labels.to_csv(output_path, index=False)
    print(f"Wrote dataset to {output_path}")


if __name__ == '__main__':
    # generate_dataset_from_asp('card_arithmetic_4p', image_folder=dir_path + '/data')
    generate_dataset_from_fun(card_arithmetic, 4, 'card_arithmetic_4p',
                            dir_path + '/data', 30000)
