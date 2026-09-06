import argparse
import os
import pickle

import pandas as pd
import torch
from dataGen import CardArithmetic
from torchvision import transforms

from examples.follow_suit.network import Net
from mvpp_new import MVPP
from newrasp import NeurASP

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((274, 174)),
        transforms.ToTensor(),
    ])

data = pd.read_csv(f'{dir_path}/data/{args.task}_labels_test.csv')
testDataset = CardArithmetic(f'{dir_path}/data/test', data[:10000], transform,
                                  f'{dir_path}/data/test/playing_card_labels.csv')
dataLoader = torch.utils.data.DataLoader(testDataset, batch_size=64)

m = Net()
m.load_state_dict(torch.load(f'{dir_path}/saved_models/{args.task}_card_{args.seed}.pth', map_location=torch.device('cpu')))
nnMapping = {'card': m}
optimizers = {'card': torch.optim.Adam(m.parameters())}

with open(dir_path + '/data/playing_card_facts.lp') as file:
    facts = file.read()
with open(dir_path + f'/data/{args.task}.lp') as file:
    task_rules = file.read()
dprogram = facts + '\n' + task_rules
NeurASPobj = NeurASP(dprogram, nnMapping, optimizers, gpu=True)
dmvpp = MVPP(NeurASPobj.mvpp['program'])
with open(f'saved_models/{args.task}_stable_models.pkl', 'rb') as fp:
    NeurASPobj.stableModels = pickle.load(fp)
downAcc, latentAcc = NeurASPobj.calculate_accuracies(dataLoader, dmvpp)
print(f"Downstream acc: {downAcc}")
print(f"Latent acc: {latentAcc}")
