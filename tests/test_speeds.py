import json
import os
import random
import time
import unittest
from unittest import mock

import torch

from mvpp import MVPP
from mvpp_slash import MVPP as MVPPSlash
from neurasp import NeurASP
from slash import SLASH

elapsed_times = {}
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def time_method(class_obj, method_name, elapsed_times_key):
    original = getattr(class_obj, method_name)
    # Create entry for this method in elapsed times
    elapsed_times[elapsed_times_key] = 0
    def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        result = original(self, *args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        elapsed_times[elapsed_times_key] += elapsed_time
        return result
    return wrapper


def sample_examples(dataList, obsList, sample_size):
    # Sample random subset of examples
    idx_selection = random.sample(range(len(dataList)), sample_size)
    dataList = [dataList[idx] for idx in idx_selection]
    obsList = [obsList[idx] for idx in idx_selection]
    return dataList, obsList


def measure_neurasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name, opt=False, batch_size=64,
                        gpu=False, epoch=1):
    """Measure the speed of the original NeurASP code for an example."""
    NeurASPobj = NeurASP(dprogram, nnMapping, optimizers, gpu=gpu)
    with (mock.patch.object(MVPP, 'find_k_SM_under_obs',
                            time_method(MVPP, 'find_k_SM_under_obs', f'og_{example_name}_model')),
        mock.patch.object(MVPP, 'find_all_opt_SM_under_obs_WC',
                            time_method(MVPP, 'find_all_opt_SM_under_obs_WC', f'og_{example_name}_model')),
        mock.patch.object(MVPP, 'prob_of_interpretation',
                            time_method(MVPP, 'prob_of_interpretation', f'og_{example_name}_prob')),
        mock.patch.object(MVPP, 'mvppLearnRule',
                            time_method(MVPP, 'mvppLearnRule', f'og_{example_name}_grad'))):
        start_time = time.perf_counter()
        NeurASPobj.learn(dataList=dataList, obsList=obsList, epoch=epoch, opt=opt, batchSize=batch_size, storeSM=True, bar=True)
    elapsed_times[f'og_{example_name}_total'] = time.perf_counter() - start_time
    return elapsed_times[f'og_{example_name}_total']


def measure_slash_speed(dprogram, nnMapping, optimizers, dataListLoader, example_name, gpu=False, p_num=1,
                        method='same', epoch=1):
    """Measure the speed of the SLASH code for an example."""
    SLASHobj = SLASH(dprogram, nnMapping, optimizers, gpu=gpu)
    if method == 'same':
        sm_function = 'find_SM_with_same'
    else:
        sm_function = 'find_k_SM_under_query'
    with (mock.patch.object(MVPPSlash, sm_function,
                            time_method(MVPPSlash, sm_function, f'slash_{example_name}_model')),
        mock.patch.object(MVPPSlash, 'prob_of_interpretation',
                            time_method(MVPPSlash, 'prob_of_interpretation', f'slash_{example_name}_prob')),
        mock.patch.object(MVPPSlash, 'mvppLearnRule',
                            time_method(MVPPSlash, 'mvppLearnRule', f'slash_{example_name}_grad'))):
        start_time = time.perf_counter()
        for epoch_idx in range(epoch):
            SLASHobj.learn(dataListLoader, epoch_idx, batched_pass=True, p_num=p_num, method=method)
        elapsed_times[f'slash_{example_name}_total'] = time.perf_counter() - start_time
        return elapsed_times[f'slash_{example_name}_total']


def save_timings(expand=1.0):
    prefixes = {
        'og_': 'neurasp_',
        'slash_': "slash_",
    }
    known_prefixes = tuple(p for p in prefixes if p)  # non-empty ones
    # --- discover base example names from unprefixed keys ---
    new_elapsed_times = {}
    for key, value in elapsed_times.items():
        if key.startswith(known_prefixes):
            new_key = ''
            for prefix in known_prefixes:
                if key.startswith(prefix):
                    new_key = prefixes[prefix] + key[len(prefix):]
                    break
            new_elapsed_times[new_key] = value * expand
        elif not isinstance(value, (int, float)):
            new_elapsed_times[key] = value
    # Pretty-print (display only)
    print(json.dumps(new_elapsed_times, indent=4))
    # Save compactly, one record per line
    with open('timings.jsonl', 'a') as fo:
        fo.write(json.dumps(new_elapsed_times) + '\n')

class TestSpeeds(unittest.TestCase):

    def test_speeds_mnist_add(self, seed=None):
        """Test speeds of different implementations for the MNIST Addition task"""
        os.chdir(os.path.abspath(ROOT_DIR + '/../examples/mnistAdd'))
        elapsed_times.clear()
        if not seed:
            seed = random.randint(0, 100000)
        torch.manual_seed(seed)
        random.seed(seed)
        from examples.mnistAdd.dataGen import dataList, obsList
        from examples.mnistAdd.network import Net
        print("\nMNIST Add speed test")
        example_name = 'mnist_add'
        dprogram = ("img(i1). img(i2).\n"
                    "addition(A,B,N) :- digit(0,A,N1), digit(0,B,N2), N=N1+N2.\n"
                    "nn(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).")
        slash_program = ("img(i1). img(i2).\n"
                        "addition(A,B,N):- digit(0,+A,-N1), digit(0,+B,-N2), N=N1+N2, A!=B.\n"
                        "npp(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).")
        dataList_slash = [{k: i.squeeze(0) for k, i in dataDict.items()} for dataDict in dataList]
        dataListLoader = torch.utils.data.DataLoader(list(zip(dataList_slash, obsList)), batch_size=64)
        # Original code
        m = Net()
        nnMapping = {'digit': m}
        optimizers = {'digit': torch.optim.Adam(m.parameters(), lr=0.001)}
        measure_neurasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name, batch_size=64)
        # SLASH code
        m = Net()
        nnMapping = {'digit': m}
        optimizers = {'digit': torch.optim.Adam(m.parameters(), lr=0.001)}
        measure_slash_speed(slash_program, nnMapping, optimizers, dataListLoader, example_name)
        elapsed_times['task'] = f'mnist_add'
        save_timings()


    def test_speeds_add2x2(self):
        """Test speeds of different implementations for the Add 2x2 task"""
        os.chdir(os.path.abspath(ROOT_DIR + '/../examples/add2x2'))
        elapsed_times.clear()
        from examples.add2x2.dataGen import dataList, obsList
        from examples.add2x2.network import Net
        print("\nAdd 2x2 speed test")
        example_name = 'add2x2'
        dprogram = ("nn(digit(4,i), [0,1,2,3,4,5,6,7,8,9]).\n"
                    "add2x2(R1,R2,C1,C2) :- digit(0,i,N1), digit(1,i,N2), digit(2,i,N3), digit(3,i,N4), "
                    "R1=N1+N2, R2=N3+N4, C1=N1+N3, C2=N2+N4.")
        slash_program = (
            "npp(digit(4,i), [0,1,2,3,4,5,6,7,8,9]).\n"
            "add2x2(R1,R2,C1,C2) :- "
            "digit(0,+i,-N1), "
            "digit(1,+i,-N2), "
            "digit(2,+i,-N3), "
            "digit(3,+i,-N4), "
            "R1=N1+N2, "
            "R2=N3+N4, "
            "C1=N1+N3, "
            "C2=N2+N4."
        )
        m = Net()
        nnMapping = {'digit': m}
        optimizers = {'digit': torch.optim.Adam(m.parameters())}
        # Choose 1000 random examples
        dataList, obsList = sample_examples(dataList, obsList, 1000)
        # Original code
        m = Net()
        nnMapping = {'digit': m}
        optimizers = {'digit': torch.optim.Adam(m.parameters())}
        measure_neurasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name)
        # SLASH code
        m = Net()
        nnMapping = {'digit': m}
        optimizers = {'digit': torch.optim.Adam(m.parameters(), lr=0.001)}
        dataList_slash = [{k: i.squeeze(0) for k, i in dataDict.items()} for dataDict in dataList]
        dataListLoader = torch.utils.data.DataLoader(list(zip(dataList_slash, obsList)), batch_size=64)
        measure_slash_speed(slash_program, nnMapping, optimizers, dataListLoader, example_name, epoch=1)
        elapsed_times['task'] = f'add2x2'
        save_timings()


    def test_speeds_member(self, seed=None, n=5, sample_size=None):
        """Test speeds of different implementations for the Member task"""
        os.chdir(os.path.dirname(os.path.abspath(__file__)) + f'/../examples/member{n}')
        elapsed_times.clear()
        if not seed:
            seed = random.randint(0, 100000)
        torch.manual_seed(seed)
        random.seed(seed)
        print("\nMember speed test")
        if os.path.isfile('saved_models/test_stable_models.pkl'):
            os.remove('saved_models/test_stable_models.pkl')
        example_name = 'member'
        if n==3:
            from examples.member3.dataGen import dataList, obsList
            from examples.member3.network import Net
            dprogram = ("nn(digit(3,i), [0,1,2,3,4,5,6,7,8,9]).\n"
                        "member(D,0) :- digit(0,i,N1), digit(1,i,N2), digit(2,i,N3),\n"
                        "check(D), D!=N1, D!=N2, D!=N3.\n"
                        "member(D,1) :- check(D), not member(D,0).")
            slash_program = ("img(i1). img(i2). img(i3).\n"
                            "npp(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).\n"
                            "member(D,0) :- digit(0,+i1,-N1), digit(0,+i2,-N2), digit(0,+i3,-N3), check(D), D!=N1, D!=N2, D!=N3.\n"
                            "member(D,1) :- check(D), not member(D,0).")
        elif n==5:
            from examples.member5.dataGen import dataList, obsList
            from examples.member5.network import Net
            dprogram = ("nn(digit(5,i), [0,1,2,3,4,5,6,7,8,9]).\n"
                        "member(D,0) :- digit(0,i,N1), digit(1,i,N2), digit(2,i,N3), digit(3,i,N4), digit(4,i,N5),\n"
                        "check(D), D!=N1, D!=N2, D!=N3, D!=N4, D!=N5.\n"
                        "member(D,1) :- check(D), not member(D,0).")
            slash_program = ("img(i1). img(i2). img(i3). img(i4). img(i5).\n"
                            "npp(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).\n"
                            "member(D,0) :- digit(0,+i1,-N1), digit(0,+i2,-N2), digit(0,+i3,-N3), digit(0,+i4,-N4), digit(0,+i5,-N5), check(D), D!=N1, D!=N2, D!=N3, D!=N4, D!=N5.\n"
                            "member(D,1) :- check(D), not member(D,0).")
        expand = 1.0
        if sample_size is not None:
            expand = len(dataList) / sample_size
        m = Net()
        nnMapping = {'digit': m}
        optimizers = {'digit': torch.optim.Adam(m.parameters(), lr=0.001)}
        # Sample random examples
        sample_count = min(sample_size, len(dataList)) if sample_size is not None else min(1000, len(dataList))
        dataList, obsList = sample_examples(dataList, obsList, sample_count)
        # Original code
        measure_neurasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name, epoch=1)
        # SLASH code
        m = Net()
        nnMapping = {'digit': m}
        optimizers = {'digit': torch.optim.Adam(m.parameters(), lr=0.001)}
        dataList_slash = [{f'i{i+1}': dataDict['i'][i] for i in range(n)}for dataDict in dataList]
        dataListLoader = torch.utils.data.DataLoader(list(zip(dataList_slash, obsList)), batch_size=64)
        measure_slash_speed(slash_program, nnMapping, optimizers, dataListLoader, example_name, epoch=1)
        elapsed_times['task'] = f'member{n}'
        save_timings(expand=expand)

    def test_speeds_member3(self):
        """Wrapper so unittest can run the member benchmark with n=3."""
        self.test_speeds_member(n=3)

    def test_speeds_member5(self):
        """Wrapper so unittest can run the member benchmark with n=5."""
        self.test_speeds_member(n=5, sample_size=10)

    def test_speeds_card_arithmetic(self, op, cards, sample_size=None):
        """Test speeds of different implementations for the card arithmetic task."""
        os.chdir(os.path.abspath(ROOT_DIR + '/../examples/card_arithmetic'))
        elapsed_times.clear()
        from examples.card_arithmetic.dataGen import get_dataset
        from examples.card_arithmetic.network import Net
        print("\nCard arithmetic speed test")
        example_name = 'card_arithmetic'
        m = Net()
        nnMapping = {'card': m}
        optimizers = {'card': torch.optim.Adam(m.parameters())}
        trainDataset, __, dprogram = get_dataset(f'card_{op}_{cards}', './data')
        expand = 1.0
        if sample_size is not None:
            expand = len(trainDataset) / sample_size
            trainDataset = torch.utils.data.Subset(trainDataset, range(min(sample_size, len(trainDataset))))
        dataList = []
        obsList = []
        for data, obs in trainDataset:
            dataList.append({'p': data['p']})
            obsList.append(obs)
        # Original code
        measure_neurasp_speed(dprogram, nnMapping, optimizers, dataList, obsList, example_name, batch_size=32)
        # SLASH code
        m = Net()
        nnMapping = {'card': m}
        optimizers = {'card': torch.optim.Adam(m.parameters())}
        with open('data/playing_card_facts_slash.lp') as file:
            facts = file.read()
        player_facts = ' '.join(f'player(p{i + 1}).' for i in range(cards))
        facts = facts.replace('player(p1). player(p2). player(p3).', player_facts)
        slash_program = ("suit_value(d,0). suit_value(c,13). suit_value(s,26). suit_value(h,39).\n"
                        "player_value(P,RV+SV) :- suit(P,S), rank(P,R), rank_value(R,RV), suit_value(S,SV).\n"
                        "result(V1+V2) :- player_value(p1,V1), player_value(p2,V2).\n"
                        "npp(card(1,P), [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,"
                        "27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51]) :- player(P).\n")
        slash_program = slash_program + facts
        dataList_slash = [
            {f'p{i + 1}': dataDict['p'][i] for i in range(cards)}
            for dataDict in dataList
        ]
        dataListLoader = torch.utils.data.DataLoader(list(zip(dataList_slash, obsList)), batch_size=64)
        measure_slash_speed(slash_program, nnMapping, optimizers, dataListLoader, example_name, epoch=1)
        elapsed_times['task'] = f'card_{op}_{cards}'
        save_timings(expand=expand)

    def test_speeds_card_arithmetic_2sum(self):
        self.test_speeds_card_arithmetic('sum', 2, sample_size=128)

    def test_speeds_card_arithmetic_3sum(self):
        self.test_speeds_card_arithmetic('sum', 3, sample_size=128)