import time

import torch

from examples.mnistAdd.dataGen import dataList, obsList, test_loader
from examples.mnistAdd.network import Net
from neurasp import NeurASP

dprogram = '''
img(i1). img(i2).
addition(A,B,N) :- digit(0,A,N1), digit(0,B,N2), N=N1+N2.
nn(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).
'''

m = Net()
nnMapping = {'digit': m}
optimizers = {'digit': torch.optim.Adam(m.parameters(), lr=0.001)}
NeurASPobj = NeurASP(dprogram, nnMapping, optimizers)


saveModelPath = 'examples/mnistAdd/data/model.pt'
startTime = time.time()
dataset = list(zip(dataList, obsList))

for i in range(1):
    print(f'Epoch {i+1}...')
    time1 = time.time()
    NeurASPobj.learn(dataset, epoch=1, storeSM=False, bar=True, task='mnistAdd')
    time2 = time.time()
    acc, _ = NeurASPobj.testNN('digit', test_loader)
    print(f'Test Acc: {acc:0.2f}%')
    print(f'Storing the trained model into {saveModelPath}')
    torch.save(m.state_dict(), saveModelPath)
    print('--- train time: %s seconds ---' % (time2 - time1))
    print('--- test time: %s seconds ---' % (time.time() - time2))
    print('--- total time from beginning: %s minutes ---' % int((time.time() - startTime)/60) )