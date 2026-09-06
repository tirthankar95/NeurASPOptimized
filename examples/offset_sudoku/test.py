import torch
from dataGen import train_loader, validation_loader
from network import Sudoku_Net_Offset_bn
from torch import optim
from trainer import Test

# =============================================================================
# Instantiate Network
# =============================================================================

model = Sudoku_Net_Offset_bn().cuda()
model.load_state_dict(torch.load('model_name'))

# =============================================================================
# Network Parameters
# =============================================================================

learning_rate=.0001
epochs=2000

opt=optim.Adam(model.parameters(),lr=learning_rate)
criterion=torch.nn.BCELoss()

# =============================================================================
# Testing
# =============================================================================

model = Test(model, train_loader, validation_loader, opt, criterion, epochs)

 
