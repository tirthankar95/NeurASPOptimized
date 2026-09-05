import torch
from dataGen import train_loader, validation_loader
from network import Sudoku_Net_Offset_bn
from torch import optim
from trainer import Train_Test

# =============================================================================
# Instantiate Network
# =============================================================================

model = Sudoku_Net_Offset_bn()
model.cuda()

# =============================================================================
# Network Parameters
# =============================================================================

learning_rate = .0001
epochs = 2000

opt = optim.Adam(model.parameters(),lr=learning_rate)
criterion = torch.nn.BCELoss()

# =============================================================================
# Training
# =============================================================================

model = Train_Test(model, train_loader, validation_loader, opt, criterion, epochs)
torch.save(model.state_dict(), 'model_data70.pt')
