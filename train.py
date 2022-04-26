import torch
from torch.utils import DataLoader
import numpy as np
from models.img_denoising import DenoisingAutoencoder
# from datasets.testset_neucon_depths import TestsetNeuconDepths
from datasets.neucon_depths import NeuconDepths

lr = 0.01
epochs = 50
batch_size = 6

data = NeuconDepths('./Desktop/data', 'test')
# data = TestsetNeuconDepths('./Desktop/data', 'train')
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
model = DenoisingAutoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
