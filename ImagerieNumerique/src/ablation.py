
import torch
import torch.nn as nn
import torch.legacy.nn as lnn

from functools import reduce
from torch.autograd import Variable

from src.models import completionnet_places2

# This was obtained from https://github.com/clcarwin/convert_torch_to_pytorch 
completionnet_ablation = lambda x: nn.Sequential( # Sequential,
	nn.Conv2d(4,64,(5, 5),(1, 1),(2, 2)),
	nn.BatchNorm2d(64),
	nn.ReLU(),
    nn.Dropout(x),
	nn.Conv2d(64,128,(3, 3),(2, 2),(1, 1)),
	nn.BatchNorm2d(128),
	nn.ReLU(),
    nn.Dropout(x),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.BatchNorm2d(128),
	nn.ReLU(),
    nn.Dropout(x),
	nn.Conv2d(128,256,(3, 3),(2, 2),(1, 1)),
	nn.BatchNorm2d(256),
	nn.ReLU(),
    nn.Dropout(x),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),(1, 1),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
    nn.Dropout(x),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),(1, 1),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
    nn.Dropout(x),
	nn.Conv2d(256,256,(3, 3),(1, 1),(2, 2),(2, 2),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
    nn.Dropout(x),
	nn.Conv2d(256,256,(3, 3),(1, 1),(4, 4),(4, 4),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
    nn.Dropout(x),
	nn.Conv2d(256,256,(3, 3),(1, 1),(8, 8),(8, 8),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
    nn.Dropout(x),
	nn.Conv2d(256,256,(3, 3),(1, 1),(16, 16),(16, 16),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
    nn.Dropout(x),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),(1, 1),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
    nn.Dropout(x),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),(1, 1),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
    nn.Dropout(x),
	nn.ConvTranspose2d(256,128,(4, 4),(2, 2),(1, 1),(0, 0)),
	nn.BatchNorm2d(128),
	nn.ReLU(),
    nn.Dropout(x),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.BatchNorm2d(128),
	nn.ReLU(),
    nn.Dropout(x),
	nn.ConvTranspose2d(128,64,(4, 4),(2, 2),(1, 1),(0, 0)),
	nn.BatchNorm2d(64),
	nn.ReLU(),
    nn.Dropout(x),
	nn.Conv2d(64,32,(3, 3),(1, 1),(1, 1)),
	nn.BatchNorm2d(32),
	nn.ReLU(),
    nn.Dropout(x),
	nn.Conv2d(32,3,(3, 3),(1, 1),(1, 1)),
	nn.Sigmoid(),
)

def copy_weights(A, B):
    weights = []
    for i, m in enumerate(A.modules()):
        if i == 0:
            continue
        if hasattr(m, 'weight'):
            weights.append(m.weight)
    j = 0
    for i, m in enumerate(B.modules()):
        if i == 0:
            continue
        if hasattr(m, 'weight'):
            m.weight = weights[j]
            j += 1

if __name__ == "__main__":
    A = completionnet_places2
    A.load_state_dict(torch.load('completionnet_places2.pth'))
    B = completionnet_ablation(0.1)
    copy_weights(A, B)
