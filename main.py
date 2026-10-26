import math
import time
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#Make the random number generation process repeatable 
#to facilitate debugging and comparison of experimental results
torch.manual_seed(42) 
torch.set_num_threads(1)
torch.set_printoptions(precision=3, sci_mode=False)
print("PyTorch:", torch.__version__, '| device: CPU')

def make_xy(sequence):
    input = sequence[:, :-1]
    output = sequence[:, 1:]
    return input, output

def lookup(ids, weight):
    return F.embedding(ids, weight)

def main():
    # embedding output: token = [1, 24] vector; (B,T,D) = (1, 5, 24)
    # Model predict output: (B,T,Vocab) = (1, 5, 12)
    sequence = torch.tensor([[1,5,2,1,5,2]])
    ids, targets = make_xy(sequence)
    weight = torch.randn(12, 24)
    embedded = lookup(ids, weight)
    assert ids.shape == targets.shape == (1, 5)
    assert embedded.shape == (1, 5, 24)
    assert torch.equal(embedded[0, 0], embedded[0, 3])
    print('input :', ids.tolist())
    print('target:', targets.tolist())
    print('embedding:', tuple(embedded.shape))

if __name__ == "__main__":
    main()