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

def attention(q, k, v, allowed=None, scale=True):
    scores = q @ k.transpose(-2, -1)
    if scale:
        scores = scores / math.sqrt(q.shape[-1])
    if allowed is not None:
        scores = scores.masked_fill(~allowed, float('-inf'))
    weights = torch.softmax(scores, dim = -1)
    output = weights @ v

    return output, weights

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

    q = torch.tensor([[1., 0.]]) * 5
    k = torch.eye(2)
    v = torch.tensor([[10., 0.], [0., 20.]])
    out, weights = attention(q, k, v)
    out2, weights2 = attention(q, k, v+100)
    assert torch.equal(weights, weights2)
    assert not torch.allclose(out, out2)
    print('weights:', weights, '| retrieved:', out)
    q_big, k_big = torch.randn(64,128), torch.randn(64, 128)
    for scaled in [False, True]:
        _, a = attention(q_big, k_big, k_big, scale=scaled)
        entropy = -(a * a.clamp_min(1e-12).log()).sum(-1).mean()
        print('scaled =', scaled, '| average entropy =', round(entropy.item(), 3))


if __name__ == "__main__":
    main()