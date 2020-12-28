import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util
import seer_train


class NNUE(nn.Module):
  def __init__(self):
    super(NNUE, self).__init__()
    BASE = 128
    self.white_affine = nn.Linear(seer_train.half_feature_numel(), BASE)
    self.black_affine = nn.Linear(seer_train.half_feature_numel(), BASE)
    self.fc0 = nn.Linear(2*BASE, 32)
    self.fc1 = nn.Linear(32, 32)
    self.fc2 = nn.Linear(64, 32)
    self.fc3 = nn.Linear(96, 3)
    

  def forward(self, pov, white, black):
    w_ = self.white_affine(white)
    b_ = self.black_affine(black)
    base = F.relu(pov * torch.cat([w_, b_], dim=1) + (1.0 - pov) * torch.cat([b_, w_], dim=1))
    x = F.relu(self.fc0(base))
    x = torch.cat([x, F.relu(self.fc1(x))], dim=1)
    x = torch.cat([x, F.relu(self.fc2(x))], dim=1)
    x = self.fc3(x)
    return x

  def to_binary_file(self, path):
    joined = np.array([])
    for p in self.parameters():
      print(p.size())
      joined = np.concatenate((joined, p.data.cpu().t().flatten().numpy()))
    print(joined.shape)
    joined.astype('float32').tofile(path)


def loss_fn(prob, pred):

  epsilon = 1e-12
  entropy = -prob * prob.clamp(epsilon, 1-epsilon).log()
  
  loss = -prob * F.log_softmax(pred, dim=-1)

  return loss.mean() - entropy.mean()