import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util
import factorizers
import seer_train


class FactoredBlock(nn.Module):
  def __init__(self, func, output_dim):
    super(FactoredBlock, self).__init__()
    self.f = torch.tensor([func(i) for i in range(seer_train.half_feature_numel())], dtype=torch.long)
    self.inter_dim = 1 + self.f.max()
    self.weights = nn.Parameter(torch.zeros(self.inter_dim, output_dim))

  def virtual(self):
    with torch.no_grad():
      identity = torch.tensor([i for i in range(seer_train.half_feature_numel())], dtype=torch.long)
      conversion = torch.sparse.FloatTensor(
        torch.stack([identity, self.f], dim=0),
        torch.ones(seer_train.half_feature_numel()),
        size=torch.Size([seer_train.half_feature_numel(), self.inter_dim])).to(self.weights.device)
      return (conversion.matmul(self.weights)).t()

  def factored(self, x):
    N, D = x.size()
    assert D == seer_train.half_feature_numel()

    batch, active = x._indices()
    factored = torch.gather(self.f.to(x.device), dim=0, index=active)
    x = torch.sparse.FloatTensor(
      torch.stack([batch, factored], dim=0), 
      x._values(),
      size=torch.Size([N, self.inter_dim])).to(x.device).to_dense()
    return x

  def forward(self, x):
    x = self.factored(x)
    return x.matmul(self.weights)


class FeatureTransformer(nn.Module):
  def __init__(self, funcs, base_dim):
    super(FeatureTransformer, self).__init__()
    self.factored_blocks = nn.ModuleList([FactoredBlock(f, base_dim) for f in funcs])
    self.affine = nn.Linear(seer_train.half_feature_numel(), base_dim)

  def virtual_bias(self):
    return self.affine.bias.data

  def virtual_weight(self):
    return self.affine.weight.data + sum([block.virtual() for block in self.factored_blocks])

  def forward(self, x):
    return self.affine(x) + sum([block(x) for block in self.factored_blocks])


class NNUE(nn.Module):
  def __init__(self):
    super(NNUE, self).__init__()
    BASE = 128
    funcs = [factorizers.piece_position,]

    self.white_affine = FeatureTransformer(funcs, BASE)
    self.black_affine = FeatureTransformer(funcs, BASE)
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

  def flattened_parameters(self, log=True):
    def join_param(joined, param):
      if log:
        print(param.size())
      joined = np.concatenate((joined, param.cpu().flatten().numpy()))
      return joined
    
    joined = np.array([])
    # white_affine
    joined = join_param(joined, self.white_affine.virtual_weight().t())
    joined = join_param(joined, self.white_affine.virtual_bias())
    # black_affine
    joined = join_param(joined, self.black_affine.virtual_weight().t())
    joined = join_param(joined, self.black_affine.virtual_bias())
    # fc0
    joined = join_param(joined, self.fc0.weight.data.t())
    joined = join_param(joined, self.fc0.bias.data)
    # fc1
    joined = join_param(joined, self.fc1.weight.data.t())
    joined = join_param(joined, self.fc1.bias.data)
    # fc2
    joined = join_param(joined, self.fc2.weight.data.t())
    joined = join_param(joined, self.fc2.bias.data)
    # fc3
    joined = join_param(joined, self.fc3.weight.data.t())
    joined = join_param(joined, self.fc3.bias.data)
    return joined.astype(np.float32)


def loss_fn(prob, pred):
  epsilon = 1e-12
  entropy = -prob * prob.clamp(epsilon, 1-epsilon).log()
  loss = -prob * F.log_softmax(pred, dim=-1)
  return loss.mean() - entropy.mean()