import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import factorizers
import seer_train


class FactoredBlock(nn.Module):
  def __init__(self, func, input_dim, output_dim):
    super(FactoredBlock, self).__init__()
    self.input_dim = input_dim
    self.f = torch.tensor([func(i) for i in range(input_dim)], dtype=torch.long)
    self.inter_dim = 1 + self.f.max()
    self.weights = nn.Parameter(torch.zeros(self.inter_dim, output_dim))

  def virtual(self):
    with torch.no_grad():
      identity = torch.tensor([i for i in range(self.input_dim)], dtype=torch.long)
      conversion = torch.sparse.FloatTensor(
        torch.stack([identity, self.f], dim=0),
        torch.ones(self.input_dim),
        size=torch.Size([self.input_dim, self.inter_dim])).to(self.weights.device)
      return (conversion.matmul(self.weights)).t()

  def factored(self, x):
    N, D = x.size()
    assert D == self.input_dim

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
  def __init__(self, funcs, input_dim, base_dim):
    super(FeatureTransformer, self).__init__()
    self.factored_blocks = nn.ModuleList([FactoredBlock(f, input_dim, base_dim) for f in funcs])
    self.affine = nn.Linear(input_dim, base_dim)

  def virtual_bias(self):
    return self.affine.bias.data

  def virtual_weight(self):
    return self.affine.weight.data + sum([block.virtual() for block in self.factored_blocks])

  def forward(self, x):
    return self.affine(x) + sum([block(x) for block in self.factored_blocks])


class NNUE(nn.Module):
  def __init__(self):
    super(NNUE, self).__init__()
    BASE = 160
    P_BASE = 256
    funcs = [factorizers.piece_position,]
    p_funcs = [factorizers.p_pawn_position,]

    self.white_affine = FeatureTransformer(funcs, seer_train.half_feature_numel(), BASE)
    self.black_affine = FeatureTransformer(funcs, seer_train.half_feature_numel(), BASE)
    self.p_white_affine = FeatureTransformer(p_funcs, seer_train.half_pawn_feature_numel(), P_BASE)
    self.p_black_affine = FeatureTransformer(p_funcs, seer_train.half_pawn_feature_numel(), P_BASE)

    self.d0 = nn.Dropout(p=0.05)
    self.fc0 = nn.Linear(2*BASE, 16)

    self.p_fc0 = nn.Linear(2*P_BASE, 16)
    self.p_fc1 = nn.Linear(16, 16)

    self.d1 = nn.Dropout(p=0.05)
    self.fc1 = nn.Linear(16, 16)
    self.d2 = nn.Dropout(p=0.05)
    self.fc2 = nn.Linear(32, 16)
    self.d3 = nn.Dropout(p=0.05)
    self.fc3 = nn.Linear(48, 1)
    

  def forward(self, pov, white, black, p_white, p_black):
    w_ = self.white_affine(white)
    b_ = self.black_affine(black)
    p_w_ = self.p_white_affine(p_white)
    p_b_ = self.p_black_affine(p_black)
    base = F.relu(pov * torch.cat([w_, b_], dim=1) + (1.0 - pov) * torch.cat([b_, w_], dim=1))
    p_base = F.relu(pov * torch.cat([p_w_, p_b_], dim=1) + (1.0 - pov) * torch.cat([p_b_, p_w_], dim=1))
    base, p_base = self.d0(base), self.d0(p_base)
    
    p = F.relu(self.p_fc0(p_base))
    p = self.p_fc1(p)

    x = F.relu(self.fc0(base) + p)
    x = self.d1(x)
    x = torch.cat([x, F.relu(self.fc1(x))], dim=1)
    x = self.d2(x)
    x = torch.cat([x, F.relu(self.fc2(x))], dim=1)
    x = self.d3(x)
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
    # p_white_affine
    joined = join_param(joined, self.p_white_affine.virtual_weight().t())
    joined = join_param(joined, self.p_white_affine.virtual_bias())
    # p_black_affine
    joined = join_param(joined, self.p_black_affine.virtual_weight().t())
    joined = join_param(joined, self.p_black_affine.virtual_bias())
    # fc0
    joined = join_param(joined, self.fc0.weight.data)
    joined = join_param(joined, self.fc0.bias.data)
    # p_fc0
    joined = join_param(joined, self.p_fc0.weight.data)
    joined = join_param(joined, self.p_fc0.bias.data)
    # p_fc1
    joined = join_param(joined, self.p_fc1.weight.data)
    joined = join_param(joined, self.p_fc1.bias.data)
    # fc1
    joined = join_param(joined, self.fc1.weight.data)
    joined = join_param(joined, self.fc1.bias.data)
    # fc2
    joined = join_param(joined, self.fc2.weight.data)
    joined = join_param(joined, self.fc2.bias.data)
    # fc3
    joined = join_param(joined, self.fc3.weight.data)
    joined = join_param(joined, self.fc3.bias.data)
    return joined.astype(np.float32)


def loss_fn(score, result, pred):
  lambda_ = 0.6
  loss = lambda_ * (score.sigmoid() - pred.sigmoid()) ** 2 + (1.0 - lambda_) * (result - pred.sigmoid()) ** 2
  return loss.mean()