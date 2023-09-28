import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import factorizers
import seer_train
import lasso

def loss_fn(score, result, pred):
    lambda_ = 0.6
    loss = lambda_ * (score.sigmoid() - pred.sigmoid()) ** 2 + (1.0 - lambda_) * (result - pred.sigmoid()) ** 2
    return loss.mean()


class FactoredBlock(nn.Module):
    def __init__(self, func, output_dim):
        super(FactoredBlock, self).__init__()
        self.f = torch.tensor([func(i) for i in range(
            seer_train.half_feature_numel())], dtype=torch.long)
        self.inter_dim = 1 + self.f.max()
        self.weights = nn.Parameter(torch.zeros(self.inter_dim, output_dim))

    def virtual(self):
        with torch.no_grad():
            identity = torch.tensor(
                [i for i in range(seer_train.half_feature_numel())], dtype=torch.long)
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
        self.factored_blocks = nn.ModuleList(
            [FactoredBlock(f, base_dim) for f in funcs])
        self.affine = nn.Linear(seer_train.half_feature_numel(), base_dim)

    def virtual_bias(self):
        return self.affine.bias.data

    def virtual_weight(self):
        return self.affine.weight.data + sum([block.virtual() for block in self.factored_blocks])

    def forward(self, x):
        return self.affine(x) + sum([block(x) for block in self.factored_blocks])


class FrozenFeatureTransformer(nn.Module):
    def __init__(self, base_dim, quantization_scale=512.0):
        super(FrozenFeatureTransformer, self).__init__()
        self.quantization_scale = quantization_scale
        self.inverse_quantization_scale = quantization_scale ** -1
        weights, biases = seer_train.feature_transformer_parameters()
        weights, biases = torch.tensor(weights).reshape(
            seer_train.half_feature_numel(), base_dim), torch.tensor(biases)
        self.register_buffer("frozen_weights", weights)
        self.register_buffer("frozen_biases", biases)
        self.register_buffer("frozen_quantized_weights",
                             weights.mul(quantization_scale).round().short())
        self.register_buffer("frozen_quantized_biases", biases.mul(
            quantization_scale).round().short())

    def virtual_bias(self):
        return self.frozen_biases.data

    def virtual_weight(self):
        return self.frozen_weights.data.t()

    def forward(self, x):
        return (x @ self.frozen_quantized_weights.float() + self.frozen_quantized_biases.float()).mul(self.inverse_quantization_scale)


class NNUE(nn.Module):
    def __init__(self, fine_tune=False):
        super(NNUE, self).__init__()
        BASE = 768
        funcs = [factorizers.piece_position, ]

        self.shared_affine = FrozenFeatureTransformer(
            BASE) if fine_tune else FeatureTransformer(funcs, BASE)
        self.fc0 = nn.Linear(2*BASE, 8)
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(24, 1)

    def forward(self, pov, white, black):
        w_ = self.shared_affine(white)
        b_ = self.shared_affine(black)
        
        base = F.relu(pov * torch.cat([w_, b_], dim=1) + (1.0 - pov) * torch.cat([b_, w_], dim=1))
        base = lasso.inject_lasso_loss(base)
        
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
        # shared_affine
        joined = join_param(joined, self.shared_affine.virtual_weight().t())
        joined = join_param(joined, self.shared_affine.virtual_bias())
        # fc0
        joined = join_param(joined, self.fc0.weight.data)
        joined = join_param(joined, self.fc0.bias.data)
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
