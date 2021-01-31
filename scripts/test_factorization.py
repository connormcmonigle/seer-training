import os
import torch

import model
import config
import seer_train
import dataset
import util
import train
import factorizers



block = model.FactoredBlock(factorizers.material, 6)



while True:
  state = seer_train.StateType.parse_fen(input('fen: '))
  sample = seer_train.Sample(state, (0, 0, 0))
  tensors = [t.unsqueeze(0) for t in dataset.sample_to_tensor(sample)]
  pov, w, b, _ = dataset.post_process(tensors)
  factored, out, vp = block.factored(w), block(w), block.virtual()
  virtual_out = w.matmul(vp.t())
  print(factored, factored.size())
  print(f'real: {out.detach()}', out.size())
  #print(vp, vp.size())
  print(f'virt: {virtual_out}', virtual_out.size())
