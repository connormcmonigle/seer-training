import os
import torch

import model
import config
import seer_train
import dataset
import util
import train



cfg = config.Config('config.yaml')

nnue = model.NNUE().to(cfg.device)

if os.path.exists(cfg.model_save_path):
  print('Loading model ... ')
  nnue.load_state_dict(torch.load(cfg.model_save_path, map_location=cfg.device))
else:
  nnue.flattened_parameters().tofile(cfg.bin_model_save_path)
  torch.save(nnue.state_dict(), cfg.model_save_path)

num_total_parameters = sum(map(lambda x: torch.numel(x), nnue.parameters()))
num_effective_parameters = len(nnue.flattened_parameters(log=False))


print(f'total: {num_total_parameters}, effective: {num_effective_parameters}')
nnue.cpu()
nnue.eval()


while True:
  state = seer_train.StateType.parse_fen(input('fen: '))
  sample = seer_train.Sample(state, (0, 0, 0))
  tensors = [t.unsqueeze(0) for t in dataset.sample_to_tensor(sample)]
  pov, w, b, _ = dataset.post_process(tensors)
  prediction = nnue(pov, w, b).softmax(dim=-1)
  print(prediction)