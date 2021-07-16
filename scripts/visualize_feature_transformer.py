import os
import torch
import matplotlib.pyplot as plt

import config
import model


def to_image(x):
  print(f'x.size(): {x.size()}, x.abs().max(): {x.abs().max()}')
  # encoding ---- king square ---- piece type ---- piece square
  x = x.reshape(8,20,  8,8,  2,6,  8,8)

  # height ---- width
  x = x.permute(0,5,6,2,  1,4,7,3)
  x = (x.reshape(8*6*8*8, 20*2*8*8) * 3.0).abs().sigmoid()
  return x


def plot_input_weights_image(x, name, dst):
  title = f'abs(feature transformer) [{name}]'
  cmap = 'plasma'

  #dst.
  dst.set_title(title)
  dst.matshow(x, cmap=cmap)
  
  line_options = {'color': 'white', 'linewidth': 1}
  for i in range(1, 20):
    dst.axvline(x=2*8*8*i-1, **line_options)

  for j in range(1, 8):
    dst.axhline(y=6*8*8*j-1, **line_options)


def main():
  DEVICE = 'cpu'
  DPI = 100

  cfg = config.Config('config.yaml')
  nnue = model.NNUE().to(DEVICE)

  if os.path.exists(cfg.model_save_path):
    print('Loading model ... ')
    nnue.load_state_dict(torch.load(cfg.model_save_path, map_location=DEVICE))


  num_total_parameters = sum(map(lambda x: torch.numel(x), nnue.parameters()))
  num_effective_parameters = len(nnue.flattened_parameters(log=False))


  print(f'total: {num_total_parameters}, effective: {num_effective_parameters}')
  nnue.eval()

  w = to_image(nnue.white_affine.virtual_weight())
  b = to_image(nnue.black_affine.virtual_weight())

  fig, (axs_w, axs_b) = plt.subplots(1, 2, figsize=(2*w.size(1) // DPI, w.size(0) // DPI), dpi=DPI)
  plot_input_weights_image(w, 'white', axs_w)
  plot_input_weights_image(b, 'black', axs_b)
  
  plt.savefig(os.path.join(cfg.visual_directory, 'feature_transformer.png'))


if __name__ == '__main__':
  main()
