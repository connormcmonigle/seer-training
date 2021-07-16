import torch

import config
import seer_train
import dataset

cfg = config.Config('config.yaml')

reader = dataset.StochasticMultiplexReader(list(map(lambda path: dataset.DataReader(path), cfg.data_read_paths)))

print(reader.name())

data = dataset.SeerData(reader, cfg)
train_data_loader = torch.utils.data.DataLoader(data, batch_size=cfg.batch_size, num_workers=6, worker_init_fn=dataset.worker_init_fn)


for i in train_data_loader:
  print(dataset.post_process(i))
