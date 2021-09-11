import torch

import config
import seer_train
import dataset

cfg = config.Config('config.yaml')

reader = dataset.StochasticMultiplexReader(list([dataset.DataReader('/media/connor/7F35A067038168A9/seer_train3/test.txt')]))

print(reader.name())

data = dataset.SeerData(reader, cfg)
train_data_loader = torch.utils.data.DataLoader(data, batch_size=cfg.batch_size, num_workers=6, worker_init_fn=dataset.worker_init_fn)


for i in train_data_loader:
  print(dataset.post_process(i))
