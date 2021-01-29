import torch

import config
import seer_train
import dataset
import util
import train

cfg = config.Config('config.yaml')
sess = seer_train.Session(cfg.root_path)


reader = dataset.StochasticMultiplexReader([dataset.DataReader(sess.get_n_man_train_path(2)), dataset.DataReader(sess.get_n_man_train_path(7))])

print(reader.name())

data = dataset.SeerData(reader, cfg)
train_data_loader = torch.utils.data.DataLoader(data, batch_size=cfg.batch_size, num_workers=6, worker_init_fn=dataset.worker_init_fn)


for i in train_data_loader:
  print(dataset.post_process(i))
