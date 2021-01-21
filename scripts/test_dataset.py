import config
import seer_train
import dataset
import util

from stochastic_multiplex_reader import StochasticMultiplexReader


cfg = config.Config('config.yaml')
sess = seer_train.Session(cfg.root_path)


reader = StochasticMultiplexReader([seer_train.train_n_man_path(cfg.root_path, i) for i in util.valid_man_counts()[:6]])
data = dataset.SeerData(reader, cfg)

for i in data:
  print(i)
