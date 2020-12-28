import config
import seer_train
import dataset

cfg = config.Config('config.yaml')
sess = seer_train.Session(cfg.root_path)

data = dataset.SeerData(sess.get_n_man_train_reader(2), cfg)

for i in data:
  print(i)
