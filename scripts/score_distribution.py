import numpy as np
import matplotlib.pyplot as plt
import config
import seer_train


cfg = config.Config('config.yaml')

reader = seer_train.SampleReader(cfg.data_write_path)

ys = np.zeros(reader.size())
for idx, sample in enumerate(reader):
  ys[idx] = sample.score()

plt.hist(ys)
plt.show()