import os
import functools
import numpy as np

import seer_train

class StochasticMultiplexReader:
  def __init__(self, paths):
    for path in paths:
      assert os.path.exists(path)
    self.readers = [seer_train.SampleReader(path) for path in paths]    
    self.probabilities = np.array([reader.size() for reader in self.readers], dtype=np.float)
    self.probabilities /= self.probabilities.sum()
    print(self.probabilities)

  def __iter__(self):
    iters = [iter(reader) for reader in self.readers]
    items = [next(it, None) for it in iters]

    while functools.reduce(lambda a, b: a if b is None else b, items) is not None:
      idx = np.random.choice(len(items),  p=self.probabilities)
      if items[idx] is not None:
        yield items[idx]

      items[idx] = next(iters[idx], None)