import os
import functools
import numpy as np

import seer_train

class StochasticMultiplexReader:
  def __init__(self, readers):
    self.readers = readers  
    
    self.probabilities = np.array([reader.size() for reader in self.readers], dtype=np.float)
    self.probabilities /= self.probabilities.sum()
    print(self.probabilities)

    # {x : x \in \union readers... and x.idx % num_process = process_id }
    self.process_id = 0
    self.num_processes = 1

  def set_subset(self, process_id, num_processes):
    assert process_id < num_processes
    self.process_id = process_id
    self.num_processes = num_processes

  def next_in_subset(self, it):
    return [next(it, None) for _ in range(self.num_processes)][self.process_id]

  def __iter__(self):
    iters = [iter(reader) for reader in self.readers]
    items = [self.next_in_subset(it) for it in iters]

    while functools.reduce(lambda a, b: a if b is None else b, items) is not None:
      idx = np.random.choice(len(items),  p=self.probabilities)
      if items[idx] is not None:
        yield items[idx]

      items[idx] = self.next_in_subset(iters[idx])