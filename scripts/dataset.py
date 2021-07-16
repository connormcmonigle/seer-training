import os
import math
import random
import torch
import chess
import functools
import torch.nn.functional as F
import numpy as np
import seer_train


class DataReader:
  def __init__(self, path):
    assert os.path.exists(path)
    self.path = path
    self.size_ = seer_train.SampleReader(self.path).size()
    self.process_id = 0
    self.num_processes = 1

  def configure_subset(self, process_id, num_processes):
    assert process_id < num_processes
    self.process_id = process_id
    self.num_processes = num_processes

  def size(self):
    return self.size_

  def name(self):
    return f'DataReader({self.path})'

  def __iter__(self):
    reader = seer_train.SampleReader(self.path)
    for i, sample in enumerate(reader):
      if self.process_id == i % self.num_processes:
        yield sample


class StochasticMultiplexReader:
  def __init__(self, readers):
    self.readers = readers
    totals = np.array([reader.size() for reader in self.readers])
    self.size_ = totals.sum()
    self.probabilities = totals.astype(np.float) / float(self.size_)

    reader_names = ', \n  '.join([f'{r.name()} : {p}' for r, p in zip(readers, self.probabilities)])
    self.name_ = f'StochasticMultiplexReader(cardinality={self.size_},\n  [{reader_names}])'

  def configure_subset(self, process_id, num_processes):
    assert process_id < num_processes
    for reader in self.readers:
      reader.configure_subset(process_id, num_processes)

  def name(self):
    return self.name_

  def size(self):
    return self.size_

  def __iter__(self):
    iters = [iter(reader) for reader in self.readers]
    items = [next(it, None) for it in iters]

    while functools.reduce(lambda a, b: a if b is None else b, items) is not None:
      idx = np.random.choice(len(items), p=self.probabilities)
      if items[idx] is not None:
        yield items[idx]

      items[idx] = next(iters[idx], None)


def active_to_tensor(active):
  N = seer_train.max_active_half_features()
  return torch.tensor((list(active) + N*[-1])[:N], dtype=torch.long)


def sample_to_tensor(sample):
  w = active_to_tensor(sample.features().white)
  b = active_to_tensor(sample.features().black)
  p = torch.tensor([sample.score()]).float()
  return torch.tensor([sample.pov()]).float(), w, b, p


def worker_init_fn(worker_id):
  worker_info = torch.utils.data.get_worker_info()
  dataset = worker_info.dataset
  dataset.reader.configure_subset(worker_info.id, worker_info.num_workers)


class SeerData(torch.utils.data.IterableDataset):
  def __init__(self, sample_reader, config):
    super(SeerData, self).__init__()
    self.reader = sample_reader
    self.config = config
    self.shuffle_buffer = [None] * config.shuffle_buffer_size

  def __len__(self):
    return self.reader.size()

  def get_shuffled(self, sample):
    shuffle_buffer_idx = random.randrange(len(self.shuffle_buffer))
    result = self.shuffle_buffer[shuffle_buffer_idx]
    self.shuffle_buffer[shuffle_buffer_idx] = sample
    return result

  def sample_iter(self):
    for sample in self.reader:
      val = self.get_shuffled(sample)
      if val != None:
        yield val
    # clear remaining entries from buffer
    for val in self.shuffle_buffer:
      if val != None:
        yield val

  def __iter__(self):
    for sample in self.sample_iter():
      mirror = np.random.rand() <= self.config.mirror_probability
      yield sample_to_tensor(sample.mirrored() if mirror else sample)


def to_sparse(x):
  N = seer_train.max_active_half_features()
  batch_idx = torch.arange(x.size(0)).unsqueeze(-1).expand(x.size(0), N)
  active_mask = x.ge(0)
  indices = torch.stack([batch_idx[active_mask], x[active_mask]], dim=0)
  return torch.sparse.FloatTensor(
    indices,
    torch.ones(active_mask.sum()),
    torch.Size([x.size(0), seer_train.half_feature_numel()]))


def post_process(x):
  pov, w, b, p = x
  w, b = to_sparse(w), to_sparse(b)
  return pov, w, b, p