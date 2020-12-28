import os
import math
import bitstring
import struct
import random
import torch
import chess
import torch.nn.functional as F
import numpy as np
import util
import seer_train

def active_to_tensor(active):
  return torch.sparse.FloatTensor(\
    torch.tensor(list(active), dtype=torch.long).unsqueeze(0),\
    torch.ones(len(active)),\
    torch.Size([seer_train.half_feature_numel()]))


def sample_to_tensor(sample):
  sample.features()
  w = active_to_tensor(sample.features().white)
  b = active_to_tensor(sample.features().black)
  p = torch.tensor([sample.win(), sample.draw(), sample.loss()]).float()
  return torch.tensor([sample.pov()]).float(), w, b, p


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

  def __iter__(self):
    for sample in self.reader:
      val = self.get_shuffled(sample)
      if val != None:
        yield sample_to_tensor(val)
    # clear remaining entries from buffer
    for val in self.shuffle_buffer:
      if val != None:
        yield sample_to_tensor(val)