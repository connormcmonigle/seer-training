import os
import random
import torch
import numpy as np
import seer_train


def active_to_tensor(active):
    N = seer_train.max_active_half_features()
    return torch.tensor((list(active) + N*[-1])[:N], dtype=torch.long)


def sample_to_tensor(sample):
    w = active_to_tensor(sample.features().white)
    b = active_to_tensor(sample.features().black)
    p = torch.tensor([sample.score()]).float()
    z = torch.tensor([sample.result()]).float()
    return torch.tensor([sample.pov()]).float(), w, b, p, z


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    dataset = worker_info.dataset
    dataset.reader.configure_subset(worker_info.id, worker_info.num_workers)


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
    pov, w, b, p, z = x
    w, b = to_sparse(w), to_sparse(b)
    return pov, w, b, p, z


class SubsetConfigurable:
    def __init__(self, size, sample_reader):
        self.size_ = size
        self.reader_ = sample_reader
        self.process_id = 0
        self.num_processes = 1

    def configure_subset(self, process_id, num_processes):
        assert process_id < num_processes
        self.process_id = process_id
        self.num_processes = num_processes

    def size(self):
        return (self.size_ // self.num_processes) + (self.process_id < (self.size_ % self.num_processes))

    def __iter__(self):
        for idx, sample in enumerate(self.reader_):
            if self.process_id == idx % self.num_processes:
                yield sample


class SeerData(torch.utils.data.IterableDataset):
    def __init__(self, sample_reader, config):
        super().__init__()
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
        for sample in self.reader:
            mirror = np.random.rand() <= self.config.mirror_probability
            yield sample_to_tensor(sample.mirrored() if mirror else sample)
