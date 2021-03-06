import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse

import config
import seer_train
import util
import dataset
import model


def all_sample_readers(sess):
  traindir = sess.get_train_path()
  paths = [os.path.join(traindir, f) for f in os.listdir(traindir)]
  return [dataset.DataReader(p) for p in paths]  


def next_incomplete(sess):
  for i in util.valid_man_counts():
    if not os.path.exists(sess.get_n_man_train_path(i)):
      return i
  return None


def train_step(nnue, sample, opt, queue, max_queue_size, report=False):
  pov, white, black, prob = sample
  pred = nnue(pov, white, black)
  loss = model.loss_fn(prob, pred)
  if report:
    print(loss.item())
  loss.backward()
  if(len(queue) >= max_queue_size):
    queue.pop(0)
  queue.append(loss.item())
  opt.step()
  nnue.zero_grad()


def main():
  cfg = config.Config('config.yaml')
  sample_to_device = lambda x: tuple(map(lambda t: t.to(cfg.device, non_blocking=True), dataset.post_process(x)))
  nnue = model.NNUE().to(cfg.device)
  valid_man_counts = util.valid_man_counts()
  sess = seer_train.Session(cfg.root_path).set_concurrency(cfg.concurrency)

  if (os.path.exists(cfg.model_save_path)):
    print('Loading model ... ')
    nnue.load_state_dict(torch.load(cfg.model_save_path))
    sess.load_weights(cfg.bin_model_save_path)


  writer = SummaryWriter(cfg.visual_directory)
  opt = optim.Adadelta(nnue.parameters(), lr=cfg.learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, cfg.step_size, gamma=cfg.gamma)

  queue = []
  total_steps = 0

  for epoch in range(cfg.epochs):
    reader = dataset.StochasticMultiplexReader(all_sample_readers(sess))

    print(f'training on: {reader.name()}')
    train_data = dataset.SeerData(reader, cfg)
    
    train_data_loader = torch.utils.data.DataLoader(train_data,
      batch_size=cfg.batch_size, 
      num_workers=cfg.concurrency,
      pin_memory=True,
      worker_init_fn=dataset.worker_init_fn)

    for i, sample in enumerate(train_data_loader):
      # update visual data
      if (i % cfg.test_rate) == 0 and i != 0:
        step = total_steps * cfg.batch_size
        train_loss = sum(queue) / len(queue)        
        writer.add_scalar('train_loss', train_loss, step)
      
      if (i % cfg.save_rate) == 0 and i != 0:
        print('Saving model ...')
        nnue.flattened_parameters().tofile(cfg.bin_model_save_path)
        sess.load_weights(cfg.bin_model_save_path)
        torch.save(nnue.state_dict(), cfg.model_save_path)

      train_step(nnue, sample_to_device(sample), opt, queue, max_queue_size=cfg.max_queue_size, report=(0 == i % cfg.report_rate))
      total_steps += 1
    
    n_man_incomplete = next_incomplete(sess)
    if n_man_incomplete is not None:
      print(f'generating data for {n_man_incomplete}')
      sess.maybe_generate_links_for(n_man_incomplete)

    scheduler.step()



if __name__ == '__main__':
  main()
