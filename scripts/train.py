import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse

import config
import seer_train
import dataset
import model


def train_step(nnue, sample, opt, queue, max_queue_size, report=False):
  pov, white, black, score, result = sample
  pred = nnue(pov, white, black)
  loss = model.loss_fn(score, result, pred)
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

  if (os.path.exists(cfg.model_save_path)):
    print('Loading model ... ')
    nnue.load_state_dict(torch.load(cfg.model_save_path))

  writer = SummaryWriter(cfg.visual_directory)
  opt = optim.Adadelta(nnue.parameters(), lr=cfg.learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, cfg.step_size, gamma=cfg.gamma)

  queue = []
  total_steps = 0
  
  reader = dataset.StochasticMultiplexReader(list(map(lambda path: dataset.DataReader(path), cfg.data_read_paths)))
  for epoch in range(cfg.epochs):
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
        torch.save(nnue.state_dict(), cfg.model_save_path)

      train_step(nnue, sample_to_device(sample), opt, queue, max_queue_size=cfg.max_queue_size, report=(0 == i % cfg.report_rate))
      total_steps += 1

    scheduler.step()



if __name__ == '__main__':
  main()
