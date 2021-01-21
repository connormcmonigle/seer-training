from os import path
import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse

import config
import seer_train
import util
from stochastic_multiplex_reader import StochasticMultiplexReader
import dataset
import model


def train_step(M, sample, opt, queue, max_queue_size, report=False):
  pov, white, black, prob = sample


  pred = M(pov, white, black)
  loss = model.loss_fn(prob, pred)
  if report:
    print(loss.item())
  loss.backward()
  if(len(queue) >= max_queue_size):
    queue.pop(0)
  queue.append(loss.item())
  opt.step()
  M.zero_grad()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--resume', type=int, default=util.valid_man_counts()[0], help='piece count to resume training from')
  parser.add_argument('--all', default=False, action='store_true', help='train finalized network on all data')
  
  args = parser.parse_args()

  cfg = config.Config('config.yaml')

  sample_to_device = lambda x: tuple(map(lambda t: t.to(cfg.device, non_blocking=True), x))

  M = model.NNUE().to(cfg.device)

  valid_man_counts = util.valid_man_counts()

  sess = seer_train.Session(cfg.root_path).set_concurrency(cfg.concurrency)

  if (path.exists(cfg.model_save_path)):
    print('Loading model ... ')
    M.load_state_dict(torch.load(cfg.model_save_path))
    sess.load_weights(cfg.bin_model_save_path)

  #train_data = dataset.SeerData(sess.get_n_man_train_reader(valid_man_counts[0]), cfg)
  
  #train_data_loader = torch.utils.data.DataLoader(train_data,\
  #  batch_size=cfg.batch_size,\
  #  pin_memory=False)


  writer = SummaryWriter(cfg.visual_directory)

  #writer.add_graph(M, sample_to_device(next(iter(train_data_loader)))[:3])

  opt = optim.Adadelta(M.parameters(), lr=cfg.learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, 1, gamma=0.95)

  queue = []
  
  total_steps = 0

  if not args.all:
    print('generating data')

    for men in valid_man_counts:
    
      if men < args.resume:
        assert(path.exists(seer_train.train_n_man_path(cfg.root_path, men)))
        total_steps += sess.get_n_man_train_reader(men).size() // cfg.batch_size
        print('skipping {} man positions'.format(men))
        continue
      else:
        print('training on {} man positions'.format(men))
    
      train_data = dataset.SeerData(sess.get_n_man_train_reader(men), cfg)
  
      train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size)

      for i, sample in enumerate(train_data_loader):
        # update visual data
        if (i % cfg.test_rate) == 0 and i != 0:
          step = total_steps * cfg.batch_size
          train_loss = sum(queue) / len(queue)        
          writer.add_scalar('train_loss', train_loss, step)
      
        if (i % cfg.save_rate) == 0 and i != 0:
          print('Saving model ...')
          M.to_binary_file(cfg.bin_model_save_path)
          torch.save(M.state_dict(), cfg.model_save_path)

        train_step(M, sample_to_device(sample), opt, queue, max_queue_size=cfg.max_queue_size, report=(0 == i % cfg.report_rate))
        total_steps += 1

      scheduler.step()
      sess.load_weights(cfg.bin_model_save_path)
  
  else:
    print('training on all positions')

    reader = StochasticMultiplexReader([seer_train.train_n_man_path(cfg.root_path, i) for i in valid_man_counts])
    train_data = dataset.SeerData(reader, cfg)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size)

    for epoch in range(cfg.epochs):
      for i, sample in enumerate(train_data_loader):
        # update visual data
        if (i % cfg.test_rate) == 0 and i != 0:
          step = total_steps * cfg.batch_size
          train_loss = sum(queue) / len(queue)        
          writer.add_scalar('train_loss', train_loss, step)
      
        if (i % cfg.save_rate) == 0 and i != 0:
          print('Saving model ...')
          M.to_binary_file(cfg.bin_model_save_path)
          torch.save(M.state_dict(), cfg.model_save_path)

        train_step(M, sample_to_device(sample), opt, queue, max_queue_size=cfg.max_queue_size, report=(0 == i % cfg.report_rate))
        total_steps += 1

      scheduler.step()



if __name__ == '__main__':
  main()
