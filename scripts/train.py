from os import path
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


def all_sample_readers(sess, n_end=None):
  paths = [sess.get_n_man_train_path(i) for i in util.valid_man_counts()][:n_end]
  return [dataset.DataReader(p) for p in paths]


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

  sample_to_device = lambda x: tuple(map(lambda t: t.to(cfg.device, non_blocking=True), dataset.post_process(x)))

  M = model.NNUE().to(cfg.device)

  valid_man_counts = util.valid_man_counts()

  sess = seer_train.Session(cfg.root_path).set_concurrency(cfg.concurrency)

  if (path.exists(cfg.model_save_path)):
    print('Loading model ... ')
    M.load_state_dict(torch.load(cfg.model_save_path))
    sess.load_weights(cfg.bin_model_save_path)


  writer = SummaryWriter(cfg.visual_directory)

  opt = optim.Adadelta(M.parameters(), lr=cfg.learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, 1, gamma=0.95)

  queue = []
  
  total_steps = 0

  if not args.all:
    print('generating data')

    for men in valid_man_counts:
    
      if men < args.resume:
        n_path = sess.get_n_man_train_path(men)
        assert path.exists(n_path)

        total_steps += seer_train.SampleReader(n_path).size() // cfg.batch_size
        print(f'skipping {men} man positions')
        continue
      else:
        print(f'training on {men} man positions')

      sess.maybe_generate_links_for(men)
      train_data = dataset.SeerData(dataset.DataReader(sess.get_n_man_train_path(men)), cfg)
  
      print(f'loaded {men} man positions')

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
          M.to_binary_file(cfg.bin_model_save_path)
          torch.save(M.state_dict(), cfg.model_save_path)

        train_step(M, sample_to_device(sample), opt, queue, max_queue_size=cfg.max_queue_size, report=(0 == i % cfg.report_rate))
        total_steps += 1

      scheduler.step()
      sess.load_weights(cfg.bin_model_save_path)
  
  else:
    print('training on all positions')

    reader = dataset.StochasticMultiplexReader(all_sample_readers(sess))
    print(reader.name())
    train_data = dataset.SeerData(reader, cfg)
    
    train_data_loader = torch.utils.data.DataLoader(train_data,
      batch_size=cfg.batch_size, 
      num_workers=cfg.concurrency,
      pin_memory=True,
      worker_init_fn=dataset.worker_init_fn)

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
