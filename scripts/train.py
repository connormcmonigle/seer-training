from os import path
import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import config
import seer_train
import util
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
  cfg = config.Config('config.yaml')

  sample_to_device = lambda x: tuple(map(lambda t: t.to(cfg.device, non_blocking=True), x))

  M = model.NNUE().to(cfg.device)

  if (path.exists(cfg.model_save_path)):
    print('Loading model ... ')
    M.load_state_dict(torch.load(cfg.model_save_path))

  valid_man_counts = util.valid_man_counts()

  sess = seer_train.Session(cfg.root_path)

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

  for men in valid_man_counts:
    
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

if __name__ == '__main__':
  main()