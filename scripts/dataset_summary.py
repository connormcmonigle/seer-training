import os
import os.path

import config
import util
import seer_train

def position_count_n_man(root, n):
  with open(seer_train.raw_n_man_path(root, n), 'r') as f:
    r = sum(1 for line in f)

  train_path = seer_train.train_n_man_path(root, n)

  if not os.path.exists(train_path):
    return r, None

  with open(train_path, 'r') as f:
    t = sum(1 for line in f)
  
  return r, t

def print_summary():
  cfg = config.Config('config.yaml')
  print('computing file lengths...')


  sizes = []

  for i in util.valid_man_counts():
    r, t = position_count_n_man(cfg.root_path, i)
    sizes.append((i, r, t))
  
  total_raw = sum([x[1] for x in sizes])
  total_train = sum([x[2] for x in sizes if x[2] is not None])


  for n, size, t_size in sizes:
    t_str = 'not found' if t_size is None else 't({}) ~ {:0.2f}%'.format(t_size, (100 * t_size) / total_train)
    print('{}: r({}) ~ {:0.2f}%, {}'.format(n, size, (100 * size) / total_raw, t_str))

  print('total raw: {}'.format(total_raw))
  print('total train: {}'.format(total_train))

  base_total = sum([x[1] for x in sizes if x[0] <= cfg.tb_cardinality])
  print('base ({} man): {} ~ {:0.2f}%'.format(cfg.tb_cardinality, base_total, (100 * base_total) / total_raw))




  

if __name__ == '__main__':
  print_summary()