import os
import chess
import chess.pgn
import tqdm

import seer_train
import util
import config


def get_handlers(root_path):
  return {i : open(seer_train.raw_n_man_path(root_path, i), 'w') for i in util.valid_man_counts()}

def position_generator(pgn_src, total_tgt):
  total = 0

  game = chess.pgn.read_game(pgn_src)
  while (total < total_tgt) and game is not None:
    count = util.startpos_man_count()
    board = game.board()

    for mv in game.mainline_moves():
      count -= int(board.is_capture(mv))
      board.push(mv)
      
      yield count, board
      
      total += 1
      if total >= total_tgt:
        break

    game = chess.pgn.read_game(pgn_src)
    


def main():
  cfg = config.Config('config.yaml')
  tgt = cfg.base_position_count_target
  handlers = get_handlers(cfg.root_path)
  pgn_src = open(cfg.pgn_src)

  for i, bd in tqdm.tqdm(position_generator(pgn_src, tgt), total=tgt):
    handlers[i].write(bd.fen() + '\n')


if __name__ == '__main__':
  main()
