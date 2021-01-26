import chess
import chess.syzygy

import config
import util
import seer_train

def fetch_wdl(tb, position):
  if len(position.piece_map()) == 2:
    return seer_train.known_draw_value()
  else:
    res = tb.get_wdl(position)
    
    if res is None:
      return None
    if res == 0:
      return seer_train.known_draw_value()
    if res > 0:
      return seer_train.known_win_value()
    if res < 0:
      return seer_train.known_loss_value()


def generate_base_dataset():
  cfg = config.Config('config.yaml')
  
  tb = chess.syzygy.Tablebase()
  tb.add_directory(cfg.tb_path, load_wdl=True, load_dtz=False)

  session = seer_train.Session(cfg.root_path)

  for i in util.base_man_counts(cfg.tb_cardinality):
    destination = seer_train.SampleWriter(session.get_n_man_train_path(i))
    for state in seer_train.RawFenReader(session.get_n_man_raw_path(i)):
      wdl = fetch_wdl(tb, chess.Board(state.fen()))
      if wdl is not None:
        destination.append_sample(seer_train.Sample(state, wdl))
      else:
        print('could not fetch: {}'.format(state.fen()))


if __name__ == '__main__':
  generate_base_dataset()