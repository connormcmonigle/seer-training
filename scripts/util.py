import chess

def valid_man_counts():
  return range(2, 32+1)

def startpos_man_count():
  return len(chess.Board().piece_map())