import seer_train

def p_pawn_position(i):
  return i % (2 * 64)

def piece_position(i):
  return i % (12 * 64)

def material(i):
  return (i % (12 * 64)) // 64

def king_position_times_material(i):
  return i // 64