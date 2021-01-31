import seer_train

def piece_position(i):
  return i % (12 * 64)

def material(i):
  return (i % (12 * 64)) // 64

def king_position_and_material(i):
  return i // 64