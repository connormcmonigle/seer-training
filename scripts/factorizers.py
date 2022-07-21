import seer_train


def piece_position(feature_idx):
    return feature_idx % (12 * 64)


def material(feature_idx):
    return (feature_idx % (12 * 64)) // 64


def king_position_times_material(feature_idx):
    return feature_idx // 64
