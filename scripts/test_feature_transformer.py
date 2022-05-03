import torch
import seer_train
import dataset
import model

fft = model.FrozenFeatureTransformer(base_dim=384)


state = seer_train.StateType.parse_fen(input('fen: '))
sample = seer_train.Sample(state, 0)
tensors = [t.unsqueeze(0) for t in dataset.sample_to_tensor(sample)]
pov, w, b, _, _ = dataset.post_process(tensors)

print((fft(w) * 512).short())
print((fft(b) * 512).short())