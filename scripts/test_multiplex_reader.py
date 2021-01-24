import stochastic_multiplex_reader

class FakeReader:
  def __init__(self, c, n):
    self.c = c
    self.n = n

  def size(self):
    return self.n

  def __iter__(self):
    for i in range(self.n):
      yield str(self.c) + str(i)


s = stochastic_multiplex_reader.StochasticMultiplexReader([FakeReader('a', 4), FakeReader('b', 6), FakeReader('c', 5)])
s.set_subset(process_id=0, num_processes=3)

for elem in s:
  print(elem)