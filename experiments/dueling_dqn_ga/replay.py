import numpy as np

class Replay:
	def __init__(self, size=100000):
		self._replay   = [None] * size
		self._size     = size

		self._cur_size = 0
		self._index    = 0

	def append(self, memento):
		self._replay[self._index] = memento
		self._index    = (self._index + 1) % self._size
		self._cur_size = min(self._cur_size + 1, self._size)
		
	def sample(self, n):
		if self._cur_size == 0:
			return []
			
		idxs = np.random.choice(self._cur_size, n).tolist()
		return [self._replay[i] for i in idxs]