import torch
from torch import nn
from dnc import DNC

class SequentialDNC(nn.Module):
	def __init__(self, states, actions, training = False):
		super(SequentialDNC, self).__init__()

		self.training = training

		self._dnc1 = DNC(states, 65, 65, 16, 32, 1, 3),
		self._dnc2 = DNC(states, 65, 65, 16, 32, 1, 3),
		self._dnc3 = DNC(states, 65, 64, 16, 32, 1, 3),

		self._dnc4 = DNC(states, 65, 65, 16, 32, 1, 3),
		self._dnc5 = DNC(states, 65, 65, 16, 32, 1, 3),
		self._dnc6 = DNC(states, 65, 64, 16, 32, 1, 3),

		self._dnc7 = DNC(states, 65, 65, 16, 32, 1, 3),
		self._dnc8 = DNC(states, 65, 65, 16, 32, 1, 3),
		self._dnc9 = DNC(states, 65, 4, 16, 32, 1, 3),

		# I do not, for the life of
		# me, understand why these
		# get transformed into tuples.
		self._dnc1 = self._dnc1[0]
		self._dnc2 = self._dnc2[0]
		self._dnc3 = self._dnc3[0]
		self._dnc4 = self._dnc4[0]
		self._dnc5 = self._dnc5[0]
		self._dnc6 = self._dnc6[0]
		self._dnc7 = self._dnc7[0]
		self._dnc8 = self._dnc8[0]
		self._dnc9 = self._dnc9[0]

		self._softmax = nn.Softmax(dim=1)

		self.reset()

	def forward(self, x):
		logits = x

		logits, self._states[0] = self._dnc1(logits, self._states[0])
		logits, self._states[1] = self._dnc2(logits, self._states[1])
		from_squares, self._states[2] = self._dnc3(logits, self._states[2])
		
		logits = x
		logits, self._states[3] = self._dnc4(logits, self._states[3])
		logits, self._states[4] = self._dnc5(logits, self._states[4])
		to_squares, self._states[5] = self._dnc6(logits, self._states[5])
		
		logits = x
		logits, self._states[6] = self._dnc7(logits, self._states[6])
		logits, self._states[7] = self._dnc8(logits, self._states[7])
		promotions, self._states[8] = self._dnc9(logits, self._states[8])

		if self.training:
			self._states, self._states2 = self._states2, self._states

		return torch.cat((
			self._softmax(from_squares),
			self._softmax(to_squares),
			self._softmax(promotions),
		), dim=1)

	def reset(self):
		self._states = [
			self._dnc1.reset(),
			self._dnc2.reset(),
			self._dnc3.reset(),
			self._dnc4.reset(),
			self._dnc5.reset(),
			self._dnc6.reset(),
			self._dnc7.reset(),
			self._dnc8.reset(),
			self._dnc9.reset(),
		]

		if self.training:
			self._states2 = [
				self._dnc1.reset(),
				self._dnc2.reset(),
				self._dnc3.reset(),
				self._dnc4.reset(),
				self._dnc5.reset(),
				self._dnc6.reset(),
				self._dnc7.reset(),
				self._dnc8.reset(),
				self._dnc9.reset(),
			]