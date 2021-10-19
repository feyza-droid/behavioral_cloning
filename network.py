import torch

class MyNetwork(torch.nn.Module):
	def __init__(self, input_size, output_size):
		super(MyNetwork, self).__init__()

		hidden_size = 128
		# defining fully-connected layers
		self.fc_1 = torch.nn.Linear(input_size, hidden_size)
		self.fc_2 = torch.nn.Linear(hidden_size, hidden_size)
		self.fc_3 = torch.nn.Linear(hidden_size, output_size)

	# get value for given state-action pair
	def forward(self, state):
		out = self.fc_1(state)
		out = torch.nn.functional.relu(out)
		out = self.fc_2(out)
		out = torch.nn.functional.relu(out)
		out = self.fc_3(out)
		return out
