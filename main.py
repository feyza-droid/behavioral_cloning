import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from network import MyNetwork
from dataset_loader import MyDataset

def validate(device, model, criterion, val_loader, val_epoch_loss):
	val_batch_loss = []
	for states, actions in val_loader:
		states, actions = states.to(device), actions.to(device)

		outputs = model(states)
		loss = criterion(outputs, actions)
		val_batch_loss.append(loss.item())
	
	val_epoch_loss.append(np.mean(val_batch_loss))
	print("Validation Loss {}".format(np.mean(val_epoch_loss)))

	return val_epoch_loss

def plot(epoch_loss, val_epoch_loss, plot_name):
	fig, (a0, a1) = plt.subplots(1, 2)
	a0.plot(epoch_loss)

	a1.plot(val_epoch_loss)

	fig.savefig(plot_name)

def train(device, model, criterion, optimizer, train_loader, val_loader, model_name, plot_name):
	epoch_loss = []
	val_epoch_loss = []
	for epoch in range(16, 101):
		batch_loss = []

		for states, actions in train_loader:
			states, actions = states.to(device), actions.to(device)

			optimizer.zero_grad()

			outputs = model(states)
			loss = criterion(outputs, actions)
			loss.backward()
			optimizer.step()

			batch_loss.append(loss.item())
		epoch_loss.append(np.mean(batch_loss))
		print("Epoch {} Loss {}".format(epoch, np.mean(epoch_loss)))

		if epoch % 5 == 0:
			torch.save(model.state_dict(), model_name)  # torch.save(model, PATH)
			model.eval()
			val_epoch_loss = validate(device, model, criterion, val_loader, val_epoch_loss)
			plot(epoch_loss, val_epoch_loss, plot_name)
			model.train()
	print('Finished Training')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
is_load_model = True

dataset_name = "training_model"
model_name = "bc_" + dataset_name
plot_name = dataset_name + "_loss.pdf"

model = MyNetwork(41, 2)  # state and action shape
if is_load_model:
	model.load_state_dict(torch.load(model_name))
	print("{} loaded".format(model_name))
	
model.to(device)

criterion = torch.nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

dataset = MyDataset(dataset_name)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=5)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=5)

train(device, model, criterion, optimizer, train_loader, val_loader, model_name, plot_name)