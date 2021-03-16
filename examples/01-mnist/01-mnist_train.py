# wget www.di.ens.fr/~lelarge/MNIST.tar.gz
# tar -zxvf MNIST.tar.gz
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


class LitMNIST(pl.LightningModule):

	def __init__(self):
		super().__init__()

		# mnist images are (1, 28, 28) (channels, width, height)
		self.layer_1 = torch.nn.Linear(28 * 28, 128)
		self.layer_2 = torch.nn.Linear(128, 256)
		self.layer_3 = torch.nn.Linear(256, 10)

	def forward(self, x):
		batch_size, _, _, _ = x.size()

		x = x.view(batch_size, -1)

		x = self.layer_1(x)
		x = torch.relu(x)

		x = self.layer_2(x)
		x = torch.relu(x)

		x = self.layer_3(x)
		x = torch.log_softmax(x, dim=1)

		return x

	def train_dataloader(self):
		transform=transforms.Compose([
								transforms.ToTensor(),
								transforms.Normalize((0.1307,), (0.3081,))
							])
		
		mnist_train = MNIST(os.getcwd(), train=True, download=False,
						transform=transform)
		
		return DataLoader(mnist_train, batch_size=64)

	# def prepare_data(self):
	# 	# download only
	# 	MNIST(os.getcwd(), train=True, download=False)

	def configure_optimizers(self):
		return Adam(self.parameters(), lr=1e-3)

	def training_step(self, batch, batch_idx):
		x, y = batch
		logits = self(x)
		loss = F.nll_loss(logits, y)
		# self.logger.log_metrics({'loss': loss}, )
		self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return {'loss': loss} # return loss (also works)

if __name__ == "__main__":
	model = LitMNIST()
	if torch.cuda.is_available():
		trainer = Trainer(max_epochs=10, gpus=1)
	else:
		trainer = Trainer(max_epochs=10)
	trainer.fit(model)