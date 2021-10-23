import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.model import MLP


class LightningModuleMLP(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        num_layers: int = 2,
        with_softmax: bool = True,
        learning_rate: float = 0.01
    ):
        super().__init__()

        self.model = MLP(
            input_dim,
            hidden_dim,
            output_dim,
            dropout,
            num_layers,
            with_softmax
        )

        self.criterion = nn.CrossEntropyLoss()
        self.lr = learning_rate

        self.training_losses = []
        self.validation_losses = []
        self.test_losses = []

    def forward(self, x):
        return self.model(x)

    def calculate_and_log_loss(
        self,
        out: torch.tensor,
        y: torch.tensor,
        loss_storage: list,
        log_name: str
    ):
        loss = self.criterion(out, y)
        loss_storage.append(loss.item())
        self.log(log_name, loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return {'optimizer': optimizer}

    def training_step(self, train_batch, train_batch_idx):
        x, y = train_batch
        out = self.forward(x)
        loss = self.calculate_and_log_loss(out, y, self.training_losses, "train_loss")

        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs):
        avg_train_loss = torch.mean(torch.tensor(self.training_losses))
        self.logger.experiment.add_scalar(
            'avg_training_loss',
            avg_train_loss,
            global_step=self.current_epoch
        )
        self.training_losses = []

    def validation_step(self, val_batch, val_batch_idx):
        x, y = val_batch
        out = self.forward(x)
        loss = self.calculate_and_log_loss(out, y, self.validation_losses, "val_loss")

        return {'loss': loss}

    def validation_epoch_end(self, validation_step_outputs):
        avg_val_loss = torch.mean(torch.tensor(self.validation_losses))
        self.logger.experiment.add_scalar(
            'avg_validation_loss',
            avg_val_loss,
            global_step=self.current_epoch
        )
        self.validation_losses = []

    def test_step(self, test_batch, test_batch_idx):
        x, y = test_batch
        out = self.forward(x)
        loss = self.calculate_and_log_loss(out, y, self.test_losses, "test_loss")

        return {'loss': loss}

    def test_epoch_end(self, test_step_outputs):
        avg_test_loss = torch.mean(torch.tensor(self.test_losses))
        self.logger.experiment.add_scalar(
            'avg_test_loss',
            avg_test_loss,
            global_step=self.current_epoch
        )
        self.test_losses = []