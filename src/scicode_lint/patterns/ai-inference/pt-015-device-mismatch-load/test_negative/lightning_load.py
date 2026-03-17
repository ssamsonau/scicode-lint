import pytorch_lightning as pl
import torch
import torch.nn as nn


class MyModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def load_trained_model(checkpoint_path, input_size=784, hidden_size=256, num_classes=10):
    model = MyModel.load_from_checkpoint(
        checkpoint_path, input_size=input_size, hidden_size=hidden_size, num_classes=num_classes
    )
    return model


def load_for_inference(checkpoint_path):
    model = MyModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def resume_training(checkpoint_path, datamodule):
    model = MyModel.load_from_checkpoint(checkpoint_path)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, datamodule)
    return model
