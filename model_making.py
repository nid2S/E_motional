import setuptools
import torch
import pandas as pd
import argparse
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.mobile_optimizer import optimize_for_mobile

class EmotionClassifier(LightningModule):
    def __init__(self, hparams):
        super(EmotionClassifier, self).__init__()

        self.RANDOM_SEED = 7777
        pl.seed_everything(self.RANDOM_SEED)

    def configure_optimizers(self):
        pass

    def configure_callbacks(self):
        pass

    def forward(self, x):
        pass

    def loss(self, output, labels):
        return torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)(output, labels)

    def accuracy(self, output, labels):
        output = torch.argmax(output, dim=1)
        return torch.sum(output == labels) / output.__len__() * 100  # %(Precentage)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        accuracy = self.accuracy(y_pred, y)

        self.log('loss', loss)
        self.log('acc', accuracy, prog_bar=True)
        return {'loss': loss, 'acc': accuracy}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        accuracy = self.accuracy(y_pred, y)

        self.log_dict({'val_loss': loss, 'val_acc': accuracy}, prog_bar=True)
        return {'val_loss': loss, 'val_acc': accuracy}

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        mean_acc = torch.stack([output['val_acc'] for output in outputs]).mean()

        self.log_dict({'avg_val_loss': mean_loss, 'avg_val_acc': mean_acc}, on_epoch=True, prog_bar=True)
        return {'avg_val_loss': mean_loss, 'avg_val_acc': mean_acc}

    def predict(self, x):
        return self(x)

    def tokenize(self, sent: str, max_length: int):
        pass


parser = argparse.ArgumentParser()
parser.add_argument("", type=int, default=None, dest="", hint="")

args = parser.parse_args()
model = EmotionClassifier(args)
trainer = Trainer(max_epochs=args.epochs, gpus=torch.cuda.device_count(), logger=TensorBoardLogger("./model/tensorboardLog/"))
trainer.fit(model)
torch.save(model.state_dict(), "./model/emotion_classifier.pt")

example_input = model.encode("이건 트레이싱을 위한 예시 입력입니다.", max_length=None)
model = torch.quantization.convert(model)
model = torch.jit.trace(model, example_input, strict=False)
opt_model = optimize_for_mobile(model)
opt_model._save_for_lite_interpreter("./model/emotion_classifier.ptl")
