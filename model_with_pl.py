import setuptools
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from dataset import charDataset, tokenize, logger
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.mobile_optimizer import optimize_for_mobile
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import pytorch_lightning as pl
import argparse
import torch
import os

RANDOM_SEED = 7777

class EmotionClassifier(pl.LightningModule):
    def __init__(self,
                 emb_dim: int = 512,
                 hidden_dim: int = 256,
                 batch_size: int = 32,
                 lr: float = 0.001,
                 gamma: float = 0.9,
                 patience: int = 5,
                 cnn_first_kernel: int = 10,
                 cnn_second_kernel: int = 4,
                 cnn_first_output_channel: int = 256,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super().__init__()
        self.input_dim = 415
        self.num_labels = 7
        self.vocab_size = 150
        self.pad_token_ids = 0

        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.patience = patience

        self.train_set = None
        self.val_set = None
        self.num_workers = os.cpu_count() if str(self.device) == "cpu" else torch.cuda.device_count()
        self.num_workers = self.num_workers if str(self.device) == "cpu" or self.num_workers <= 2 else 2
        conv_layer_output_dim = (((emb_dim - cnn_first_kernel + 1) // 2) - cnn_second_kernel + 1) // 2
        pl.seed_everything(RANDOM_SEED)

        self.embeddingLayer = torch.nn.Sequential(
            torch.nn.Embedding(self.vocab_size, emb_dim, padding_idx=self.pad_token_ids, device=self.device),
            torch.nn.LayerNorm(emb_dim, device=self.device)
        )
        self.convLayer = torch.nn.Sequential(
            torch.nn.Conv1d(self.input_dim, cnn_first_output_channel, (cnn_first_kernel, ), device=self.device),
            torch.nn.MaxPool1d(2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(cnn_first_output_channel, 1, (cnn_second_kernel, ), device=self.device),
            torch.nn.MaxPool1d(2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.LayerNorm(conv_layer_output_dim, device=self.device),
            torch.nn.Dropout(dropout_rate)
        )
        self.classificationHead = torch.nn.Sequential(
            torch.nn.Linear(conv_layer_output_dim, hidden_dim, device=self.device),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim//2, device=self.device),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim//2, self.num_labels, device=self.device),
            torch.nn.Softmax(dim=1)
        )

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.gamma)
        return [optim], [lr_scheduler]

    def configure_callbacks(self):
        model_ckp = ModelCheckpoint(dirpath=f"./model/CNN/model_ckp/", filename='{epoch:02d}_{loss:.2f}', verbose=True,
                                    save_last=True, monitor='val_loss', mode='min')
        early_stoppint = EarlyStopping(monitor="val_loss", mode="min", patience=self.patience)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        return [model_ckp, early_stoppint, lr_monitor]

    def loss(self, pred, y):
        return torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_ids)(pred, y)

    def accuracy(self, pred, y):
        return Accuracy(num_classes=self.num_labels, ignore_index=self.pad_token_ids).to(self.device)(pred, y)

    def forward(self, text: torch.LongTensor) -> torch.FloatTensor:
        embedded = self.embeddingLayer(text)
        conv = self.convLayer(embedded)
        output = self.classificationHead(conv)
        return output

    def training_step(self, batch, batch_idx):
        text, labels = batch
        pred = self(text)
        loss = self.loss(pred, labels)
        acc = self.accuracy(pred, labels)
        self.log_dict({"loss": loss, "acc": acc}, prog_bar=True)
        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        text, labels = batch
        pred = self(text)
        val_loss = self.loss(pred, labels)
        val_acc = self.accuracy(pred, labels)
        self.log_dict({"val_loss": val_loss, "val_acc": val_acc}, prog_bar=True)
        return {'val_loss': val_loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        avg_valLoss = torch.mean(torch.stack([output["val_loss"] for output in outputs]), dim=0)
        avg_valAcc = torch.mean(torch.stack([output["val_acc"] for output in outputs]), dim=0)

        self.log_dict({"avg_val_loss": avg_valLoss, "avg_val_acc": avg_valAcc})
        logger.info(f"\nEpoch {self.current_epoch} - avg_val_loss : {avg_valLoss:.4f}, avg_val_acc : {avg_valAcc:.2f}")

    def prepare_data(self):
        train, val = charDataset.prepare_data()
        self.train_set = charDataset(train["data"], train["label"], self.device)
        self.val_set = charDataset(val["data"], val["label"], self.device)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", type=int, default=50, dest="epochs", help="epochs")
    parser.add_argument("-batch_size", type=int, default=32, dest="batch_size", help="batch_size")
    parser.add_argument("-lr", type=int, default=0.001, dest="lr", help="learning rate")
    parser.add_argument("-embedding_size", type=int, default=512, dest="emb_dim", help="size of embedding layer")
    parser.add_argument("-hidden_size", type=int, default=256, dest="hidden_dim", help="size of hidden layer")
    parser.add_argument("-gamma", type=int, default=0.9, dest="gamma", help="rate of multiplied with lr for each epoch")
    parser.add_argument("-patience", type=int, default=5, dest="patience", help="num of times monitoring metric can be reduced")
    parser.add_argument("-dropout_rate", type=int, default=0.1, dest="dropout_rate", help="rate of dropout")
    parser.add_argument("-cnn_first_kernel", type=int, default=10, dest="cnn_first_kernel", help="kernel size in first CNN")
    parser.add_argument("-cnn_second_kernel", type=int, default=4, dest="cnn_second_kernel", help="kernel size in second CNN")
    parser.add_argument("-cnn_first_output_channel", type=int, default=256, dest="cnn_first_output_channel", help="output channel at first CNN")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmotionClassifier(**args.__dict__)
    trainer = Trainer(max_epochs=args.epochs, gpus=torch.cuda.device_count(),
                      logger=TensorBoardLogger("./model/CNN/tensorboardLog/"))
    trainer.fit(model)
    torch.save(model.state_dict(), "model/CNN/emotion_classifier_state.pt")

    example_input = tokenize("이건 트레이싱을 위한 예시 입력입니다.", device)
    model = torch.quantization.convert(model)
    model = torch.jit.trace(model, example_input, strict=False)
    opt_model = optimize_for_mobile(model)
    opt_model._save_for_lite_interpreter("./model/CNN/emotion_classifier.ptl")
