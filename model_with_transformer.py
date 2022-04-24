import setuptools
import os
import wandb
import torch
import argparse
from torch.functional import F
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from dataset import charDataset, tokenize, logger
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.mobile_optimizer import optimize_for_mobile

MAX_LEN = 415  # train - 415, val - 364, test - 289
VOCAB_SIZE = 150
RANDOM_SEED = 7777
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class EmotionClassifier(torch.nn.Module):
    def __init__(self,
                 embedding_size: int = 512,
                 hidden_size: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super(EmotionClassifier, self).__init__()
        self.num_labels = 7
        self.pad_token_ids = 0
        self.input_dim = MAX_LEN
        self.vocab_size = VOCAB_SIZE
        self.device = DEVICE
        pass

    def forward(self, x):
        pass


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')  # for using GPU
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    wandb.login()
    wandb.init(project="E_motional")
    wandb.run.name = wandb.run.id

    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", type=int, default=50, dest="epochs", help="epochs")
    parser.add_argument("-batch_size", type=int, default=32, dest="batch_size", help="batch_size")
    parser.add_argument("-lr", type=int, default=1e-4, dest="lr", help="learning rate")
    parser.add_argument("-embedding-size", type=int, default=512, dest="embedding_size", help="size of embedding layer")
    parser.add_argument("-hidden_size", type=int, default=256, dest="hidden_size", help="size of hidden layer")
    parser.add_argument("-gamma", type=int, default=0.9, dest="gamma", help="rate of multiplied with lr for each epoch")
    parser.add_argument("-patience", type=int, default=5, dest="patience", help="num of times monitoring metric can be reduced")
    parser.add_argument("-dropout-rate", type=int, default=0.1, dest="dropout_rate", help="rate of dropout")
    parser.add_argument("-num-heads", type=int, default=8, dest="num_heads", help="num of attention heads")
    parser.add_argument("-num-layers", type=int, default=6, dest="num_layers", help="num of attention layers")
    num_worker = os.cpu_count() - 1 if DEVICE == "cpu" else torch.cuda.device_count()
    num_worker = num_worker if DEVICE == "cpu" or num_worker <= 2 else 2
    pad_token_id = 0
    num_labels = 7

    # define model, optim, lr_scehduler, accuracy
    args = parser.parse_args()
    wandb.config.update(args)
    model = EmotionClassifier(**args.__dict__)
    wandb.watch(model)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = ExponentialLR(optim, gamma=args.gamma)
    accuracy = Accuracy(num_classes=num_labels, ignore_index=pad_token_id).to(DEVICE)
    # make dataloader
    train, val = charDataset.prepare_data()
    train_set = DataLoader(charDataset(train["data"].to_list(), train["label"].to_list(), DEVICE),
                           batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=num_worker)
    val_set = DataLoader(charDataset(val["data"].to_list(), val["label"].to_list(), DEVICE),
                         batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=num_worker)

    last_metric = 0
    best_metric = 0
    patience_cnt = 0
    logger.info(model)
    logger.info("start Training.")
    for i in range(args.epochs):
        # train step
        for j, (train_x, train_Y) in enumerate(train_set):
            optim.zero_grad()
            pred = model(train_x).logits
            loss = F.cross_entropy(pred, train_Y, ignore_index=pad_token_id)
            acc = accuracy(torch.argmax(pred, dim=1), train_Y)

            if j % 10 == 0:
                logger.info(f"Epoch {i} - loss : %.4f, acc : %.2f | progress : {j}/{len(train_set)}" % (float(loss), acc))
                wandb.log({"loss": float(loss), "acc": acc})
                for name, param in model.named_parameters():
                    try:
                        wandb.Histogram(name, param.to("cpu"))
                    except RuntimeError:
                        pass
            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
            loss.backward()
            optim.step()

        # validation step
        loss_list = []
        acc_list = []
        for j, (val_x, val_Y) in enumerate(val_set):
            with torch.no_grad():
                pred = model(val_x).logits
                loss = F.cross_entropy(pred, val_Y, ignore_index=pad_token_id)
                acc = accuracy(torch.argmax(pred, dim=1), val_Y)

                loss_list.append(loss)
                acc_list.append(acc)
                if j % 10 == 0:
                    logger.info(f"Epoch {i} - val_loss : %.4f, val_acc : %.2f | progress : {j}/{len(val_set)}" % (float(loss), acc))
                    wandb.log({"val_loss": float(loss), "val_acc": acc})
        logger.info(f"Epoch {i} - avg_val_loss : %.4f, avg_val_acc : %.2f" % (sum(loss_list)/len(loss_list), sum(acc_list)/len(acc_list)))
        wandb.log({"avg_val_loss": sum(loss_list)/len(loss_list), "avg_val_acc": sum(acc_list)/len(acc_list)})

        # Ealry Stopping
        avg_val_acc = sum(acc_list)/len(acc_list)
        if last_metric > avg_val_acc:
            patience_cnt += 1
            if patience_cnt > args.patience:
                logger.info(f"metrics was not improved at {args.patience} times. stop training.")
                break
        else:
            logger.info("metrics was improved.")
            # ModelCheckpoint(SaveBestOnly)
            if best_metric < avg_val_acc:
                best_metric = avg_val_acc
                logger.info(f"avg_val_acc has achived to best({avg_val_acc}). save model state.")
                torch.save(model.state_dict(), "./model/Transformer/best_state.pt")
            patience_cnt = 0
    torch.save(model.state_dict(), "./model/Transformer/emotion_classifier_state.pt")

    example_input = tokenize("이건 트레이싱을 위한 예시 입력입니다.", DEVICE)
    model = torch.quantization.convert(model)
    model = torch.jit.trace(model, example_input, strict=False)
    opt_model = optimize_for_mobile(model)
    opt_model._save_for_lite_interpreter("./model/Transformer/emotion_classifier.ptl")
