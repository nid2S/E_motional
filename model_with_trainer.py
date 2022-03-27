from torch.utils.mobile_optimizer import optimize_for_mobile
from transformers import MobileBertForSequenceClassification, MobileBertTokenizerFast
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from transformers.integrations import TensorBoardCallback
from transformers import PrinterCallback
import pandas as pd
import datetime
import torch
import os

MAX_LEN = 320  # train-311, val-275, test-209
PREMODEL_NAME = "google/mobilebert-uncased"
RANDOM_SEED = 7777

def accuracy(pred):
    labels = torch.from_numpy(pred.label_ids)
    output = torch.from_numpy(pred.predictions)

    output = torch.argmax(output, dim=1)
    output = torch.sum(output == labels) / output.__len__() * 100  # %(Precentage)
    return {'accuracy': output.item()}

def getDataset(isTrain: bool, using_device: str):
    # [{key: value}]
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    if isTrain:
        data_path = "./data/train.txt"
    else:
        data_path = "./data/val.txt"

    data = pd.read_csv(data_path, sep="\t", encoding="utf-8", index_col=0)
    encoded_data = []
    for _, (d, label) in data.iterrows():
        encoded_d = tokenizer(d, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt").to(using_device)
        encoded_dict = dict()
        for k, v in encoded_d.items():
            encoded_dict[k] = v[0].to(using_device)
        encoded_dict["labels"] = label
        encoded_data.append(encoded_dict)
    return encoded_data


device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 10
batch_size = 32

model = MobileBertForSequenceClassification.from_pretrained(PREMODEL_NAME, num_labels=7).to(device)
tokenizer = MobileBertTokenizerFast.from_pretrained(PREMODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

log_dir = os.path.join('./model/trainer/logs/', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
train_args = TrainingArguments(output_dir="./model/trainer/output_dir/",
                               logging_dir=log_dir,
                               do_train=True,
                               do_eval=True,
                               learning_rate=3e-5,
                               per_device_train_batch_size=batch_size,
                               per_device_eval_batch_size=batch_size,
                               num_train_epochs=epochs,
                               weight_decay=0.01,
                               load_best_model_at_end=True,
                               evaluation_strategy="epoch",
                               # save_strategy="epoch",  # it makes error in pyCharm, but it prevents error in colab
                               )

trainer = Trainer(model=model, args=train_args, data_collator=data_collator, compute_metrics=accuracy,
                  callbacks=[PrinterCallback(), TensorBoardCallback()],
                  train_dataset=getDataset(isTrain=True, using_device=device),
                  eval_dataset=getDataset(isTrain=False, using_device=device))
trainer.train()
torch.save(model, "./model/trainer/pytorch_model.bin")

# model = MobileBertForSequenceClassification.from_pretrained("./model/trainer")
# tokenizer = MobileBertTokenizerFast.from_pretrained(PREMODEL_NAME)

example_input = tokenizer.encode("이건 트레이싱을 위한 예시 입력입니다.", return_tensors="pt", max_length=MAX_LEN, padding="max_length", truncation=True)
model = torch.quantization.convert(model)
model = torch.jit.trace(model, example_input, strict=False)
opt_model = optimize_for_mobile(model)
opt_model._save_for_lite_interpreter("./model/emotion_classifier.ptl")
