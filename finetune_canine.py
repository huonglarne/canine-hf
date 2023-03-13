from datasets import load_dataset
from pytorch_lightning import Trainer

# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from transformers import CanineTokenizer

from model import CanineReviewClassifier

# load data set

train_ds, test_ds = load_dataset("imdb", split=["train[:2%]", "test[:2%]"])

labels = train_ds.features["label"].names

id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}

tokenizer = CanineTokenizer.from_pretrained("google/canine-s")

train_ds = train_ds.map(
    lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True),
    batched=True,
)
test_ds = test_ds.map(
    lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True),
    batched=True,
)

train_ds.set_format(
    type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"]
)
test_ds.set_format(
    type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"]
)

train_ds = train_ds.rename_column(
    original_column_name="label", new_column_name="labels"
)
test_ds = test_ds.rename_column(original_column_name="label", new_column_name="labels")

# Data loader
train_dataloader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=16)


model = CanineReviewClassifier(len(labels), id2label, label2id)

trainer = Trainer(gpus=1, callbacks=[EarlyStopping(monitor="validation_loss")])
trainer.fit(model, train_dataloader, test_dataloader)
