import pytorch_lightning as pl
import torch.nn as nn
from transformers import AdamW, CanineForSequenceClassification


class CanineReviewClassifier(pl.LightningModule):
    def __init__(self, num_labels, id2label, label2id):
        super().__init__()
        self.model = CanineForSequenceClassification.from_pretrained(
            "google/canine-s",
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

        return outputs

    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits

        predictions = logits.argmax(-1)
        correct = (predictions == batch["labels"]).sum().item()
        accuracy = correct / batch["input_ids"].shape[0]

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)

        return loss

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        return AdamW(self.parameters(), lr=5e-5)

    # def train_dataloader(self):
    #     return train_dataloader

    # def val_dataloader(self):
    #     return test_dataloader
