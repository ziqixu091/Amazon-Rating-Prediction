import torch
from transformers import BertTokenizer
import pathlib as pl
import os
os.chdir('/new-stg/home/banghua/Amazon-Rating-Prediction')
from tqdm import tqdm
import pickle
import random
random.seed(42)
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from balanced_loss import Loss
from datasets import load_metric, load_from_disk
import numpy as np


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(torch.int64)
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = focal_loss(logits, labels)
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss_fct, outputs) if return_outputs else loss_fct

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name())
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    current_path = pl.Path.cwd()

    dataset_train = load_from_disk(current_path / 'dataset_huggingface_full' / 'train_dataset')
    dataset_valid = load_from_disk(current_path / 'dataset_huggingface_full' / 'valid_dataset')
    dataset_test = load_from_disk(current_path / 'dataset_huggingface_full' / 'test_dataset')

    samples_per_class = [0] * 5
    for entry in dataset_train["labels"]:
        samples_per_class[entry] += 1

    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('LiYuan/amazon-review-sentiment-analysis')
    # 17,280 rows of training set 4,320 rows of dev set. test set: 2,400 rows.

    model = AutoModelForSequenceClassification.from_pretrained("LiYuan/amazon-review-sentiment-analysis", num_labels = 5)
    model.cuda()

    metric_name = "accuracy"
    print(metric_name)
    model_name = "Amazon-Pet-BERT"
    print(model_name)


    focal_loss = Loss(
        loss_type="focal_loss",
        samples_per_class=samples_per_class,
        class_balanced=True
    )

    batch_size = 48
    args = TrainingArguments(
        f"{model_name}-finetuned",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        logging_steps=100,
    )

    actual_task = "mnli"
    metric = load_metric('glue', actual_task)

    trainer = CustomTrainer(
        model,
        args,
        train_dataset=dataset_test,
        eval_dataset=dataset_valid,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    # Save model
    trainer.save_model(current_path / 'best_model')

    # Evaluate
    trainer.evaluate(dataset_train)
    trainer.evaluate(dataset_valid)
    trainer.evaluate(dataset_test)

    # Save preds
    preds_train, labels_train, metrics_train = trainer.predict(dataset_train)
    preds_valid, labels_valid, metrics_valid = trainer.predict(dataset_valid)
    preds_test, labels_test, metrics_test = trainer.predict(dataset_test)

    train_output_obj = {
        "preds": preds_train,
        "labels": labels_train,
        "metrics": metrics_train
    }

    valid_output_obj = {
        "preds": preds_valid,
        "labels": labels_valid,
        "metrics": metrics_valid
    }

    test_output_obj = {
        "preds": preds_test,
        "labels": labels_test,
        "metrics": metrics_test
    }

    with open(current_path / 'best_model' / 'train_output_obj.pkl', 'wb') as f:
        pickle.dump(train_output_obj, f)

    with open(current_path / 'best_model' / 'valid_output_obj.pkl', 'wb') as f:
        pickle.dump(valid_output_obj, f)

    with open(current_path / 'best_model' / 'test_output_obj.pkl', 'wb') as f:
        pickle.dump(test_output_obj, f)
