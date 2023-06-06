import os
from typing import List
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import ParameterGrid

class BERTSentimentClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, data, text_column, label_column):
        return data.map(lambda example: {'text': example[text_column], 'label': example[label_column]})

    def tokenize_dataset(self, dataset, max_length=512):
        tokenized_dataset = dataset.map(lambda x: self.tokenizer(x['text'], truncation=True, padding='max_length', max_length=max_length), batched=True)
        return tokenized_dataset

    def create_training_arguments(self, output_dir, num_train_epochs=3, logging_steps=100, **kwargs):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=logging_steps,
            **kwargs
        )
        return training_args

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    def train(self, train_data, val_data, output_dir, text_column='text', label_column='label', num_train_epochs=3, logging_steps=100, **kwargs):
        preprocessed_train_data = self.preprocess_data(train_data, text_column, label_column)
        preprocessed_val_data = self.preprocess_data(val_data, text_column, label_column)
        tokenized_train_dataset = self.tokenize_dataset(preprocessed_train_data)
        tokenized_val_dataset = self.tokenize_dataset(preprocessed_val_data)
        training_args = self.create_training_arguments(output_dir, num_train_epochs, logging_steps, **kwargs)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()

    def predict(self, text: str) -> int:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        return logits.argmax(dim=1).item()

    def predict_batch(self, texts: List[str]) -> List[int]:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs) 
        logits = outputs.logits
        return logits.argmax(dim=1).tolist()

def save_pretrained(self, output_dir):
    self.tokenizer.save_pretrained(output_dir)
    self.model.save_pretrained(output_dir)

def load_pretrained(self, input_dir):
    self.tokenizer = AutoTokenizer.from_pretrained(input_dir)
    self.model = AutoModelForSequenceClassification.from_pretrained(input_dir)

def hyperparameter_search(self, train_data, val_data, output_dir, text_column='text', label_column='label', param_grid=None, **kwargs):
    if param_grid is None:
        param_grid = {
            'num_train_epochs': [2, 3, 4],
            'learning_rate': [2e-5, 3e-5, 5e-5],
            'per_device_train_batch_size': [8, 16],
        }
    best_score = -1
    best_params = None

    for params in ParameterGrid(param_grid):
        print(f"Training with params: {params}")
        self.train(train_data, val_data, output_dir, text_column, label_column, **params, **kwargs)
        metrics = self.evaluate(val_data, text_column, label_column)
        score = metrics['f1']

        if score > best_score:
            best_score = score
            best_params = params
            self.save_pretrained(os.path.join(output_dir, 'best_model'))

    print(f"Best score: {best_score}")
    print(f"Best params: {best_params}")

def evaluate(self, val_data, text_column='text', label_column='label'):
    preprocessed_val_data = self.preprocess_data(val_data, text_column, label_column)
    tokenized_val_dataset = self.tokenize_dataset(preprocessed_val_data)
    trainer = Trainer(
        model=self.model,
        compute_metrics=self.compute_metrics,
    )
    eval_result = trainer.evaluate(tokenized_val_dataset)
    return eval_result

