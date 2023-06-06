import os
import json
import numpy as np
import pandas as pd
from datasets import Dataset
import logging
import threading
from sklearn import pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, roc_auc_score
import joblib
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from unittest import TestCase, main
from glob import glob
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments,GPT2LMHeadModel, GPT2Tokenizer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tensorflow import keras
from websearch import WebSearcher
from ml_learning_classes import data_collector, data_transformer, model_trainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.info("Collecting data...")
logging.info("Training model...")
logging.info("Evaluating model...")

class DataCollector:
    def __init__(self):
        self.gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.web_searcher = WebSearcher()  # Create an instance of WebSearcher
        self.topics = ["python", "medicine", "mathematics"]  # r


    def generate_data_from_models(self):
        for topic in self.topics:
            # Generate text with GPT-2
            input_context = f"The past year in {topic}"
            input_ids = self.gpt_tokenizer.encode(input_context, return_tensors='pt')
            output = self.gpt_model.generate(input_ids, max_length=1000, temperature=0.7, num_return_sequences=1)
            generated_text = self.gpt_tokenizer.decode(output[0], skip_special_tokens=True)

            # Save the generated text to a file
            with open(f"{topic}.txt", "a") as f:
                f.write(generated_text + "\n")

    def scrape_data_from_web(self):
        best_match_urls = self.web_searcher.search_and_train(self.topics)
        
        for topic, url in best_match_urls.items():
            if url is not None:
                text = self.web_searcher.download_data([url])[0]

                # Save the extracted text to a file
                with open(f"{topic}.txt", "a") as f:
                    f.write(text + "\n")

class DataTransformer:
    @staticmethod
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
    def convert_to_csv(data_directory, csv_directory, delimiter=' ', header=None):
        if not os.path.isdir(data_directory) or not os.path.isdir(csv_directory):
            logging.error("One or both directories do not exist.")
            return
        
        data_files = glob(os.path.join(data_directory, '*.data'))
        for data_file in data_files:
            try:
                df = pd.read_csv(data_file, delimiter=delimiter, header=header)
                csv_file_path = os.path.join(csv_directory, os.path.basename(data_file).replace('.data', '.csv'))
                df.to_csv(csv_file_path, index=False)
                logging.info(f"File converted: {csv_file_path}")
            except Exception as e:
                logging.error(f"An error occurred during file conversion: {e}")

    def transform(self, data):
        # Apply transformations as needed, e.g., sentiment analysis
        sentiment = self.sentiment_analyzer(data["text"])

class ModelTrainer:
    def __init__(self, config_path, model_type="traditional", active_learning_strategy='uncertainty_sampling'):
        self.config = self.load_config(config_path)
        self.data_collector = DataCollector()
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.models = []
        self.model_type = self.config.get('model_type', "traditional")

    def tune_hyperparameters(self, model, X, y):
        # Define hyperparameter grid
        param_grid = {'parameter_name': [values]}

        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X, y)

        return grid_search.best_params_
    
    def train(self, X, y):
        # ...
        # Tune hyperparameters
        best_params = self.tune_hyperparameters(model, X, y)


    def split_data(self, test_size=None, random_state=None):
        test_size = self.config.get('test_size', 0.3)
        random_state = self.config.get('random_state', 42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        self.train_dataset = Dataset.from_pandas(pd.concat([self.X_train, self.y_train], axis=1))
        self.val_dataset = Dataset.from_pandas(pd.concat([self.X_test, self.y_test], axis=1))

    def train_and_update(self):
        num_models = self.config.get('num_models', 1)
        model_params = self.config.get('model_params', {})

        if self.model_type == "traditional":
            model_class = self.config.get('model_class', RandomForestClassifier)
            self.train_traditional_model(model_class, num_models, **model_params)
        elif self.model_type == "neural_network":
            self.train_neural_network(num_models)
        elif self.model_type == "transformer":
            self.train_transformer_model()
        elif self.model_type == "ucl":
            self.train_ucl_model()
        elif self.model_type == "gpt2":
            self.train_gpt2_model()
        else:
            logging.error(f"Invalid model_type: {self.model_type}")

    def train_model(self, model_class, **model_params):
        model = model_class(**model_params)
        model.fit(self.X_train, self.y_train)
        return model

    def train_traditional_model(self, model_class, num_models, **model_params):
        with ThreadPoolExecutor(max_workers=4) as executor:
            for _ in range(num_models):
                if model_class == "RandomForestClassifier":
                    model = RandomForestClassifier(**model_params)

                elif model_class == "XGBClassifier":
                    model = XGBClassifier(**model_params)
                elif model_class == "LGBMClassifier":
                    model = LGBMClassifier(**model_params)
                elif model_class == "AdaBoostClassifier":
                    model = AdaBoostClassifier(**model_params)
                else:
                    logging.error(f"Invalid model_class: {model_class}")
                    return

                model.fit(self.X_train, self.y_train)
                self.models.append(model)


    def train_neural_network(self, num_models):
        for _ in range(num_models):
            model = keras.Sequential([
                keras.layers.Dense(self.config.get('neural_network_params', {}).get('first_layer_units', 64), activation='relu', input_shape=(self.X_train.shape[1],)),
                keras.layers.Dense(self.config.get('neural_network_params', {}).get('second_layer_units', 64), activation='relu'),
                keras.layers.Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(self.X_train, self.y_train, epochs=self.config.get('neural_network_params', {}).get('epochs', 10), batch_size=self.config.get('neural_network_params', {}).get('batch_size', 32))
            self.models.append(model)

        with ThreadPoolExecutor(max_workers=4) as executor:
            for _ in range(num_models):
                self.model = RandomForestClassifier()
                self.model.fit(self.X_train, self.y_train)
                self.models.append(self.model)

    def train_transformer_model(self):
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=3,              # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
        )

        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=self.train_dataset,    # training dataset
            eval_dataset=self.val_dataset        # evaluation dataset
        )

        trainer.train()

    def train_ucl_model(self):
        # Add your UCL model training code here.
        pass

    def train_gpt2_model(self):
        num_labels = len(self.y.unique())
        model = GPT2LMHeadModel.from_pretrained('gpt2', num_labels=num_labels)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        def tokenize(batch):
            return tokenizer(batch['text'], padding=True, truncation=True)

        self.train_dataset = self.train_dataset.map(tokenize, batched=True, batch_size=len(self.train_dataset))
        self.val_dataset = self.val_dataset.map(tokenize, batched=True, batch_size=len(self.val_dataset))

        self.train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        self.val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=self.config.get('transformer_params', {}).get('num_epochs', 3),
            per_device_train_batch_size=self.config.get('transformer_params', {}).get('batch_size', 16),
            per_device_eval_batch_size=self.config.get('transformer_params', {}).get('batch_size', 16),
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            greater_is_better=True,
            evaluation_strategy="steps"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset
        )

        trainer.train()

    def evaluate_and_select_model(self):
        scores = []
        for model in self.models:
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            scores.append(accuracy)
        self.model = self.models[np.argmax(scores)]

class ModelEvaluator:
    def calculate_metrics(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, y_pred)

        return {'precision': precision, 'recall': recall, 'f1': f1, 'auc_roc': auc_roc}

    def evaluate(self, model, X, y):
        # ...
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred)
        
    def save_model(self, path=None):
        path = self.config.get('model_save_path', 'model.pkl')
        joblib.dump(self.model, path)
        logging.info(f"Model saved: {path}")
        
    def load_model(self, path=None):
        path = self.config.get('model_load_path', 'model.pkl')
        self.model = joblib.load(path)
        logging.info(f"Model loaded: {path}")

    def load_config(self, config_path):
        with open(config_path) as f:
            return json.load(f)

    def load_data(self, data_path=None):
        data_path = self.config.get('data_path')
        if data_path and os.path.isfile(data_path):
            self.df = pd.read_csv(data_path)
        else:
            logging.error(f"No file found at {data_path}")
            raise FileNotFoundError(f"No file found at {data_path}")
        
    def preprocess_data(self):
        if self.df.isnull().sum().any():
            logging.error("Missing values found in the data.")
            raise ValueError("Missing values found in the data.")
        self.X = self.df.drop('class', axis=1)
        self.y = self.df['class']

    def split_data(self, test_size=None, random_state=None):
        test_size = self.config.get('test_size', 0.3)
        random_state = self.config.get('random_state', 42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)


    def run(self):
        DataTransformer.convert_to_csv(self.config.get('data_directory'), self.config.get('csv_directory'))
        self.load_data()
        self.preprocess_data()
        self.split_data()
        if self.model_type == "traditional":
            self.train_and_update()
            self.evaluate_and_select_model()
            self.save_model()
        elif self.model_type == "transformer":
            self.train_transformer_model()
 
    def start_training(self):
        self.run()  # trains the model once
        threading.Timer(self.config.get('train_interval', 86400), self.start_training).start()  # retrains the model every specified interval (in seconds)

    def train_transformer_model(self):
        model_class = self.config.get('model_class', "bert-base-uncased")
        num_labels = len(self.y.unique())
        model = BertForSequenceClassification.from_pretrained(model_class, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(model_class)

        def tokenize(batch):
            return tokenizer(batch['text'], padding=True, truncation=True)

        self.train_dataset = self.train_dataset.map(tokenize, batched=True, batch_size=len(self.train_dataset))
        self.val_dataset = self.val_dataset.map(tokenize, batched=True, batch_size=len(self.val_dataset))

        self.train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        self.val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=self.config.get('transformer_params', {}).get('num_epochs', 3),
            per_device_train_batch_size=self.config.get('transformer_params', {}).get('batch_size', 16),
            per_device_eval_batch_size=self.config.get('transformer_params', {}).get('batch_size', 16),
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            greater_is_better=True,
            evaluation_strategy="steps"
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset
        )
        trainer.train()
trainer = ModelTrainer('config.json')
trainer.start_training()  # starts the training schedule

# Unit test
class TestModelTrainer(TestCase):
    def setUp(self):
        self.trainer = ModelTrainer('config.json')

    def test_load_data(self):
        self.trainer.load_data('path/to/data.csv')
        self.assertIsNotNone(self.trainer.df)

    def test_preprocess_data(self):
        self.trainer.load_data('path/to/data.csv')
        self.trainer.preprocess_data()
        self.assertIsNotNone(self.trainer.X)
        self.assertIsNotNone(self.trainer.y)

    def test_split_data(self):
        self.trainer.load_data('path/to/data.csv')
        self.trainer.preprocess_data()
        self.trainer.split_data()
        self.assertIsNotNone(self.trainer.X_train)
        self.assertIsNotNone(self.trainer.X_test)
        self.assertIsNotNone(self.trainer.y_train)
        self.assertIsNotNone(self.trainer.y_test)

    def test_train_and_update(self):
        self.trainer.load_data('path/to/data.csv')
        self.trainer.preprocess_data()
        self.trainer.split_data()
        self.trainer.train_and_update()
        self.assertIsNotNone(self.trainer.model)

    def test_save_and_load_model(self):
        self.trainer.load_data('path/to/data.csv')
        self.trainer.preprocess_data()
        self.trainer.split_data()
        self.trainer.train_and_update()
        self.trainer.save_model('path/to/save/model')
        self.trainer.model = None
        self.trainer.load_model('path/to/save/model')
        self.assertIsNotNone(self.trainer.model)

if __name__ == "__main__":
    trainer = ModelTrainer("config.json")
    trainer.start_training()

