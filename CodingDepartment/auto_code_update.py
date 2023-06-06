import os
import time
import numpy as np
from bs4 import BeautifulSoup
import requests
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from code_evaluator import CodeEvaluator
from code_generator_agent import CodeGenerator
from code_tester import CodeTester
from git_integration import GitIntegration
from scheduler import Scheduler

import logging

# Set up logging
logging.basicConfig(filename='code_generator.log', level=logging.INFO)

class AdvancedCodeGenerator(CodeEvaluator, GitIntegration, CodeTester, Scheduler):
    def __init__(self, data, model, repo_path):
        super().__init__(data=data, model=model, repo_path=repo_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = model

    def generate_code(self, description):
        try:
            inputs = self.tokenizer.encode(description, return_tensors='pt')

            # Generate a sequence of tokens
            output = self.model.generate(inputs, max_length=1000, temperature=0.7)

            # Decode the tokens into a string
            code = self.tokenizer.decode(output[0])

            return code
        except Exception as e:
            logging.error(f"Error during code generation: {e}")
            return None

    def tokenize_data(self):
        self.data = self.tokenizer.encode(self.data, return_tensors='pt')

    def create_sequences(self):
        sequences = [self.data[:, i: i + self.sequence_length] for i in range(self.data.size(1) - self.sequence_length)]
        self.dataset = sequences

    def train(self, epochs=5):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for batch in self.dataset:
                optimizer.zero_grad()
                inputs, labels = batch[:, :-1], batch[:, 1:]
                outputs = self.model(inputs)
                loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                loss.backward()
                optimizer.step()

    def continuous_learning(self):
        # This method implements a simple dynamic learning schedule
        # The model is retrained every time a certain amount of new data is collected
        new_data_threshold = 10000  # Arbitrary threshold

        while True:
            # Collect new data
            urls = ["https://www.python.org/, https://www.learnpython.org/, https://github.com/vinta/awesome-python,https://docs.python-guide.org/"]  # Replace with actual URLs
            self.collect_data(urls)

            # Retrain model if enough new data has been collected
            if len(self.data) >= new_data_threshold:
                self.tokenize_data()
                self.create_sequences()
                self.train()
                self.data = None  # Reset data

            time.sleep(3600)  # Sleep for an hour before collecting more data

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def update_model(self, model_path):
        # Implement model update mechanism
        torch.save(self.model.state_dict(), model_path)

    def collect_data(self, urls):
        # This method uses BeautifulSoup to scrape Python code from specified URLs
        for url in urls:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')

                # Assuming Python code is in <code> tags
                code_blocks = soup.find_all('code')

                for block in code_blocks:
                    self.data += block.text + '\n'
            except Exception as e:
                logging.error(f"Error during data collection: {e}")

    def update_all_files(self, description, directory="."):
        try:
            # Find the most relevant file for the given description
            target_file = self.get_most_relevant_file(description, directory)

            # Update the relevant file with the generated code
            new_code = self.generate_code(description)

            if new_code:
                with open(os.path.join(directory, target_file), "a") as f:
                    f.write(new_code)

                self.commit_changes(description)
                return True
            else:
                print("Generated code was not considered good. Not updating the code file.")
                return False
        except Exception as e:
            logging.error(f"Error during file update: {e}")

    def get_most_relevant_file(self, description, directory):
        python_files = [f for f in os.listdir(directory) if f.endswith('.py')]
        file_contents = []

        for file in python_files:
            with open(os.path.join(directory, file), "r") as f:
                file_contents.append(f.read())

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(file_contents)
        query_vector = vectorizer.transform([description])

        cosine_similarities = cosine_similarity(query_vector, X).flatten()
        most_relevant_index = np.argmax(cosine_similarities)

        return python_files[most_relevant_index]

# Load the initial data and model
initial_data = ""  # Provide initial data
initial_model = GPT2LMHeadModel.from_pretrained('gpt2')

repo_path = '/Users/mariojuric/Desktop/V.I.C.K.'  # Set the repository path

# Initialize CodeGenerator with initial data and model
code_gen = AdvancedCodeGenerator(initial_data, initial_model, repo_path)

# Start the continuous learning process
code_gen.continuous_learning()

# Schedule daily updates
code_gen.schedule_daily_updates()
