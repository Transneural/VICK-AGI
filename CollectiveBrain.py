from queue import PriorityQueue
import threading
import spacy

from AmbualnceDepartment.help_agent import HelpAgent
from NLP.sentiment_analysis import SentimentAnalyzer
from SchoolDepartment.quantum_inspired_agent import QuantumInspiredNetwork
from SchoolDepartment.transfer_learning import BERTSentimentClassifier


class CollectiveBrain:
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.quantum_network = QuantumInspiredNetwork()
        self.bert_classifier = BERTSentimentClassifier()

    def train_sentiment_analyzer(self, data_file, label_file):
        self.sentiment_analyzer.load_data_labels(data_file, label_file)
        self.sentiment_analyzer.train()

    def predict_sentiment(self, text):
        return self.sentiment_analyzer.predict(text)

    def evaluate_sentiment_analyzer(self):
        self.sentiment_analyzer.evaluate()

    def plot_sentiment_training_history(self, history):
        self.sentiment_analyzer.plot_training_history(history)

    def train_quantum_network(self, X, y, task):
        self.quantum_network.train(X, y, task)

    def predict_quantum_network(self, X, task):
        return self.quantum_network.predict(X, task)

    def tune_quantum_network_hyperparameters(self, X, y, task):
        self.quantum_network.tune_hyperparameters(X, y, task)

    def increase_quantum_network_complexity(self, X, y):
        self.quantum_network.increase_complexity(X, y)

    def detect_task(self, query):
        return self.quantum_network.detect_task(query)

    def train_bert_classifier(self, train_data, val_data, output_dir, text_column='text', label_column='label', num_train_epochs=3, logging_steps=100, **kwargs):
        self.bert_classifier.train(train_data, val_data, output_dir, text_column, label_column, num_train_epochs, logging_steps, **kwargs)

    def predict_bert_sentiment(self, text):
        return self.bert_classifier.predict(text)

    def evaluate_bert_classifier(self, val_data, text_column='text', label_column='label'):
        return self.bert_classifier.evaluate(val_data, text_column, label_column)

    def hyperparameter_search_bert_classifier(self, train_data, val_data, output_dir, text_column='text', label_column='label', param_grid=None, **kwargs):
        self.bert_classifier.hyperparameter_search(train_data, val_data, output_dir, text_column, label_column, param_grid, **kwargs)


    def register_agent(self, agent_name, agent):
        self.agents[agent_name] = agent

    def handle_message(self, message):
        agent_name = message.get('agent')
        agent = self.agents.get(agent_name)

        if agent:
            method_name = message.get('method')
            method_args = message.get('args', [])
            method_kwargs = message.get('kwargs', {})
            
            # Call the method on the agent dynamically
            if hasattr(agent, method_name) and callable(getattr(agent, method_name)):
                method = getattr(agent, method_name)
                result = method(*method_args, **method_kwargs)
                return result
            else:
                # Handle unknown method
                return f"Unknown method '{method_name}' for agent '{agent_name}'"
        else:
            # Handle unknown agent
            return f"Unknown agent '{agent_name}'"
