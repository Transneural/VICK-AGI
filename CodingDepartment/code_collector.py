import ast
import os
import shutil
import tempfile
import git
import schedule
import time
from ast import parse
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression

class CodeCollector:
    def __init__(self, destination_folder):
        self.destination_folder = destination_folder
        self.feature_extraction_model = None
        self.vectorizer = None

    def fetch_codebases(self, repository_urls: List[str]):
        for url in repository_urls:
            self.clone_repository(url)

    def clone_repository(self, url: str):
        temp_dir = tempfile.mkdtemp()
        try:
            git.Repo.clone_from(url, temp_dir)
            self.process_repository(temp_dir)
        except git.GitCommandError:
            print(f"Failed to clone repository: {url}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def process_repository(self, repo_folder: str):
        python_files = self.find_python_files(repo_folder)
        for file_path in python_files:
            try:
                code = self.read_file(file_path)
                self.save_code(code)
            except Exception as e:
                print(f"Error processing file: {file_path}. {e}")

    def find_python_files(self, folder: str) -> List[str]:
        python_files = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        return python_files

    def read_file(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as file:
            code = file.read()
        return code

    def save_code(self, code: str):
        function_name = self.extract_function_name(code)
        if function_name:
            function_path = os.path.join(self.destination_folder, f"{function_name}.py")
            with open(function_path, "w", encoding="utf-8") as file:
                file.write(code)

                # Extract additional features
                features = self.extract_features(code)

                # Predict relevant features using the feature extraction model
                predicted_features = self.feature_extraction_model.predict(self.vectorizer.transform([code]))

                # Update the feature extraction model with the new example
                X = self.vectorizer.transform([code])
                y = predicted_features
                self.feature_extraction_model.partial_fit(X, y)

                print(f"Extracted features: {features}")
                print(f"Predicted features: {predicted_features}")

    def extract_function_name(self, code: str) -> str:
        module_ast = parse(code)
        for node in module_ast.body:
            if isinstance(node, ast.FunctionDef):
                return node.name
        return None

    def collect_code_from_folder(self, folder: str):
        python_files = self.find_python_files(folder)
        for file_path in python_files:
            try:
                code = self.read_file(file_path)
                self.save_code(code)
            except Exception as e:
                print(f"Error processing file: {file_path}. {e}")

    def collect_code_periodically(self, folder: str, interval: int):
        while True:
            self.collect_code_from_folder(folder)
            time.sleep(interval)

    def train_feature_extraction_model(self, code_folder: str):
        functions = []
        function_features = []
        for file_name in os.listdir(code_folder):
            with open(os.path.join(code_folder, file_name), "r", encoding="utf-8") as file:
                function_code = file.read()
                functions.append(function_code)
                features = self.extract_features(function_code)
                function_features.append(features)

        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(functions)

        self.feature_extraction_model = LogisticRegression()
        self.feature_extraction_model.fit(X, function_features)

    def extract_features(self, code: str) -> List[str]:
        features = []
        module_ast = parse(code)
        for node in module_ast.body:
            if isinstance(node, ast.FunctionDef):
                features.append(node.name)
                features += self.extract_arguments(node)
                features += self.extract_returns(node)
                features += self.extract_decorators(node)
                features += self.extract_comments(node)
                features += self.extract_exceptions(node)
                features += self.extract_imports(node)
        return features

    def extract_arguments(self, function_node: ast.FunctionDef) -> List[str]:
        arguments = []
        for arg_node in function_node.args.args:
            arguments.append(arg_node.arg)
        return arguments

    def extract_returns(self, function_node: ast.FunctionDef) -> List[str]:
        returns = []
        if function_node.returns:
            returns.append("return")
        return returns

    def extract_decorators(self, function_node: ast.FunctionDef) -> List[str]:
        decorators = []
        for decorator_node in function_node.decorator_list:
            if isinstance(decorator_node, ast.Name):
                decorators.append(decorator_node.id)
        return decorators

    def extract_comments(self, function_node: ast.FunctionDef) -> List[str]:
        comments = []
        for child_node in ast.iter_child_nodes(function_node):
            if isinstance(child_node, ast.Expr) and isinstance(child_node.value, ast.Str):
                comments.append(child_node.value.s)
        return comments

    def extract_exceptions(self, function_node: ast.FunctionDef) -> List[str]:
        exceptions = []
        for child_node in ast.iter_child_nodes(function_node):
            if isinstance(child_node, ast.Try):
                for handler in child_node.handlers:
                    if isinstance(handler.type, ast.Name):
                        exceptions.append(handler.type.id)
        return exceptions

    def extract_imports(self, function_node: ast.FunctionDef) -> List[str]:
        imports = []
        for child_node in ast.iter_child_nodes(function_node):
            if isinstance(child_node, ast.Import):
                for alias in child_node.names:
                    if isinstance(alias, ast.alias):
                        imports.append(alias.name)
            elif isinstance(child_node, ast.ImportFrom):
                if isinstance(child_node.module, ast.Module):
                    imports.append(child_node.module.name)
                else:
                    imports.append(child_node.module)
        return imports


# Code collection and vectorization
code_folder = "code_snippets"
output_folder = "vectorized_code"
collector = CodeCollector(output_folder)

# Fetch code from repositories
repository_urls = ["", ""]
collector.fetch_codebases(repository_urls)

# Collect code from a specific folder
folder_to_collect = "additional_code"
collector.collect_code_from_folder(folder_to_collect)

# Schedule periodic code collection
interval = 3600  # 1 hour
collector.collect_code_periodically(folder_to_collect, interval)

# Train the feature extraction model
collector.train_feature_extraction_model(code_folder)
