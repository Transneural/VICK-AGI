import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class DomainSpecificModel:
    def __init__(self, data=None, cv=5):
        self.data = data
        self.cv = cv

    def fetch_domain_specific_data(self):
        # Load the Iris dataset
        iris = load_iris()
        iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                               columns=iris['feature_names'] + ['target'])
        self.data = iris_df.values

    def integrate_domain_knowledge(self):
        # Add a new feature based on domain knowledge
        X = self.data[:, :-1]
        new_feature = X[:, 0] * X[:, 1]
        new_feature = new_feature.reshape(-1, 1)
        self.data = np.hstack((X, new_feature, self.data[:, -1].reshape(-1, 1)))

    def preprocess_data(self, X):
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    def get_classifier(self):
        # Create a pipeline for the Decision Tree classifier with hyperparameter tuning
        pipeline = Pipeline([
            ('classifier', DecisionTreeClassifier())
        ])

        # Define the hyperparameter search space
        param_grid = {
            'classifier__max_depth': [None, 10, 20, 30, 40],
            'classifier__min_samples_leaf': [1, 2, 4]
        }

        # Perform a grid search with cross-validation to find the best hyperparameters
        grid_search = GridSearchCV(pipeline, param_grid, cv=self.cv, scoring='accuracy')
        return grid_search

    def train(self):
        if self.data is None:
            self.fetch_domain_specific_data()
        self.integrate_domain_knowledge()

        X = self.data[:, :-1]
        y = self.data[:, -1]

        X = self.preprocess_data(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        grid_search = self.get_classifier()
        grid_search.fit(X_train, y_train)

        print("Training the domain-specific model...")
        classifier = grid_search.best_estimator_
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        cv_scores = cross_val_score(classifier, X_train, y_train, cv=self.cv)
        print("Cross-validation scores:", cv_scores)
        print("Mean cross-validation score:", np.mean(cv_scores))

        return classifier

# Usage
model = DomainSpecificModel(cv=5)
trained_classifier = model.train()
