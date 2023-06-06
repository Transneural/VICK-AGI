import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

class AdaptiveNeuroFuzzyAgent:
    def __init__(self):
        # Define the input and output variables
        self.input_variable1 = ctrl.Antecedent(np.arange(0, 11, 1), 'input_variable1')
        self.input_variable2 = ctrl.Antecedent(np.arange(0, 11, 1), 'input_variable2')
        self.output_variable = ctrl.Consequent(np.arange(0, 11, 1), 'output_variable')

        # Define the membership functions for input and output variables
        self.input_variable1['low'] = fuzz.trimf(self.input_variable1.universe, [0, 2, 4])
        self.input_variable1['medium'] = fuzz.gaussmf(self.input_variable1.universe, 5, 1)
        self.input_variable1['high'] = fuzz.trapmf(self.input_variable1.universe, [6, 7, 10, 10])

        self.input_variable2['low'] = fuzz.trimf(self.input_variable2.universe, [0, 2, 4])
        self.input_variable2['medium'] = fuzz.gaussmf(self.input_variable2.universe, 5, 1)
        self.input_variable2['high'] = fuzz.trapmf(self.input_variable2.universe, [6, 7, 10, 10])

        self.output_variable['low'] = fuzz.trimf(self.output_variable.universe, [0, 2, 4])
        self.output_variable['medium'] = fuzz.gaussmf(self.output_variable.universe, 5, 1)
        self.output_variable['high'] = fuzz.trapmf(self.output_variable.universe, [6, 7, 10, 10])

        # Define the fuzzy control system
        self.control_system = ctrl.ControlSystem()
        self.control = ctrl.ControlSystemSimulation(self.control_system)

        # Create the neural network for learning the fuzzy logic
        self.neural_network = MLPRegressor(hidden_layer_sizes=(10, 10))
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 100

        # Other variables for adaptive learning
        self.data_history = []
        self.label_history = []

    def train_neuro_fuzzy(self):
        # Fine-tune hyperparameters using GridSearchCV
        param_grid = {'learning_rate_init': [0.001, 0.01, 0.1],
                      'batch_size': [16, 32, 64],
                      'max_iter': [50, 100, 200]}
        grid_search = GridSearchCV(self.neural_network, param_grid, cv=3)
        grid_search.fit(self.data_history, self.label_history)

        # Update hyperparameters with best values
        self.learning_rate = grid_search.best_params_['learning_rate_init']
        self.batch_size = grid_search.best_params_['batch_size']
        self.num_epochs = grid_search.best_params_['max_iter']

        # Train the neural network with the fine-tuned hyperparameters
        self.neural_network.set_params(learning_rate_init=self.learning_rate, batch_size=self.batch_size, max_iter=self.num_epochs)
        self.neural_network.fit(self.data_history, self.label_history)

    def update_rules(self, input_data, output_data):
        # Define the fuzzy rules based on new data
        rules = [
            ctrl.Rule(self.input_variable1[term1] & self.input_variable2[term2], self.output_variable[output])
            for term1 in self.input_variable1.terms.keys()
            for term2 in self.input_variable2.terms.keys()
            for output in self.output_variable.terms.keys()
        ]

        # Add the new rules to the control system
        for rule in rules:
            self.control_system.addrule(rule)

        # Update neural network with new data
        self.data_history.append(input_data)
        self.label_history.append(output_data)
        self.train_neuro_fuzzy()

    def evaluate_input(self, input_data):
        # Set the input values for evaluation
        self.control.input['input_variable1'] = input_data[0]
        self.control.input['input_variable2'] = input_data[1]

        # Evaluate the fuzzy control system using the Mamdani inference method
        self.control.compute(using='mamdani')

        # Perform centroid-based defuzzification
        centroid_defuzzification = fuzz.defuzz(self.output_variable.universe, self.control.output['output_variable'], 'centroid')

        # Return the centroid defuzzification result
        return centroid_defuzzification

    def reset(self):
        # Reset the agent's state and history
        self.control.reset()
        self.neural_network = MLPRegressor(hidden_layer_sizes=(10, 10))
        self.data_history = []
        self.label_history = []

    def add_input_variable(self, variable_name, universe, membership_functions):
        # Add a new input variable to the agent's fuzzy logic system
        new_variable = ctrl.Antecedent(universe, variable_name)
        for membership in membership_functions:
            new_variable[membership['name']] = getattr(fuzz, membership['function'])(new_variable.universe, *membership['params'])
        setattr(self, variable_name, new_variable)

    def add_output_variable(self, variable_name, universe, membership_functions):
        # Add a new output variable to the agent's fuzzy logic system
        new_variable = ctrl.Consequent(universe, variable_name)
        for membership in membership_functions:
            new_variable[membership['name']] = getattr(fuzz, membership['function'])(new_variable.universe, *membership['params'])
        setattr(self, variable_name, new_variable)

    def add_rule(self, rule):
        # Add a new rule to the agent's fuzzy control system
        self.control_system.addrule(rule)

    def set_neural_network_params(self, hidden_layer_sizes, learning_rate=0.001, batch_size=32, num_epochs=100):
        # Set the parameters of the neural network used for learning the fuzzy logic
        self.neural_network = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def get_current_rules(self):
        # Get the current rules in the agent's fuzzy control system
        return self.control_system.rules

    def get_current_membership_functions(self, variable_name):
        # Get the current membership functions of a given variable in the agent's fuzzy logic system
        variable = getattr(self, variable_name, None)
        if variable:
            return variable.terms
        return None

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

class AdaptiveNeuroFuzzyAgent:
    def __init__(self):
        # Define the input and output variables
        self.input_variable1 = ctrl.Antecedent(np.arange(0, 11, 1), 'input_variable1')
        self.input_variable2 = ctrl.Antecedent(np.arange(0, 11, 1), 'input_variable2')
        self.output_variable = ctrl.Consequent(np.arange(0, 11, 1), 'output_variable')

        # Define the membership functions for input and output variables
        self.input_variable1['low'] = fuzz.trimf(self.input_variable1.universe, [0, 2, 4])
        self.input_variable1['medium'] = fuzz.gaussmf(self.input_variable1.universe, 5, 1)
        self.input_variable1['high'] = fuzz.trapmf(self.input_variable1.universe, [6, 7, 10, 10])

        self.input_variable2['low'] = fuzz.trimf(self.input_variable2.universe, [0, 2, 4])
        self.input_variable2['medium'] = fuzz.gaussmf(self.input_variable2.universe, 5, 1)
        self.input_variable2['high'] = fuzz.trapmf(self.input_variable2.universe, [6, 7, 10, 10])

        self.output_variable['low'] = fuzz.trimf(self.output_variable.universe, [0, 2, 4])
        self.output_variable['medium'] = fuzz.gaussmf(self.output_variable.universe, 5, 1)
        self.output_variable['high'] = fuzz.trapmf(self.output_variable.universe, [6, 7, 10, 10])

        # Define the fuzzy control system
        self.control_system = ctrl.ControlSystem()
        self.control = ctrl.ControlSystemSimulation(self.control_system)

        # Create the neural network for learning the fuzzy logic
        self.neural_network = MLPRegressor(hidden_layer_sizes=(10, 10))
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 100

        # Other variables for adaptive learning
        self.data_history = []
        self.label_history = []

    def train_neuro_fuzzy(self):
        # Fine-tune hyperparameters using GridSearchCV
        param_grid = {'learning_rate_init': [0.001, 0.01, 0.1],
                      'batch_size': [16, 32, 64],
                      'max_iter': [50, 100, 200]}
        grid_search = GridSearchCV(self.neural_network, param_grid, cv=3)
        grid_search.fit(self.data_history, self.label_history)

        # Update hyperparameters with best values
        self.learning_rate = grid_search.best_params_['learning_rate_init']
        self.batch_size = grid_search.best_params_['batch_size']
        self.num_epochs = grid_search.best_params_['max_iter']

        # Train the neural network with the fine-tuned hyperparameters
        self.neural_network.set_params(learning_rate_init=self.learning_rate, batch_size=self.batch_size, max_iter=self.num_epochs)
        self.neural_network.fit(self.data_history, self.label_history)

    def update_rules(self, input_data, output_data):
        # Define the fuzzy rules based on new data
        rules = [
            ctrl.Rule(self.input_variable1[term1] & self.input_variable2[term2], self.output_variable[output])
            for term1 in self.input_variable1.terms.keys()
            for term2 in self.input_variable2.terms.keys()
            for output in self.output_variable.terms.keys()
        ]

        # Add the new rules to the control system
        for rule in rules:
            self.control_system.addrule(rule)

        # Update neural network with new data
        self.data_history.append(input_data)
        self.label_history.append(output_data)
        self.train_neuro_fuzzy()

    def evaluate_input(self, input_data):
        # Set the input values for evaluation
        self.control.input['input_variable1'] = input_data[0]
        self.control.input['input_variable2'] = input_data[1]

        # Evaluate the fuzzy control system using the Mamdani inference method
        self.control.compute(using='mamdani')

        # Perform centroid-based defuzzification
        centroid_defuzzification = fuzz.defuzz(self.output_variable.universe, self.control.output['output_variable'], 'centroid')

        # Return the centroid defuzzification result
        return centroid_defuzzification

    def reset(self):
        # Reset the agent's state and history
        self.control.reset()
        self.neural_network = MLPRegressor(hidden_layer_sizes=(10, 10))
        self.data_history = []
        self.label_history = []

    def add_input_variable(self, variable_name, universe, membership_functions):
        # Add a new input variable to the agent's fuzzy logic system
        new_variable = ctrl.Antecedent(universe, variable_name)
        for membership in membership_functions:
            new_variable[membership['name']] = getattr(fuzz, membership['function'])(new_variable.universe, *membership['params'])
        setattr(self, variable_name, new_variable)

    def add_output_variable(self, variable_name, universe, membership_functions):
        # Add a new output variable to the agent's fuzzy logic system
        new_variable = ctrl.Consequent(universe, variable_name)
        for membership in membership_functions:
            new_variable[membership['name']] = getattr(fuzz, membership['function'])(new_variable.universe, *membership['params'])
        setattr(self, variable_name, new_variable)

    def update_rules(self, input_data, output_data):
        # Define the fuzzy rules based on new data
        rules = [
            ctrl.Rule(self.input_variable1[term1] & self.input_variable2[term2], self.output_variable[output])
            for term1 in self.input_variable1.terms.keys()
            for term2 in self.input_variable2.terms.keys()
            for output in self.output_variable.terms.keys()
        ]

        # Add the new rules to the control system
        for rule in rules:
            self.add_rule(rule)

        # Update neural network with new data
        self.data_history.append(input_data)
        self.label_history.append(output_data)
        self.train_neuro_fuzzy()
    def set_neural_network_params(self, hidden_layer_sizes, learning_rate=0.001, batch_size=32, num_epochs=100):
        # Set the parameters of the neural network used for learning the fuzzy logic
        self.neural_network = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def get_current_rules(self):
        # Get the current rules in the agent's fuzzy control system
        return self.control_system.rules

    def get_current_membership_functions(self, variable_name):
        # Get the current membership functions of a given variable in the agent's fuzzy logic system
        variable = getattr(self, variable_name, None)
        if variable:
            return variable.terms
        return None
    
    def remove_rule(self, rule):
    # Remove the specified rule from the control system
     self.control_system.rules.remove(rule)

    def clear_rules(self):
    # Clear all the rules from the control system
     self.control_system.rules.clear()

    def update_membership_functions(self, variable, mf_names, mf_params):
    # Update the membership functions of the specified variable
     for name, params in zip(mf_names, mf_params):
        variable[name].mf[0].updatemf(params)

    def evaluate_membership_functions(self, variable, input_values):
    # Evaluate the membership values for the specified variable and input values
     membership_values = []
     for value in input_values:
        membership_values.append([variable[name].mf[0].membership(value) for name in variable.terms.keys()])
     return np.array(membership_values)


    def add_rule(self, rule):
        # Add a new rule to the agent's fuzzy control system
        self.control_system.addrule(rule)