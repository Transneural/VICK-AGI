import random
from sensory_integration import SensoryIntegration
from ability_manager import AbilityManager
from learning_module import LearningModule
from attention_mechanisam import AttentionMechanism


class PerceptionModule:
    def __init__(self):
        self.attention_mechanism = AttentionMechanism()
        self.sensory_integration = SensoryIntegration()
        self.learning_module = LearningModule()
        self.ability_manager = AbilityManager()
        self.discovered_information = {}
       

    def process_sensory_data(self, sensory_data):
        attended_data = self.attention_mechanism.attend(sensory_data)
        integrated_data = self.sensory_integration.integrate(attended_data)
        reasoning_output = self.perform_reasoning(integrated_data)
        self.learning_module.learn(reasoning_output)
        self.explore_new_abilities()
        self.optimize_abilities()
        self.generate_more_outputs()
        self.autonomously_discover_properties()
        self.autonomously_discover_conditions()
        self.save_discovered_information(integrated_data, reasoning_output)
        self.enhance_powerfulness()
        self.continue_autonomous_development()
        return reasoning_output

    def perform_reasoning(self, integrated_data):
        reasoning_output = self.learning_module.generate_output(integrated_data)
        return reasoning_output

    def explore_new_abilities(self):
        new_ability = self.learning_module.generate_new_ability()
        self.learning_module.train_new_ability(new_ability)
        self.learning_module.update_knowledge(new_ability)
        self.learning_module.optimize_abilities()

        if new_ability.is_promising():
            self.ability_manager.add_ability(new_ability)
            self.ability_manager.update_ability_scores()

        self.ability_manager.explore_abilities()

    def optimize_abilities(self):
        self.learning_module.optimize_abilities()

    def generate_more_outputs(self):
        self.learning_module.generate_more_outputs()

    def autonomously_discover_properties(self):
        new_property = self.learning_module.autonomously_discover_property()
        self.sensory_integration.add_property(new_property)

    def autonomously_discover_conditions(self):
        new_condition = self.learning_module.autonomously_discover_condition()
        self.add_new_condition_to_reasoning_algorithms(new_condition)

    def add_new_condition_to_reasoning_algorithms(self, new_condition):
        self._reasoning_algorithm_A.add_condition(new_condition)
        self._reasoning_algorithm_B.add_condition(new_condition)

    def save_discovered_information(self, integrated_data, reasoning_output):
        key = (integrated_data.condition, integrated_data.property)
        self.discovered_information[key] = reasoning_output

    def retrieve_discovered_information(self, integrated_data):
        key = (integrated_data.condition, integrated_data.property)
        if key in self.discovered_information:
            return self.discovered_information[key]
        else:
            return None

    def enhance_discovery(self):
        if self.enhance_discovery.is_supported():
            self.enhance_discovery.boost_discovery()
            self.enhance_discovery.optimize_parameters()

    def generate_random_number(self):
        return random.randint(0, 100)

    def generate_random_output(self):
        return self.enhance_discovery.generate_output()

    def continue_autonomous_development(self):
        self.enhance_discovery.autonomous_development()
        self.learning_module.autonomous_development()
        self.ability_manager.autonomous_development()
