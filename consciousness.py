from embodied_situated import EmbodiedSituatedModule
from cognitive_architectures import CognitiveArchitecturesModule
from iit import IITModule
from chatbot import Chatbot

# Instantiate the modules
embodied_situated_module = EmbodiedSituatedModule()
cognitive_architectures_module = CognitiveArchitecturesModule()
iit_module = IITModule()

# Instantiate the chatbot
chatbot = Chatbot()

# Connect the modules to the chatbot
chatbot.add_module(embodied_situated_module)
chatbot.add_module(cognitive_architectures_module)
chatbot.add_module(iit_module)

# Start the chatbot
chatbot.start()
