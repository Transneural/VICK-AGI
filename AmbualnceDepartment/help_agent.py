from queue import PriorityQueue
import threading
import spacy

class HelpAgent:
    def __init__(self, manager_agent):
        self.manager_agent = manager_agent
        self.help_requests = PriorityQueue()
        self.max_responses = 3
        self.max_helps = 2
        self.num_active_helps = 0
        self.lock = threading.Lock()
        self.agent_registry = {}

    def request_help(self, agent, help_data, priority=0, complexity=1):
        min_responses = min(3, complexity)  # Minimum responses based on complexity
        max_helps = min(2, complexity)  # Maximum helps based on complexity

        with self.lock:
            self.max_responses = min_responses
            self.max_helps = max_helps

        self.help_requests.put((priority, agent, help_data))
        
    def process_help_requests(self, num_threads=1):
        # Create and start multiple threads to handle the help requests
        for _ in range(num_threads):
            threading.Thread(target=self._process_help_request).start()

    def _process_help_request(self):
        while not self.help_requests.empty():
            with self.lock:
                if self.num_active_helps >= self.max_helps:
                    # Maximum number of active helps reached, skip this request for now
                    break

            priority, agent, help_data = self.help_requests.get()
            responses = self.manager_agent.get_agent_responses(agent, help_data)[:self.max_responses]

            if not responses:
                self.handle_no_responses(agent, help_data)
            elif len(responses) >= self.max_responses:
                self.handle_too_many_responses(agent, help_data, responses)
            else:
                self.handle_help(agent, help_data, responses)

    def handle_no_responses(self, agent, help_data):
        # Logic to handle the scenario where no agents respond to the help request
        pass

    def handle_too_many_responses(self, agent, help_data, responses):
        # Logic to handle the scenario where too many agents respond to the help request
        pass

    def handle_help(self, agent, help_data, responses):
        with self.lock:
            self.num_active_helps += 1

        try:
            problem_type = self.discover_problem_type(help_data)
            suitable_agents = self.find_suitable_agents(problem_type)

            if not suitable_agents:
                self.handle_no_suitable_agents(agent, help_data)
            else:
                selected_agents = self.select_agents(suitable_agents, self.max_responses)
                self.assign_task_to_agents(selected_agents, help_data)

        finally:
            with self.lock:
                self.num_active_helps -= 1

    def discover_problem_type(self, help_data):
        # Analyze the help data to determine the problem type using spaCy for keyword extraction as a fallback

        # Load the spaCy English model
        nlp = spacy.load("en_core_web_lg")

        problem_type = None  # Initialize the problem type as None

        # Predefined problem keywords
        problem_keywords = {
            'error': ['error', 'bug', 'crash'],
            'performance': ['slow', 'lag', 'performance'],
            'configuration': ['setting', 'configure', 'setup'],
            # Add more problem types and associated keywords as needed
        }

        # Tokenize the help data using spaCy
        doc = nlp(help_data.lower())

        # Check for direct matches with predefined problem keywords
        for problem, keywords in problem_keywords.items():
            for keyword in keywords:
                if keyword in help_data.lower():
                    problem_type = problem
                    break

            if problem_type is not None:
                break

        # If no direct match is found, use spaCy for keyword extraction
        if problem_type is None:
            extracted_keywords = set()
            for token in doc:
                if token.is_alpha and not token.is_stop:
                    extracted_keywords.add(token.lemma_)

            # Check the extracted keywords against the predefined problem keywords
            for problem, keywords in problem_keywords.items():
                if any(keyword in extracted_keywords for keyword in keywords):
                    problem_type = problem
                    break

        return problem_type

    def find_suitable_agents(self, problem_type):
        # Query the ManagerAgent and the registry to find agents capable of handling the problem type
        # Return a list of suitable agents

        suitable_agents = []

        # Query the ManagerAgent for agents with the specified problem type
        agents_from_manager = self.manager_agent.query_agents(problem_type)

        # Check if the agents from the ManagerAgent are already in the registry
        for agent in agents_from_manager:
            if agent in self.agent_registry:
                suitable_agents.append(agent)

        # Check the registry for agents with the specified problem type
        for agent_id, agent_capabilities in self.agent_registry.items():
            if problem_type in agent_capabilities and agent_id not in suitable_agents:
                suitable_agents.append(agent_id)

        return suitable_agents

    def handle_no_suitable_agents(self, agent, help_data):
        # Logic to handle the scenario where no suitable agents are found

        # Create a new agent to handle the request
        new_agent_id = self.manager_agent.create_agent(help_data)  # Call the appropriate method in the ManagerAgent

        # Communicate with the ManagerAgent to inform about the new agent
        self.manager_agent.register_agent(new_agent_id)  # Register the new agent in the ManagerAgent

        # Perform any additional actions or communication as needed
        # ...

        # Request help from the ManagerAgent with the new agent
        self.manager_agent.request_help(new_agent_id, help_data)  # Request help for the specific agent

        # Update the registry to include the new agent
        self.register_agent(new_agent_id)  # Register the new agent in the HelpAgent's registry

        # Perform any other actions or error handling as needed
        # ...

    def assign_task_to_agents(self, agents, help_data):
        # Assign the task to the selected agents
        # Additional logic for task assignment and coordination
        pass

    def select_agents(self, agents, num_agents):
        # Select the top-ranked agents from the list based on specific criteria
        # Return the selected agents

        # Implement your custom logic to select the agents based on specific criteria
        # Example: Random selection or ranking based on agent capabilities

        return agents[:num_agents]

    def register_agent(self, agent_id):
        # Register an agent in the HelpAgent's registry
        self.agent_registry.add(agent_id)


#Apologies for the confusion. If you want to add a standalone "Help" 
# function that can be called by other agents, you can modify the CollectiveBrain class to include the help functionality and expose it to other agents. Here's an example of how you can do that:


