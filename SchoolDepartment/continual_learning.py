import torch
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import ParameterGrid
import importlib

class EWC:
    def __init__(self, task_registry, device_ids=None, log_dir='./logs', **kwargs):
        super().__init__(**kwargs)
        self.task_registry = task_registry
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.writer = SummaryWriter(log_dir)
        self._fisher_information = {}
        self._means = {}
        self._importance = {}
        self._old_params = {}
        self._old_output = {}
        self._old_input = {}
        self.best_hyperparameters = {}

    def _configure_task(self, task_id):
        task_info = self.task_registry[task_id]
        module_name = task_info['module']
        class_name = task_info['class']
        hyperparameters = task_info['hyperparameters']

        module = importlib.import_module(module_name)
        task_class = getattr(module, class_name)
        task = task_class(**hyperparameters)

        self.models[task_id] = DataParallel(task.model, self.device_ids)
        self.optimizers[task_id] = torch.optim.Adam(task.model.parameters(), lr=task.learning_rate)
        self.schedulers[task_id] = StepLR(self.optimizers[task_id], step_size=10, gamma=0.1)
        self.best_hyperparameters[task_id] = hyperparameters

    def update(self, task_id, data_loader, **kwargs):
        model = self.models[task_id]
        optimizer = self.optimizers[task_id]
        scheduler = self.schedulers[task_id]

        model.train()
        criterion = kwargs.get('criterion')

        for epoch in range(kwargs.get('num_epochs', 1)):
            for batch in data_loader:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        scheduler.step()
        self.writer.add_scalar('Loss/train', loss, epoch)

    def compute_fisher(self, task_id, data_loader, criterion):
        model = self.models[task_id]
        fisher_matrices = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

        model.train()
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            for n, p in model.named_parameters():
                fisher_matrices[n].add_(p.grad.detach() ** 2)

        self._fisher_information[task_id] = fisher_matrices
        self._means[task_id] = {n: p.clone().detach() for n, p in model.named_parameters()}
        self._importance[task_id] = {n: torch.ones_like(p) for n, p in model.named_parameters()}

    def update_importance(self, task_id, data_loader):
        model = self.models[task_id]
        model.eval()
        for inputs, _ in data_loader:
            inputs = inputs.to(self.device)
            _ = model(inputs)  # Run forward pass to compute activations
            for n, p in model.named_parameters():
                importance = torch.abs(p.grad)  # Compute importance as absolute value of gradients
                self._importance[task_id][n] += importance

    def update_old_params(self, task_id):
        model = self.models[task_id]
        self._old_params[task_id] = {n: p.detach().clone() for n, p in model.named_parameters()}

    def update_old_output(self, task_id, data_loader):
        model = self.models[task_id]
        model.eval()
        self._old_input[task_id] = []
        self._old_output[task_id] = []
        for inputs, _ in data_loader:
            inputs = inputs.to(self.device)
            self._old_input[task_id].append(inputs.detach().clone())
            self._old_output[task_id].append(model(inputs).detach().clone())
        self._old_input[task_id] = torch.cat(self._old_input[task_id])
        self._old_output[task_id] = torch.cat(self._old_output[task_id])

    def update_all(self, task_id, data_loader, **kwargs):
        self.compute_fisher(task_id, data_loader, kwargs.get('criterion'))
        self.update_importance(task_id, data_loader)
        self.update_old_params(task_id)
        self.update_old_output(task_id, data_loader)
        self.update(task_id, data_loader, **kwargs)

    def evaluate(self, task_id, data_loader, metric):
        model = self.models[task_id]
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                correct += metric(outputs, targets)
                total += targets.size(0)
        return correct / total

    def to(self, hyperparameters):
        device = hyperparameters.get('device', 'cuda:0')
        num_cpus = hyperparameters.get('num_cpus', 1)
        device_ids = hyperparameters.get('device_ids', None)

        self.device = device
        self.device_ids = device_ids if device_ids is not None else [device]

        if 'cuda' in device:
            torch.cuda.set_device(device)  # Set the current CUDA device
            num_cpus = 0  # Set num_cpus to 0 if using GPU device
        else:
            torch.set_num_threads(num_cpus)  # Set the number of CPU threads

        for model in self.models.values():
            model.to(device)
        for task_id in self._means:
            self._means[task_id] = {n: p.to(device) for n, p in self._means[task_id].items()}
            self._importance[task_id] = {n: p.to(device) for n, p in self._importance[task_id].items()}

        return

    def select_model(self, task_id):
        # your model selection logic here
        task_info = self.task_registry[task_id]
        available_models = task_info['available_models']
        # Perform model selection based on your logic
        model_name = ...
        return model_name

    def select_learning_rate(self, task_id):
        # your dynamic learning rate scheduling logic here
        task_info = self.task_registry[task_id]
        available_learning_rates = task_info['available_learning_rates']
        # Perform learning rate selection based on your logic
        learning_rate = ...
        return learning_rate

    def select_penalty(self, task_id):
        # your dynamic penalty selection logic here
        task_info = self.task_registry[task_id]
        available_penalties = task_info['available_penalties']
        # Perform penalty selection based on your logic
        penalty = ...
        return penalty

    def tune_hyperparameters(self, task_id):
        # your automatic hyperparameter tuning logic here
        task_info = self.task_registry[task_id]
        hyperparameters_grid = task_info['hyperparameters_grid']
        best_hyperparameters = None
        best_score = float('-inf')

        # Perform grid search over hyperparameters
        for params in ParameterGrid(hyperparameters_grid):
            # Update model, optimizer, and scheduler based on the selected hyperparameters
            model_name = self.select_model(task_id)
            learning_rate = self.select_learning_rate(task_id)
            penalty = self.select_penalty(task_id)

            # Run training and evaluate performance
            # ...
            # Compute score based on performance metrics
            score = ...

            if score > best_score:
                best_score = score
                best_hyperparameters = params

        self.best_hyperparameters[task_id] = best_hyperparameters

    def update(self, task_id, data_loader, **kwargs):
        if task_id not in self.models:
            self._configure_task(task_id)

        if 'optimizer_name' not in kwargs or 'penalty_name' not in kwargs:
            if task_id in self.best_hyperparameters:
                hyperparameters = self.best_hyperparameters[task_id]
                kwargs['optimizer_name'] = hyperparameters['model_name']
                kwargs['penalty_name'] = hyperparameters['penalty']
            else:
                self.tune_hyperparameters(task_id)
                hyperparameters = self.best_hyperparameters[task_id]
                kwargs['optimizer_name'] = hyperparameters['model_name']
                kwargs['penalty_name'] = hyperparameters['penalty']

        super().update(task_id, data_loader, **kwargs)
