import datetime
import os
import subprocess
import logging
import time
import requests
import json

from auto_code_update.automated_deploymen_modules.logging_manager import setup_logging

class ServiceManager:
    def __init__(self, service_name):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)
        setup_logging('config.json')

    def log_message(self, level, msg):
        if level.lower() == 'debug':
            self.logger.debug(msg)
        elif level.lower() == 'info':
            self.logger.info(msg)
        elif level.lower() == 'warning':
            self.logger.warning(msg)
        elif level.lower() == 'error':
            self.logger.error(msg)
        elif level.lower() == 'critical':
            self.logger.critical(msg)
        else:
            self.logger.info(msg)

    def start_service(self):
        self.log_message('info', f'Starting service: {self.service_name}')
        self.run_command(['sudo', 'systemctl', 'start', self.service_name])

    def stop_service(self):
        self.log_message('info', f'Stopping service: {self.service_name}')
        self.run_command(['sudo', 'systemctl', 'stop', self.service_name])

    def restart_service(self):
        self.log_message('info', f'Restarting service: {self.service_name}')
        self.run_command(['sudo', 'systemctl', 'restart', self.service_name])

    def status_service(self):
        self.log_message('info', f'Checking status of service: {self.service_name}')
        self.run_command(['sudo', 'systemctl', 'status', self.service_name])

    def enable_service(self):
        self.log_message('info', f'Enabling service: {self.service_name}')
        self.run_command(['sudo', 'systemctl', 'enable', self.service_name])

    def disable_service(self):
        self.log_message('info', f'Disabling service: {self.service_name}')
        self.run_command(['sudo', 'systemctl', 'disable', self.service_name])

    def reload_service(self):
        self.log_message('info', f'Reloading service: {self.service_name}')
        self.run_command(['sudo', 'systemctl', 'reload', self.service_name])

    def run_command(self, command):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            self.log_message('error', f'Command execution failed: {result.stderr}')
            raise subprocess.CalledProcessError(result.returncode, command)
        return result.stdout

    def deploy_service(self, deployment_script):
        self.log_message('info', f'Deploying service: {self.service_name}')

        # Execute the deployment script or commands specific to the service
        try:
            subprocess.run(deployment_script, shell=True, check=True)
            self.log_message('info', 'Service deployed successfully.')
        except subprocess.CalledProcessError as e:
            self.log_message('error', f'Deployment failed: {e}')
            raise e

    def scale_service(self, num_instances, scaling_script=None):
        self.log_message('info', f'Scaling service: {self.service_name} to {num_instances} instances')
        # Update the configuration or infrastructure to scale the service
        if scaling_script:
            self.run_command([scaling_script, str(num_instances)])
        else:
            self.log_message('warning', 'Scaling script not provided. Unable to scale the service.')

    def monitor_service(self):
     self.log_message('info', f'Monitoring service: {self.service_name}')
    
    # Retrieve service metrics
     metrics = self.retrieve_service_metrics()

    # Check health checks
     health_checks = self.check_health_checks()

    # Integrate with monitoring tools
     self.integrate_with_monitoring_tools(metrics, health_checks)

    # Perform additional monitoring logic
     if metrics:
        # Example: Log the metrics
        for metric_name, metric_value in metrics.items():
            self.log_message('info', f'Metric - {metric_name}: {metric_value}')

     if health_checks:
        # Example: Log the health check results
        for check_name, check_result in health_checks.items():
            self.log_message('info', f'Health Check - {check_name}: {check_result}')

        # Example: Trigger actions based on health check results
        if not health_checks['service_available']:
            self.log_message('warning', 'Service is not available')

            # Perform recovery actions
            self.auto_recover_service()
     else:
        self.log_message('warning', 'No health check results available')

    # Add more monitoring logic as needed

     self.log_message('info', 'Monitoring complete')

    def retrieve_service_metrics(self):
        metrics = {}
        try:
            # Define the Prometheus API endpoint
            prometheus_api = 'http://localhost:9090/api/v1/query'

            # Define the metrics that you want to retrieve. You can add more queries as per your requirement
            queries = ['up', 'process_cpu_seconds_total', 'process_memory_bytes']

            for query in queries:
                # Send a request to the Prometheus API
                response = requests.get(prometheus_api, params={'query': query})
                response.raise_for_status()  # Raise exception if the request failed

                # Parse the JSON response
                results = response.json()['data']['result']

                # Loop over the results and store the metrics
                for result in results:
                    metric_name = result['metric'].get('__name__', '')
                    metric_value = float(result['value'][1])

                    # Store the metric in the metrics dictionary
                    metrics[metric_name] = metric_value

        except Exception as e:
            self.logger.error(f"Error occurred while retrieving service metrics: {e}")
            # If any error occurs, raise it further
            raise

        # Return the metrics
        return metrics

    def check_health_checks(self):
        health_checks = {'service_available': False, 'response_time': None, 'dependencies': {}}

        try:
            # Send a request to the service's health check endpoint
            start_time = datetime.now()
            response = requests.get('http://localhost:8080/health')
            end_time = datetime.now()

            # Calculate the response time (in seconds)
            response_time = (end_time - start_time).total_seconds()

            # Parse the JSON response
            health_data = response.json()

            # Update the health checks
            health_checks['service_available'] = True
            health_checks['response_time'] = response_time

            # Check the status of the dependencies
            for dependency, status in health_data.get('dependencies', {}).items():
                health_checks['dependencies'][dependency] = (status == 'healthy')

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")

        return health_checks

    def integrate_with_monitoring_tools(self, metrics, health_checks):
        # Prepare the data to send
        data = {
            'metrics': metrics,
            'health_checks': health_checks,
        }

        # Convert the data to JSON
        json_data = json.dumps(data)

        # Send the data to the monitoring tool's API
        url = 'http://monitoringtool.com/api/v1/data'  # replace with your actual API URL
        headers = {'Content-Type': 'application/json'}

        try:
            response = requests.post(url, headers=headers, data=json_data)
            response.raise_for_status()  # raises an exception if the response status is not successful
            self.logger.info('Successfully sent data to the monitoring tool.')

        except requests.exceptions.HTTPError as e:
            self.logger.error(f'Failed to send data to the monitoring tool: {e}')

    def auto_recover_service(self):
        while True:
            if not self.check_service_health():
                self.log_message('info', 'Attempting to recover service')
                self.restart_service()
                time.sleep(10)
                if not self.check_service_health():
                    self.log_message('error', f"Service {self.service_name} is down. Tried to restart but it's still not working.")
                    self.send_alert(f"Service {self.service_name} is down. Tried to restart but it's still not working.")
            time.sleep(60)  # Sleep for 60 seconds between health checks

    def check_service_health(self):
        # Retrieve metrics
        metrics = self.retrieve_service_metrics()
        if metrics is None or 'up' not in metrics['data']['result'][0]['value'][1]:
            self.logger.error("Unable to retrieve service metrics.")
            return False

        # Perform health checks
        health_checks = self.check_health_checks()
        if health_checks is None or not health_checks['status']:
            self.logger.error("Health check failed.")
            return False

        # Integrate with monitoring tools
        try:
            self.integrate_with_monitoring_tools(metrics, health_checks)
        except Exception as e:
            self.logger.error(f"Failed to integrate with monitoring tools: {e}")
            return False

        self.logger.info("Service is healthy.")
        return True

    def perform_recovery_actions(self):
        try:
            # In this example, the recovery action is restarting the service.
            # The specific command will depend on your service and system.
            command = f'systemctl restart {self.service_name}'
            subprocess.run(command, check=True, shell=True)
            self.logger.info("Service restart command issued.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to restart service: {e}")
            return False

        return True

    def read_configuration_file(self, config_file):
        try:
            with open(config_file, 'r') as f:
                new_config = json.load(f)
            self.logger.info("New configuration file successfully read.")
            return new_config
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse configuration file: {e}")
            return None
        except FileNotFoundError as e:
            self.logger.error(f"Configuration file not found: {e}")
            return None

    def apply_configuration_changes(self, new_config):
        self.logger.info(f'Applying configuration changes for service: {self.service_name}')

        # The changes may require the service to be stopped
        self.stop_service()

        # Update the service configuration based on the new configuration
        for key, value in new_config.items():
            if key in self.service_config:
                self.service_config[key] = value
            else:
                self.logger.warning(f'Invalid configuration key: {key}. Ignoring the change.')

        # Here you should write the new config to the actual config file
        # Or apply it directly if your service supports it

        try:
            with open(self.config_file_path, 'w') as config_file:
                json.dump(self.service_config, config_file)
            self.logger.info(f'Successfully wrote changes to config file at: {self.config_file_path}')
        except Exception as e:
            self.logger.error(f'Error occurred while writing to config file: {e}')
            return False

        # Start the service back up after the changes
        self.start_service()

        return True

    def validate_service_configuration(self):
        self.logger.info(f'Validating configuration for service: {self.service_name}')

        # Load the service configuration from file
        self.service_config = self.read_configuration_file(self.config_file_path)

        # Check if the configuration is valid and meets the required criteria
        required_fields = ['field1', 'field2', 'field3']  # List of required configuration fields

        for field in required_fields:
            if field not in self.service_config:
                self.logger.error(f'Missing required configuration field: {field}')
                return False

        # Additional validation checks
        if self.service_config.get('field1') <= 0:
            self.logger.error('Invalid value for field1. It should be greater than 0.')
            return False

        if not isinstance(self.service_config.get('field2'), str):
            self.logger.error('Invalid value for field2. It should be a string.')
            return False

        # TODO: Add more validation checks as needed
        self.logger.info('Service configuration validated successfully.')
        return True

    def backup_service_data(self, backup_dir):
        self.logger.info(f'Backing up service data: {self.service_name}')

        # Assuming the service data is stored in a directory called "data_dir"
        data_dir = self.data_dir  # Replace with the actual data directory path

        # Check if the data directory exists
        if not os.path.isdir(data_dir):
            self.logger.error(f'Data directory does not exist: {data_dir}')
            return False

        # Generate a timestamp for the backup file name
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        # Create a backup file name with the service name and timestamp
        backup_file = f'{self.service_name}_backup_{timestamp}.tar.gz'

        # Create the full backup file path
        backup_path = os.path.join(backup_dir, backup_file)

        try:
            # Create a tarball of the data directory and save it to the backup path
            subprocess.run(['tar', '-czf', backup_path, data_dir], check=True)
            self.logger.info(f'Service data backup created: {backup_path}')
        except Exception as e:
            self.logger.error(f'Error during backup: {e}')
            return False

        return True

    def restore_service_data(self, backup_dir):
        self.logger.info(f'Restoring service data: {self.service_name}')

        # Assuming the service data will be restored to a directory called "data_dir"
        data_dir = self.data_dir  # Replace with the actual data directory path

        # Find the latest backup file in the backup directory
        backup_files = [file for file in os.listdir(backup_dir) if file.endswith('.tar.gz')]
        if not backup_files:
            self.logger.error(f'No backup files found in directory: {backup_dir}')
            return False

        latest_backup_file = max(backup_files, key=lambda file: os.path.getctime(os.path.join(backup_dir, file)))
        backup_path = os.path.join(backup_dir, latest_backup_file)

        try:
            # Restore the backup file to the data directory
            subprocess.run(['tar', '-xzf', backup_path, '-C', data_dir], check=True)
            self.logger.info(f'Service data restored from backup: {backup_path}')
        except Exception as e:
            self.logger.error(f'Error during data restore: {e}')
            return False

        return True
    
    def execute_custom_command(self, command):
        self.logger.info(f'Executing custom command for service: {self.service_name}')

        # Check if command is empty or None
        if not command:
            self.logger.error('Cannot execute empty or None command.')
            return False

        try:
            # Execute the command
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.logger.info('Custom command executed successfully.')
            self.logger.info(f'Command output: {result.stdout.decode()}')
        except subprocess.CalledProcessError as e:
            self.logger.error(f'Failed to execute custom command: {e.stderr.decode()}')
            return False

        return True
