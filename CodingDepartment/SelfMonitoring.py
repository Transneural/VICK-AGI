import os
import psutil
import time
import datetime
import platform
import socket
import subprocess
import json

class SelfMonitoring:
    def __init__(self, log_file):
        self.log_file = log_file
        self.interval = 60

    def log(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as file:
            file.write(f'{timestamp}: {message}\n')

    def check_memory(self):
        memory_info = psutil.virtual_memory()
        total_memory = memory_info.total
        available_memory = memory_info.available
        used_memory = memory_info.used
        memory_percent = memory_info.percent
        self.log(f'Total Memory: {self._format_bytes(total_memory)}')
        self.log(f'Available Memory: {self._format_bytes(available_memory)}')
        self.log(f'Used Memory: {self._format_bytes(used_memory)}')
        self.log(f'Memory Usage: {memory_percent}%')

    def check_cpu(self):
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        self.log(f'CPU Usage: {cpu_percent}%')
        self.log(f'CPU Count: {cpu_count}')

    def check_disk_usage(self, path):
        disk_usage = psutil.disk_usage(path)
        total_space = disk_usage.total
        used_space = disk_usage.used
        free_space = disk_usage.free
        disk_percent = disk_usage.percent
        self.log(f'Total Space: {self._format_bytes(total_space)}')
        self.log(f'Used Space: {self._format_bytes(used_space)}')
        self.log(f'Free Space: {self._format_bytes(free_space)}')
        self.log(f'Disk Usage: {disk_percent}%')

    def check_network_stats(self):
        network_stats = psutil.net_io_counters()
        sent_bytes = network_stats.bytes_sent
        received_bytes = network_stats.bytes_recv
        self.log(f'Sent Bytes: {self._format_bytes(sent_bytes)}')
        self.log(f'Received Bytes: {self._format_bytes(received_bytes)}')

    def check_system_info(self):
        system_info = {
            'Hostname': socket.gethostname(),
            'Operating System': f'{platform.system()} {platform.release()}',
            'Processor': platform.processor(),
            'Python Version': platform.python_version()
        }
        self.log(json.dumps(system_info, indent=4))

    def check_processes(self, limit=5):
        processes = []
        for process in psutil.process_iter(['pid', 'name', 'username']):
            processes.append(process.info)
        sorted_processes = sorted(processes, key=lambda x: x['pid'])
        self.log(f'Top {limit} Processes:')
        for process in sorted_processes[:limit]:
            self.log(f'PID: {process["pid"]} | Name: {process["name"]} | User: {process["username"]}')

    def _format_bytes(self, bytes):
        sizes = ["B", "KB", "MB", "GB", "TB"]
        index = 0
        while bytes >= 1024 and index < len(sizes) - 1:
            bytes /= 1024
            index += 1
        return f"{bytes:.2f} {sizes[index]}"

    def set_interval(self, interval):
        self.interval = interval

    def run(self):
        while True:
            self.check_memory()
            self.check_cpu()
            self.check_disk_usage('/')
            self.check_network_stats()
            self.check_system_info()
            self.check_processes()
            time.sleep(self.interval)

