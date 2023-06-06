import time

class DynamicImprovements:
    def __init__(self, self_improving_ai):
        self.ai = self_improving_ai
        self.interval = 60  # Default interval of 60 seconds
        self.running = False
        self.timeout = None
        self.callback = None

    def set_interval(self, seconds):
        self.interval = seconds

    def set_timeout(self, seconds):
        self.timeout = seconds

    def set_callback(self, callback):
        self.callback = callback

    def start(self):
        if self.running:
            print("DynamicImprovements is already running.")
            return

        self.running = True
        start_time = time.time()
        while self.running:
            self.ai.improve()
            if self.timeout and time.time() - start_time >= self.timeout:
                print("DynamicImprovements reached the timeout.")
                self.stop()
            time.sleep(self.interval)

        if self.callback:
            self.callback()

    def stop(self):
        self.running = False



