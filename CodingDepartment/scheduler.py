from sched import scheduler
import time

class Scheduler:
    def __init__(self, task, time="00:00"):
        self.task = task
        self.schedule = scheduler

    def schedule_daily_updates(self):
        # Schedule daily updates
        self.schedule.every().day.at(self.time).do(self.task)

        while True:
            self.schedule.run_pending()
            time.sleep(1)
