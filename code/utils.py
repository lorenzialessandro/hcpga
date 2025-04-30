import numpy as np


class Task():
    """
    Represents a job task with timing and resource information.
    """

    def __init__(self, start_time: int, job_id: int, task_id: int, duration: int, machine: int):
        self.start_time = start_time
        self.job_id = job_id
        self.task_id = task_id
        self.duration = duration
        self.machine = machine
        self.end_time = start_time + duration

    def __repr__(self):
        return f"({self.start_time}, {self.job_id}, {self.task_id}, {self.duration})"


class Instance():
    """
    Instance of Job Shop Scheduling Problem.
    """

    def __init__(self, name, num_jobs, num_machines, tasks):
        self.name = name
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.tasks = tasks  # task = (machine_id, processing_time)

    def __str__(self):
        return f"{self.name}: {self.num_jobs} jobs, {self.num_machines} machines"

    def __repr__(self):
        return f"{self.name}: {self.num_jobs} jobs, {self.num_machines} machines"


def load_instance(filename):
    """
    Load and parse a JSP instance file
    """
    tasks = []
    num_machines = None

    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()
                 and not line.startswith('#')]

    # Parse first line for dimensions of the instance : num_jobs, num_machines
    num_jobs, num_machines = map(int, lines[0].split())

    # Parse each job's tasks
    for line in lines[1:]:
        if len(tasks) >= num_jobs:
            break

        numbers = list(map(int, line.split()))
        job_tasks = []
        i = 0

        while i < len(numbers) and numbers[i] >= 0:
            machine, duration = numbers[i:i+2]
            job_tasks.append((machine, duration))
            i += 2

        if job_tasks:
            tasks.append(job_tasks)

    # Create instance object
    instance = Instance(filename, num_jobs, num_machines, tasks)

    return instance
