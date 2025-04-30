import collections
import numpy as np
from typing import List, Dict, Tuple
from ortools.sat.python import cp_model
import time
import psutil
from code.utils import *


class Limiter(cp_model.CpSolverSolutionCallback):
    """Class to collect solutions from the CP-SAT solver."""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__solution_list = []

    def on_solution_callback(self) -> None:
        self.__solution_count += 1
        self.__solution_list.append([self.Value(v) for v in self.__variables])

    def solution_list(self):
        return self.__solution_list

    def solution_count(self):
        return self.__solution_count


class CPSATC:
    """
    Class to solve Job Shop Scheduling Problem (JSP) using OR-Tools CP-SAT solver with solution collection.
    """

    def __init__(self, instance, time_limit):
        self.instance = instance
        self.time_limit = time_limit
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()

    def solve(self):
        """Solve JSP instance using OR-Tools (CP-SAT)"""

        # Track time
        start_time_t = time.time()

        # Calculate reasonable horizon
        horizon = sum(sum(task[1] for task in job)
                      for job in self.instance.tasks)

        # Create variables
        starts = {}         # (job_id, task_id) -> start_time_var
        ends = {}           # (job_id, task_id) -> end_time_var
        intervals = {}      # (job_id, task_id) -> interval_var

        # Create job intervals and add to the corresponding machine lists
        for job_id, job in enumerate(self.instance.tasks):
            for task_id, task in enumerate(job):
                machine, duration = task
                suffix = f"_{job_id}_{task_id}"

                # Create start time variable
                start = self.model.NewIntVar(0, horizon, f"start{suffix}")
                end = self.model.NewIntVar(0, horizon, f"end{suffix}")
                interval = self.model.NewIntervalVar(
                    start, duration, end, f"interval{suffix}")

                starts[job_id, task_id] = start
                ends[job_id, task_id] = end
                intervals[job_id, task_id] = interval

        # Add precedence constraints within each job
        for job_id, job in enumerate(self.instance.tasks):
            for task_id in range(len(job) - 1):
                self.model.Add(ends[job_id, task_id] <=
                               starts[job_id, task_id + 1])

        # Add no-overlap constraints for machines
        machine_to_intervals = collections.defaultdict(list)
        for job_id, job in enumerate(self.instance.tasks):
            for task_id, task in enumerate(job):
                machine, _ = task
                machine_to_intervals[machine].append(
                    intervals[job_id, task_id])

        for machine in range(self.instance.num_machines):
            # Only add constraint if machine has tasks
            if machine_to_intervals[machine]:
                self.model.AddNoOverlap(machine_to_intervals[machine])

        # Makespan objective
        makespan = self.model.NewIntVar(0, horizon, 'makespan')
        self.model.AddMaxEquality(
            makespan,
            [ends[job_id, len(job)-1]
             for job_id, job in enumerate(self.instance.tasks)]
        )
        self.model.Minimize(makespan)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        limiter = Limiter(starts.values())
        status = solver.Solve(self.model, limiter)

        # Check if solution found
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return None, None, None, None, None, None

        # Collect all the schedules
        schedules = []
        makespans = []

        for solution in limiter.solution_list():
            schedule = {}
            makespan_value = solver.Value(makespan)

            for job_id, job in enumerate(self.instance.tasks):
                for task_id, (machine, duration) in enumerate(job):
                    start_time = solution[job_id * len(job) + task_id]
                    task = Task(start_time, job_id, task_id, duration, machine)

                    if machine not in schedule:
                        schedule[machine] = []
                    schedule[machine].append(task)

            # Sort tasks on each machine by start time
            for machine in schedule:
                schedule[machine].sort(key=lambda x: x.start_time)

            schedules.append(schedule)
            makespans.append(makespan_value)

        # Extract Solution
        schedule = {}
        makespan_value = solver.Value(makespan)

        for job_id, job in enumerate(self.instance.tasks):
            for task_id, (machine, duration) in enumerate(job):
                start_time = solver.Value(starts[job_id, task_id])
                task = Task(start_time, job_id, task_id, duration, machine)

                if machine not in schedule:
                    schedule[machine] = []
                schedule[machine].append(task)

        # Sort tasks on each machine by start time
        for machine in schedule:
            schedule[machine].sort(key=lambda x: x.start_time)

        end_time_t = time.time()

        cp_time = end_time_t - start_time_t

        return schedule, makespan_value, solver, status, cp_time, schedules
