import collections
import numpy as np
from typing import List, Dict, Tuple
import argparse
import random
import psutil

from code.cpsatc import *
from code.ga import *
from code.utils import *


class HCPGA:
    """
    Hybrid Constrained Programming with Genetic Algorithm for Job Shop Scheduling Problem.
    """

    def __init__(self, instance, seed, time_budget=2000,
                 split=0.3, num_copies=1, num_random=30, ga_args=None):
        """
        Initialize HCPGA solver with the given instance and parameters.

        args:
            instance : Instance, the JSP instance to solve
            seed : int, random seed for reproducibility
            time_budget : int, total time budget for the solver (in seconds)
            split : float, fraction of time allocated to CP-SAT solver (default 0.3)
            num_copies : int, number of exact copies of the base chromosome to add to the population (default 1)
            num_random : int, number of random chromosomes to add to the population (default 30)
            ga_args : dict, additional arguments for GA solver (default None)
        """
        self.instance = instance
        self.seed = seed
        self.time_budget = time_budget
        self.split = split
        self.num_copies = num_copies
        self.num_random = num_random
        self.ga_args = ga_args

        # CP-SAT solver
        # allocate time for CP-SAT solver (default 30% of the total time)
        self.cpsat_solver = CPSATC(instance, self.time_budget * self.split)
        self.cpsat_solver.solver.parameters.random_seed = seed

        # GA solver
        # set hybrid to True
        self.ga_solver = GA(instance, seed=seed, hybrid=True)
        # allocate remaining time for GA solver (default 70% of the total time)
        self.ga_solver.max_time = self.time_budget * (1 - self.split)

    def create_initial_population(self, base_chromosomes, pop_size=50,
                                  num_copies=1,  num_random=1):
        """
        Create initial population for GA solver.

        args:
            base_chromosomes : list, list of the base chromosomes to copy in the initial population from CP-SAT solver solutions
            pop_size : int, total population size (given by args of GA solver)
            num_copies : int, number of exact copies of the base chromosome to add to the population
            num_random : int, number of random chromosomes to add to the population
        """
        initial_population = []

        # Add exact copies of the original solution
        for _ in range(num_copies):
            for base_chromosome in base_chromosomes:
                initial_population.append(base_chromosome.copy())

        # Add random chromosomes using generator
        for _ in range(num_random):
            initial_population.append(
                self.ga_solver.generator_chromosome(self.ga_solver.prng, None))

        # Add remaining population applying mutation to the base chromosome
        remaining = int((pop_size - len(base_chromosomes) -
                        num_random) / len(base_chromosomes))

        for i in range(remaining):
            for base_chromosome in base_chromosomes:
                mutated = base_chromosome.copy()

                # Mutation attempts for each chromosome (1 to 5)
                attempts = random.randint(1, 5)
                for _ in range(attempts):
                    # Choose two random positions to swap
                    pos1 = random.randint(0, len(mutated) - 1)
                    pos2 = random.randint(0, len(mutated) - 1)

                    # Swap
                    mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]

                    # Check if still valid
                    if self.ga_solver.is_valid_chromosome(mutated):
                        break
                    else:
                        # Undo the swap and try again
                        mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]

                # Add to population even if no valid mutation was found
                initial_population.append(mutated)

        # check if the population size is less than the required size
        if len(initial_population) < pop_size:
            remaining = pop_size - len(initial_population)
            for _ in range(remaining):
                initial_population.append(
                    self.ga_solver.generator_chromosome(self.ga_solver.prng, None))

        # resize the population to the required size
        initial_population = initial_population[:pop_size]

        return initial_population

    def solve(self):
        """
        Solve JSP instance using HCPGA approach.

        Steps:
            - (1) Solve using CP-SAT solver
            - Collect chromosomes from the solution(s) found by CP-SAT solver
            - Create initial population for GA solver based on the chromosomes collected
            - (2) Solve using GA solver
            - Return the best solution found by GA solver
        """
        # -
        # 1. Solve using CP-SAT solver
        cpsat_schedule, cpsat_makespan, cpsat_solver, cpsat_status, cpsat_time, cpsat_schedules = self.cpsat_solver.solve()

        # If the solution found by CP-SAT solver is optimal, return it (no need to use GA solver)
        if cpsat_status == cp_model.OPTIMAL:
            return cpsat_schedule, cpsat_makespan, cpsat_makespan, cpsat_time, 0

        if cpsat_status == cp_model.UNKNOWN:  # If CP-SAT solver could not find a solution, return None
            return None, 0, 0, 0, 0

        # -
        # Collect chromosome(s) from the solution(s) found by CP-SAT solver
        chromosomes = []
        # For each solution found by CP-SAT solver, create a chromosome
        for schedule in cpsat_schedules:
            all_tasks = []
            for machine in schedule.values():
                all_tasks.extend(machine)
            all_tasks.sort(key=lambda x: x.start_time)
            chromosome = [task.job_id for task in all_tasks]
            chromosomes.append(chromosome)

        # -
        # Create initial population for GA solver
        initial_population = self.create_initial_population(
            chromosomes, pop_size=self.ga_args["pop_size"], num_copies=self.num_copies, num_random=self.num_random)
        # Add the initial population to the ga_args
        self.ga_args["initial_population"] = initial_population

        # -
        # 2. Solve using GA solver
        ga_schedule, ga_makespan, ga_time, ga_memory = self.ga_solver.solve(
            args=self.ga_args)

        # -
        # Results
        total_time = cpsat_time + ga_time

        return ga_schedule, ga_makespan, cpsat_makespan, total_time, ga_memory
