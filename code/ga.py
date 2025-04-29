import collections
import numpy as np
import random
from typing import List, Dict, Tuple
from inspyred import ec
import argparse
import time
import tracemalloc
from code.utils import *

class GA():
    """Class to solve Job Shop Scheduling Problem (JSP) using Genetic Algorithm (GA) with custom operators."""
    def __init__(self, instance, seed, hybrid):
        self.instance = instance
        self.prng = random.Random(seed)
        self.hybrid = hybrid # True if using CP-SAT solution as initial population

        # number of tasks in each job
        self.lengths_jobs = [len(job) for job in self.instance.tasks]
        self.num_tasks = sum(self.lengths_jobs)  # total number of tasks

        # Create a mapping of tasks to (job_id, task_id)
        self.machine_tasks = collections.defaultdict(list)
        for job_id, job in enumerate(self.instance.tasks):
            for task_id, (machine, _) in enumerate(job):
                self.machine_tasks[machine].append((job_id, task_id))

        self.best_schedule = None
        self.best_makespan = float('inf')

        self.max_time = 60  # default time limit for GA solver
        
        self.history = []           # track makespan over generations
        self.history_best = []      # track best makespan over generations

    # - 
    # Properties and methods
    
    def is_valid_chromosome(self, chromosome):
        """Validate if a chromosome represents a valid job_id sequence"""
        # A chromosome is a list of job_id
        
        # Check if all jobs are present
        if len(chromosome) != self.num_tasks:
            return False
        
        # Check if all jobs are present the correct number of times
        job_counts = collections.Counter(chromosome)
        for job_id, expected_count in enumerate(self.lengths_jobs):
            if job_counts[job_id] != expected_count:
                return False
            
        return True # valid chromosome
        
    def decode_to_task_sequence(self, chromosome): 
        """Convert chromosome to task sequence (job_id, task_id)"""
        task_sequences = []
        job_task_tracker = [0] * self.instance.num_jobs
        for job_id in chromosome:
            task_id = job_task_tracker[job_id]
            task_sequences.append((job_id, task_id))
            job_task_tracker[job_id] += 1
        return task_sequences

    def decoder(self, chromosome, args): 
        """Decode chromosome: convert job ID sequence to schedule and calculate makespan"""
        if not self.is_valid_chromosome(chromosome):
            chromosome = self.repair_chromosome(chromosome)
            
        # Convert job_id sequence to (job_id, task_id) sequence
        task_sequences = self.decode_to_task_sequence(chromosome)

        # Track end times
        job_ends = [0] * self.instance.num_jobs   
        machine_ends = [0] * self.instance.num_machines  
        schedule = collections.defaultdict(list)  

        # Process tasks in chromosome order
        for job_id, task_id in task_sequences:
            # Get machine and duration for this task
            machine, duration = self.instance.tasks[job_id][task_id]
            
            # The earliest start time is the maximum of:
            # 1. When the previous task of this job finishes
            # 2. When the machine becomes available
            start_time = max(job_ends[job_id], machine_ends[machine])

            # Create task object
            task = Task(
                start_time=start_time,
                job_id=job_id,
                task_id=task_id,
                duration=duration,
                machine=machine
            )

            # Update schedule and end times
            schedule[machine].append(task)
            job_ends[job_id] = start_time + duration
            machine_ends[machine] = start_time + duration

        makespan = max(job_ends)
        return schedule, makespan

    def validate_schedule(self, schedule): 
        """Validate that the schedule respects all constraints"""
        
        # Check for machine conflicts
        for machine, tasks in schedule.items():
            sorted_tasks = sorted(tasks, key=lambda x: x.start_time)
            for i in range(len(sorted_tasks) - 1):
                current = sorted_tasks[i]
                next_task = sorted_tasks[i + 1]
                if current.start_time + current.duration > next_task.start_time:
                    return False
                    
        # Check job precedence
        job_task_times = collections.defaultdict(list)
        for machine, tasks in schedule.items():
            for task in tasks:
                job_task_times[task.job_id].append((task.task_id, task.start_time, task.duration))
                
        for job_id, task_times in job_task_times.items():
            sorted_tasks = sorted(task_times, key=lambda x: x[0])  # Sort by task index
            for i in range(len(sorted_tasks) - 1):
                current = sorted_tasks[i]
                next_task = sorted_tasks[i + 1]
                if current[1] + current[2] > next_task[1]:  # start + duration > next_start
                    return False
                    
        return True

    def repair_chromosome(self, chromosome): 
        """Repair invalid chromosome to make it valid"""
        # Count occurrences of each job_id
        job_positions = collections.defaultdict(list)

        # Build a valid chromosome template
        valid_chromosome = []

        # Create a list of all tasks that need to be scheduled
        all_tasks = []
        for job_id in range(self.instance.num_jobs):
            for task_id in range(self.lengths_jobs[job_id]):
                all_tasks.append((job_id, task_id))

        # Add existing jobs as long as they don't exceed their count
        job_added = [0] * self.instance.num_jobs # Number of tasks added for each job
        for job_id in chromosome:
            if job_added[job_id] < self.lengths_jobs[job_id] and job_id < self.instance.num_jobs: # check if job_id is valid
                valid_chromosome.append(job_id)
                job_added[job_id] += 1
        
        # Add missing jobs
        for job_id in range(self.instance.num_jobs):
            remaining = self.lengths_jobs[job_id] - job_added[job_id]
            valid_chromosome.extend([job_id] * remaining)
        
        # Length should match original chromosome
        assert len(valid_chromosome) == len(chromosome)
        
        # Check if the repaired chromosome is valid
        assert self.is_valid_chromosome(valid_chromosome)
        
        return valid_chromosome

    # -
    # GA methods
    
    def generator_chromosome(self, random, args): 
        """Generate a valid chromosome (list of job_id)"""
        chromosome = []
        for job_id, length in enumerate(self.lengths_jobs):
            # Add each job's tasks in order
            chromosome.extend([job_id] * length)
        
        # Shuffle the chromosome
        random.shuffle(chromosome)
        return chromosome            

    def generator(self, random, args): 
        """Generate a new chromosome for the population"""
        # Chromosome if flag of hybrid is True
        if self.hybrid and args['initial_population'] is not None:
            # Use CP-SAT output as initial population
            return args['initial_population'].pop()
        else:
            # Generate random chromosome
            return self.generator_chromosome(random, args)

    def evaluator(self, candidates, args):
        """Evaluate chromosome"""
        fitness = []
        for chromosome in candidates:
            # Repair invalid chromosomes
            if not self.is_valid_chromosome(chromosome):
                chromosome = self.repair_chromosome(chromosome)

            _, makespan = self.decoder(chromosome, args)
            fitness.append(makespan)

        return fitness

    def observer(self, population, num_generations, num_evaluations, args):
        """Observer function to track best solution"""
        best = min(population, key=lambda l: l.fitness)
        schedule, makespan = self.decoder(best.candidate, args)
        # print(f"Generation {num_generations}: makespan = {makespan}")
        if self.validate_schedule(schedule):
            if makespan < self.best_makespan:
                self.best_makespan = makespan
                self.best_schedule = schedule

        if num_generations % 100 == 0:
            self.history_best.append(self.best_makespan) # track best makespan over generations on 100th generation
            # print(
                # f"Generation {num_generations}: Best makespan = {self.best_makespan}")
            
        self.history.append(makespan)

    def solve(self, args):
        """Solve JSP instance using GA"""
        
        # Track time
        start_time_t = time.time()

        ga = ec.GA(random=self.prng)
        ga.observer = self.observer
        ga.terminator = self.time_and_patient_terminator        # Custom terminator that combines time limit and patience
        ga.replacer = ec.replacers.generational_replacement 
        ga.variator = [self.multi_custom_crossover, self.custom_mutation]
        ga.selector = ec.selectors.tournament_selection

        # Use initial population if provided
        initial_population = None
        if args is not None and 'initial_population' in args:
            # print("Using initial population from args")
            initial_population = args['initial_population']

        # Track memory
        tracemalloc.start()
        # Run GA solver
        final_pop = ga.evolve(
            generator=self.generator,
            evaluator=self.evaluator,
            pop_size=100,
            maximize=False,
            bounder=None, 
            max_generations=1000,
            mutation_rate=0.7,
            crossover_rate=0.6,
            num_selected=70,
            initial_population=initial_population,
            max_time=self.max_time,
            start_time=time.time(),
            num_elites = 10,
            tournament_size=5,
            patience = 3
        )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        end_time_t = time.time()
        ga_time = end_time_t - start_time_t

        return self.best_schedule, self.best_makespan, ga_time, peak

    # - 
    # Opertators
    
    def multi_custom_crossover(self, random, candidates, args):
        """Job-preserving multi-point crossover operator for job ID sequences"""
        children = []
        
        # Get number of crossover points (can be configured in args)
        num_crossover_points = args.get('num_crossover_points', 3)
        
        for i in range(0, len(candidates) - 1, 2):
            # Select parents
            parent1 = candidates[i]
            parent2 = candidates[i + 1]
            
            # Create child chromosomes
            child1 = [None] * len(parent1)
            child2 = [None] * len(parent2)
            
            # Select multiple random crossover points and sort them
            crossover_points = sorted([random.randint(1, len(parent1) - 2) for _ in range(num_crossover_points)])
            
            # Add the end point to simplify segment handling
            crossover_points.append(len(parent1))
            
            # Copy segments from parents alternately
            start = 0
            parent_switch = False  # False: use parent1 for child1, True: use parent2 for child1
            
            for point in crossover_points:
                if not parent_switch:
                    # Use parent1 for child1, parent2 for child2
                    child1[start:point] = parent1[start:point]
                    child2[start:point] = parent2[start:point]
                else:
                    # Use parent2 for child1, parent1 for child2
                    child1[start:point] = parent2[start:point]
                    child2[start:point] = parent1[start:point]
                
                # Switch for next segment
                parent_switch = not parent_switch
                start = point
            
            # Count jobs already added to each child
            child1_job_counts = collections.Counter(job for job in child1 if job is not None)
            child2_job_counts = collections.Counter(job for job in child2 if job is not None)
            
            # Identify positions that still need to be filled
            child1_empty_positions = [idx for idx, val in enumerate(child1) if val is None]
            child2_empty_positions = [idx for idx, val in enumerate(child2) if val is None]
            
            # Fill in the remaining positions while respecting job counts
            # Fill child1 from parent2
            idx1 = 0
            for job_id in parent2:
                # Check if we can still add this job (haven't reached its limit)
                if child1_job_counts.get(job_id, 0) < self.lengths_jobs[job_id] and idx1 < len(child1_empty_positions):
                    position = child1_empty_positions[idx1]
                    child1[position] = job_id
                    child1_job_counts[job_id] += 1
                    idx1 += 1
                    
            # Fill child2 from parent1
            idx2 = 0
            for job_id in parent1:
                # Check if we can still add this job (haven't reached its limit)
                if child2_job_counts.get(job_id, 0) < self.lengths_jobs[job_id] and idx2 < len(child2_empty_positions):
                    position = child2_empty_positions[idx2]
                    child2[position] = job_id
                    child2_job_counts[job_id] += 1
                    idx2 += 1
            
            # Ensure chromosomes are valid (repair if needed)
            if not self.is_valid_chromosome(child1):
                child1 = self.repair_chromosome(child1)
            if not self.is_valid_chromosome(child2):
                child2 = self.repair_chromosome(child2)
                
            children.append(child1)
            children.append(child2)
        
        # Add the last candidate if there's an odd number
        if len(candidates) % 2 == 1:
            children.append(candidates[-1])
            
        return children

    def custom_mutation(self, random, candidates, args):
        """Custom mutation operator: Swap Mutation"""
        mutants = []
        for chromosome in candidates:
            mutant = chromosome.copy()

            if random.random() < args['mutation_rate']:
                # Make a random number of mutation swaps (1 to 5)
                num_swaps = random.randint(1, 5)
                for _ in range(num_swaps):
                    # Choose two random positions to swap
                    pos1, pos2 = random.sample(range(len(mutant)), 2)
                    mutant[pos1], mutant[pos2] = mutant[pos2], mutant[pos1]

                    # Undo the swap if invalid 
                    if not self.is_valid_chromosome(mutant):
                        mutant[pos1], mutant[pos2] = mutant[pos2], mutant[pos1]
                        
            mutants.append(mutant)

        return mutants

    # Custom terminator that combines time limit and patience
    def time_and_patient_terminator(self, population, num_generations, num_evaluations, args):
        """Terminate when either max time is reached or no improvement for patience generations"""
        time_elapsed = time.time() - args['start_time']
        if time_elapsed >= args['max_time']:
            return True
        
        # Check for patience termination
        if len(self.history_best) >= args['patience']:
            if all (i == self.history_best[-1] for i in self.history_best[-args['patience']:]):
                return True  
        return False
