import time             
import argparse
import random
import csv
import os
import logging
import numpy as np
from datetime import datetime
from contextlib import contextmanager
import concurrent.futures
import threading
import psutil  

from code.hcpga import *
from code.ga import *
from code.cpsat import *

# Ensure log directory exists
os.makedirs('log', exist_ok=True)
# Set up logging with thread safety
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)-13s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'log/solver_log_{datetime.now().strftime("%Y%m%d-%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Thread-safe CSV writer
csv_lock = threading.Lock()

@contextmanager
def timer():
    """Context manager for timing code execution."""
    start = time.time()
    yield
    end = time.time()
    return end - start

@contextmanager
def memory_tracker():
    """Context manager for tracking peak memory usage during execution."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    peak_memory = [initial_memory]
    stop_monitoring = threading.Event()
    
    def memory_monitor():
        while not stop_monitoring.is_set():
            try:
                current_memory = process.memory_info().rss
                peak_memory[0] = max(peak_memory[0], current_memory)
                time.sleep(0.1)  # Check every 100ms
            except:
                pass
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=memory_monitor)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        yield lambda: peak_memory[0] - initial_memory
    finally:
        stop_monitoring.set()
        monitor_thread.join(timeout=1.0)

# -
# Experiment functions
def run_and_log_experiment(instance, args, run_id, seed, first_run=False):
    """Run all solvers on an instance and log results to CSV file."""
    try:
        logger.info(f"Running experiment for instance {instance.name} (Run {run_id})")
        # define the args for the GA solver
        ga_args = {
            'pop_size': args.pop_size,
            'max_generations': args.max_generations,
            'mutation_rate': args.mutation_rate,
            'crossover_rate': args.crossover_rate,
            'num_selected': args.num_selected,
            'num_elites': args.num_elites,
            'tournament_size': args.tournament_size,
            'patience': args.patience
        }
        
        # - 
        # 1. Run CP-SAT solver to find the optimal solution
        logger.info(f"[{instance.name}] [Run {run_id}] Running CP-SAT to find optimal solution...")
        cpsat_make, cpsat_time, cpsat_memory, cpsat_status = run_cpsat(instance, seed, args.max_time_budget)
        logger.info(f"[{instance.name}] [Run {run_id}] CP-SAT completed: makespan={cpsat_make}, time={cpsat_time:.2f}s, memory={cpsat_memory / 1024 / 1024:.2f}MB")

        # 2. Run HCPGA solver with collector
        logger.info(f"[{instance.name}] [Run {run_id}] Running Hybrid solver with collector...")
        hcpga_make, hcpga_time, hcpga_memory = run_hcpga(instance, seed, cpsat_time, args.split, args.num_copies, args.num_random, ga_args)
        logger.info(f"[{instance.name}] [Run {run_id}] HCPGA completed: makespan={hcpga_make}, time={hcpga_time:.2f}s, memory={hcpga_memory / 1024 / 1024:.2f}MB")
        
        # 3. Run GA solver
        logger.info(f"[{instance.name}] [Run {run_id}] Running GA solver...")
        ga_make, ga_time, ga_memory = run_ga(instance, seed, args.max_time_budget, ga_args)
        logger.info(f"[{instance.name}] [Run {run_id}] GA completed: makespan={ga_make}, time={ga_time:.2f}s, memory={ga_memory / 1024 / 1024:.2f}MB")
        
        # -
        # Thread-safe CSV writing
        with csv_lock:
            with open(args.csv_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write results
                writer.writerow([
                    instance.name, run_id, seed,
                    cpsat_make, cpsat_time, cpsat_memory,
                    hcpga_make, hcpga_time, hcpga_memory,
                    ga_make, ga_time, ga_memory
                ])
        
        logger.info(f"[{instance.name}] [Run {run_id}] Experiment completed and results saved")
        return True
    
    except Exception as e:
        logger.error(f"Error running experiment for {instance.name} (Run {run_id}): {str(e)}")
        
        # Log error to CSV with thread safety
        with csv_lock:
            with open(args.csv_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([instance.name, run_id, seed, "ERROR", str(e), "", "", "", "", "", "", ""]) 
        
        return False

# Runner functions 
def run_hcpga(instance, seed, time_budget, split, num_copies, num_random, ga_args):
    """Run HCPGA solver with collector."""
    try:
        hybrid_solver = HCPGA(instance, seed=seed, time_budget=time_budget, split=split, num_copies=num_copies, num_random=num_random, ga_args=ga_args)
        
        with memory_tracker() as get_peak_usage:
            schedule, makespan_ga, makespan_icp, tot_time, ga_memory = hybrid_solver.solve()
            memory_used = get_peak_usage()
            
        return makespan_ga, tot_time, (memory_used + ga_memory)
    except Exception as e:
        logger.error(f"Error in hybrid collector for {instance.name}: {str(e)}")
        return float('inf'), float('inf'), 0, 
        

def run_cpsat(instance, seed, max_time_budget):
    """Run CP-SAT solver."""
    try:
        cp_solver = CPSAT(instance, max_time_budget)
        cp_solver.solver.parameters.random_seed = seed
        
        with memory_tracker() as get_peak_usage:
            schedule, makespan, solver, status, cp_time = cp_solver.solve()
            memory_used = get_peak_usage()
            
        return makespan, cp_time, memory_used, status
    except Exception as e:
        logger.error(f"Error in CP-SAT solver for {instance.name}: {str(e)}")
        return float('inf'), 0, 0, "ERROR"
    
def run_ga(instance, seed, time_budget, ga_args):
    """Run GA solver."""
    try:
        ga_solver = GA(instance, seed=seed, hybrid=False)
        ga_solver.max_time = time_budget
        schedule, makespan, tot_time, memory_used = ga_solver.solve(args=ga_args)
            
        return makespan, tot_time, memory_used
    except Exception as e:
        logger.error(f"Error in GA solver for {instance.name}: {str(e)}")
        return float('inf'), 0, 0

# -
# File and instance handling functions 
def create_csv_file(csv_path):
    """Create CSV file with headers if it doesn't exist."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'instance', 'run_id', 'seed',
                'cp_sat_makespan', 'cp_sat_time', 'cp_sat_memory',
                'hcpga_makespan', 'hcpga_time', 'hcpga_memory',
                'ga_makespan', 'ga_time', 'ga_memory'
            ])
        return True
    return False

def process_instance(file_path, args, worker_id):
    """Process a single instance file multiple times with different seeds."""
    instance = load_instance(file_path)
    instance_name = os.path.basename(file_path)
    success_count = 0
    num_runs = args.num_runs
    
    for run_id in range(num_runs):
        # Generate a unique seed for each run based on the base seed
        run_seed = args.seed + run_id
        random.seed(run_seed)
        
        logger.info(f"Worker {worker_id}: Starting run {run_id+1}/{num_runs} for instance {instance_name} with seed {run_seed}")
        success = run_and_log_experiment(instance, args, run_id, run_seed, first_run=(run_id==0 and worker_id==0))
        if success:
            success_count += 1
            
    return success_count, num_runs

# -
def args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Solve and log Job Shop Problem Solver using HCPGA')
    parser.add_argument('--folder', type=str, help='Path to the folder containing the instances', default='instances/ClassicBenchmark')
    parser.add_argument('--csv_file', type=str, default=f'csv/results_{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv', help='CSV file to log the results')
    parser.add_argument('--max_time_budget', type=int, help='Max time budget for each solver in seconds', default=10800)
    parser.add_argument('--seed', type=int, help='Base random seed', default=10)
    parser.add_argument('--num_runs', type=int, help='Number of runs per instance with different seeds', default=1)
    parser.add_argument('--workers', type=int, help='Number of parallel workers (default: 1)', default=1)
    parser.add_argument('--split', type=float, help='Split of time budget between CP-SAT and GA solvers', default=0.3)
    parser.add_argument('--num_copies', type=int, help='Number of copies of the base chromosome to add to the population', default=1)
    parser.add_argument('--num_random', type=int, help='Number of random chromosomes to add to the population', default=30)
    parser.add_argument('--pop_size', type=int, help='Population size for GA solver', default=100)
    parser.add_argument('--max_generations', type=int, help='Max of generations for GA solver', default=1000)
    parser.add_argument('--mutation_rate', type=float, help='Mutation rate for GA solver', default=0.7)
    parser.add_argument('--crossover_rate', type=float, help='Crossover rate for GA solver', default=0.6)
    parser.add_argument('--num_selected', type=int, help='Number of selected chromosomes for GA solver', default=70)
    parser.add_argument('--num_elites', type=int, help='Number of elite chromosomes for GA solver', default=10)
    parser.add_argument('--tournament_size', type=int, help='Tournament size for GA solver', default=5)
    parser.add_argument('--patience', type=int, help='Patience for GA solver', default=3)
    args = parser.parse_args()
    
    return args
    
# - 
def main():
    args = args_parser() # Parse command line arguments

    # Ensure directories exist or create them
    os.makedirs(os.path.dirname(args.csv_file), exist_ok=True)
    os.makedirs('log', exist_ok=True)
    
    # Create CSV file with headers
    first_run = create_csv_file(args.csv_file)
    
    # Set number of workers
    max_workers = args.workers if args.workers else None  # None will use CPU count
    
    # Run for all instances in the folder with parallelization
    if args.folder is not None:
        instance_files = sorted([os.path.join(args.folder, f) 
                              for f in os.listdir(args.folder) 
                              if os.path.isfile(os.path.join(args.folder, f))])
        
        if not instance_files:
            logger.error(f"No instance files found in folder: {args.folder}")
            return
            
        logger.info(f"Starting parallel processing of {len(instance_files)} instances with {max_workers or 'default'} workers")
        
        # Initialize CSV with headers before parallel execution
        create_csv_file(args.csv_file)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all instances to the executor
            future_to_instance = {}
            
            for i, file_path in enumerate(instance_files):
                # Each worker gets a different instance but runs it multiple times
                future = executor.submit(
                    process_instance, 
                    file_path, 
                    args,
                    i
                )
                future_to_instance[future] = file_path
            
            # Process results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_instance):
                instance_path = future_to_instance[future]
                instance_name = os.path.basename(instance_path)
                try:
                    success_count, total_runs = future.result()
                    completed += 1
                    logger.info(f"Progress: {completed}/{len(instance_files)} instances completed. {instance_name}: {success_count}/{total_runs} runs succeeded")
                except Exception as e:
                    logger.error(f"Instance {instance_name} generated an exception: {str(e)}")
            
    logger.info(f"All experiments completed. Results saved to {args.csv_file}")
    
if __name__ == '__main__':
    main()