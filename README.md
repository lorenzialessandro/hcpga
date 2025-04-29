# A Hybrid Constrained Programming with Genetic Algorithm for the Job Shop Scheduling Problem

## Abstract
The Job Shop Scheduling Problem (JSSP) is a widely studied NP-hard optimization problem with significant academic and industrial relevance, particularly in the context of Industry 4.0, where efficient scheduling algorithms are crucial for improving decision-making in increasingly automated production systems. Despite extensive theoretical advancements, a gap remains between academic research and real-world implementation, as most studies either focus on theoretical aspects or emphasize numerical advantages while neglecting practical deployment challenges including those related to computational constraints. 

To fill this gap, we propose a hybrid optimization approach, **`HCPGA`**, which integrates a state-of-the-art Constraint Programming (CP) solver, CP-SAT, with a custom Genetic Algorithm (GA). 

The CP solver generates feasible solutions, which are then used to initialize the GA's population. The GA further optimizes the schedule, minimizing the makespan. 

Our experimental evaluation on 74 JSSP benchmark instances of varying sizes demonstrates that, while standalone CP-SAT and HCPGA achieve comparable makespan results, the latter significantly reduces the time and memory required to find a good solution. This makes our approach highly valuable for industrial applications. To our knowledge, this is the first attempt to combine an evolutionary approach with an exact solver for solving the JSSP, specifically addressing the need for computational efficiency.

## Installation
To install the project follow these steps:

```bash
git clone https://github.com/lorenzialessandro/hcpga
cd hcpga
pip install -r requirements.txt
```


## Usage
The main entry point for running experiments is the `runner.py` script. It provides several command-line options:

```bash
python runner.py [options]
```

### Paper Results
To replicate the results from our paper, you can run the `runner.py` file which automatically executes three different experiments on each instance with the paper parameters:

1. **CP-SAT Solver**: The pure constraint programming approach using Google's OR-Tools CP-SAT solver
2. **HCPGA**: Our hybrid approach combining CP-SAT and Genetic Algorithm
3. **GA Solver**: The pure genetic algorithm approach

The script runs these three solvers sequentially on each instance, collects performance metrics, and outputs the results to a CSV file for analysis.

### Command-line Options

The `runner.py` script accepts the following parameters:

#### Basic Parameters:
- `--folder`: Path to the folder containing the JSSP instances (default: 'instances/ClassicBenchmark')
- `--csv_file`: CSV file to log the results (default: 'csv/results_YYYYMMDD-HHMMSS.csv')
- `--max_time_budget`: Maximum time budget for each solver in seconds (default: 10800, i.e., 3 hours)
- `--seed`: Base random seed (default: 10)
- `--num_runs`: Number of runs per instance with different seeds (default: 1)
- `--workers`: Number of parallel workers (default: 1)

#### HCPGA-specific Parameters:
- `--split`: Split of time budget between CP-SAT and GA solvers (default: 0.3, meaning 30% for CP-SAT and 70% for GA)
- `--num_copies`: Number of copies of the base chromosome to add to the GA population (default: 1)
- `--num_random`: Number of random chromosomes to add to the GA population (default: 30)

#### GA Parameters (used by both GA and HCPGA solvers):
- `--pop_size`: Population size for GA solver (default: 100)
- `--max_generations`: Maximum number of generations for GA solver (default: 1000)
- `--mutation_rate`: Mutation rate for GA solver (default: 0.7)
- `--crossover_rate`: Crossover rate for GA solver (default: 0.6)
- `--num_selected`: Number of selected chromosomes for GA solver (default: 70)
- `--num_elites`: Number of elite chromosomes for GA solver (default: 10)
- `--tournament_size`: Tournament size for GA solver (default: 5)
- `--patience`: Patience for early stopping in GA solver (default: 3)


### Example Usage

Run experiments on all instances in the default folder with 4 parallel workers:

```bash
python runner.py --workers 4 --num_runs 5 --max_time_budget 3600
```

This will:
1. Process each instance 5 times with different random seeds
2. Use a time budget of 1 hour (3600 seconds) per solver
3. Run 4 instances in parallel
4. Save results to a CSV file in the csv/ directory

### Using the HCPGA Class in Your Own Code

You can also use the HCPGA class directly in your own code:

```python
from code.hcpga import HCPGA
from code.utils import load_instance

# Load a JSSP instance
instance = load_instance('path/to/instance/file')

# Create and configure the HCPGA solver
hybrid_solver = HCPGA(
    instance=instance,
    seed=42,
    time_budget=1800,  # 30 minutes
    split=0.3,         # 30% for CP-SAT, 70% for GA
    num_copies=1,
    num_random=30
)

# Solve the instance
schedule, makespan_ga, makespan_cp, total_time, memory_used = hybrid_solver.solve()

# Print results
print(f"HCPGA makespan: {makespan_ga}")
print(f"CP-SAT makespan: {makespan_cp}")
print(f"Total solution time: {total_time:.2f} seconds")
print(f"Memory used: {memory_used / (1024 * 1024):.2f} MB")
```

### Experiments and Measurements

For each instance and solver, the following metrics are measured and recorded:
- **Makespan**: The total completion time of all jobs (the objective to minimize)
- **Solution Time**: The time taken to find the best solution
- **Memory Usage**: The peak memory consumption during solving

The results are stored in a CSV file with the following columns:
- `instance`: Name of the JSSP instance
- `run_id`: Run identifier (for multiple runs with different seeds)
- `seed`: Random seed used
- `cp_sat_makespan`, `cp_sat_time`, `cp_sat_memory`: Results for the CP-SAT solver
- `hcpga_makespan`, `hcpga_time`, `hcpga_memory`: Results for our hybrid approach
- `ga_makespan`, `ga_time`, `ga_memory`: Results for the pure GA solver


## Project Structure

The project is organized as follows:

```
hcpga/
├── code/
│   ├── __init__.py
│   ├── cpsat.py        # CP-SAT solver implementation
│   ├── cpsatc.py       # CP-SAT solver with solution collection
│   ├── ga.py           # Genetic Algorithm implementation
│   ├── hcpga.py        # Hybrid CP-SAT and GA implementation
│   └── utils.py        # Utility functions and classes
├── instances/          # JSSP benchmark instances
│   └── ClassicBenchmark/
├── log/                # Log files directory (created if needed)
├── csv/                # Results CSV files directory (created if needed)
├── runner.py           # Main script to run experiments
├── requirements.txt    
└── README.md           
```

## License

## Contact