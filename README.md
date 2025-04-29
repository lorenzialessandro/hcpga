# A Hybrid Constrained Programming with Genetic Algorithm for the Job Shop Scheduling Problem

## Abstract
The Job Shop Scheduling Problem (JSSP) is a widely studied NP-hard optimization problem with significant academic and industrial relevance, particularly in the context of Industry 4.0, where efficient scheduling algorithms are crucial for improving decision-making in increasingly automated production systems. Despite extensive theoretical advancements, a gap remains between academic research and real-world implementation, as most studies either focus on theoretical aspects or emphasize numerical advantages while neglecting practical deployment challenges including those related to computational constraints. 

To fill this gap, we propose a hybrid optimization approach, **`HCPGA`**, which integrates a state-of-the-art Constraint Programming (CP) solver, CP-SAT, with a custom Genetic Algorithm (GA). 

The CP solver generates feasible solutions, which are then used to initialize the GA's population. The GA further optimizes the schedule, minimizing the makespan. 

Our experimental evaluation on 74 JSSP benchmark instances of varying sizes demonstrates that, while standalone CP-SAT and HCPGA achieve comparable makespan results, the latter significantly reduces the time and memory required to find a good solution. This makes our approach highly valuable for industrial applications. To our knowledge, this is the first attempt to combine an evolutionary approach with an exact solver for solving the JSSP, specifically addressing the need for computational efficiency.

## Installation

## Paper Results and HCPGA class usage

## License

## Contact