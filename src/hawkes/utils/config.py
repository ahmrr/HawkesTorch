from dataclasses import dataclass


@dataclass
class HawkesFitConfig:
    num_steps: int = 1000  # Number of optimization steps/epochs
    batch_size: int = None  # Size of each batch during fitting (None for no batching)
    monitor_interval: int = 100  # Print fitting progress every monitor_interval steps
    learning_rate: float = 0.01  # Learning rate for the Adam optimizer
    l1_penalty: float = 0.01  # L1 regularization strength for sparsity
    l1_hinge: float = 1  # Rate values below this are regularized using L1 penalty
    nuc_penalty: float = 0.01  # Nuclear norm regularization strength for low-rank


@dataclass
class HawkesDebugConfig:
    deterministic_sim: bool = True  # Set to make simulated data deterministic (each run with the same params will simulate the exact same data)
    profile_mem_iters: int = (
        0  # Set to profile the memory usage of some amount of training iterations
    )
    profile_mem_entries: int = 100000  # Maximum number of recorded memory entries, if memory profiling is enabled
