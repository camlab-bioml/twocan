from typing import Dict, Any, Optional
import optuna
import numpy as np
import pandas as pd


class SaveTrialsDFCallback:
    """Callback to save optimization trials to a CSV file during study execution.
    
    This callback writes trial results to a CSV file after each trial completion,
    allowing for real-time monitoring and persistent storage of optimization
    progress. Useful for long-running optimizations or when resuming studies.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file where trial data will be saved.
        File will be created if it doesn't exist.
    save_frequency : int, default=1
        Number of trials between saves. Set to 1 to save after every trial,
        higher values for less frequent saves to reduce I/O overhead.
        
    Attributes
    ----------
    filepath : str
        Current filepath for saving trial data.
    save_frequency : int
        Current save frequency setting.
    trial_count : int
        Internal counter tracking number of completed trials.
    """
    
    def __init__(self, filepath: str, save_frequency: int = 1):
        self.filepath = filepath
        self.save_frequency = save_frequency
        self.trial_count = 0
    
    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Save study trials to CSV file.
        
        This method is called by Optuna after each trial completion.
        It converts the study's trial data to a DataFrame and saves it
        to the specified CSV file.
        
        Parameters
        ----------
        study : optuna.Study
            The Optuna study object containing all trials.
        trial : optuna.Trial
            The just-completed trial (not directly used but required
            by Optuna's callback interface).
        """
        self.trial_count += 1
        if self.trial_count % self.save_frequency == 0:
            df = study.trials_dataframe()
            df.to_csv(self.filepath, index=False)


class ThresholdReachedCallback:
    """Callback to stop optimization when a target metric threshold is reached.
    
    This callback monitors a specified metric during optimization and raises
    optuna.TrialPruned to stop the study when the threshold is reached. Useful
    for stopping optimization early when satisfactory results are achieved.
    
    Parameters
    ----------
    threshold : float
        Target threshold value for the monitored metric.
    metric_name : str, default='iou'
        Name of the metric to monitor in trial.user_attrs.
        Must be a key present in the user attributes of trials.
    direction : str, default='maximize'
        Whether to stop when metric goes 'above' ('maximize') or 'below' 
        ('minimize') the threshold.
        
    Attributes
    ----------
    threshold : float
        Current threshold value.
    metric_name : str
        Current metric being monitored.
    direction : str
        Current direction ('maximize' or 'minimize').
    """
    
    def __init__(self, threshold: float, metric_name: str = 'iou', direction: str = 'maximize'):
        self.threshold = threshold
        self.metric_name = metric_name
        self.direction = direction
    
    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Check if threshold has been reached and stop study if so.
        
        This method is called after each trial. It checks if the specified
        metric has reached the threshold and raises TrialPruned to stop
        the optimization if the condition is met.
        
        Parameters
        ----------
        study : optuna.Study
            The Optuna study object.
        trial : optuna.Trial
            The just-completed trial containing the metric value.
            
        Raises
        ------
        optuna.TrialPruned
            When the threshold condition is met, stopping the optimization.
        """
        if self.metric_name in trial.user_attrs:
            metric_value = trial.user_attrs[self.metric_name]
            
            if self.direction == 'maximize' and metric_value >= self.threshold:
                print(f"Threshold reached: {self.metric_name} = {metric_value:.4f} >= {self.threshold}")
                raise optuna.TrialPruned()
            elif self.direction == 'minimize' and metric_value <= self.threshold:
                print(f"Threshold reached: {self.metric_name} = {metric_value:.4f} <= {self.threshold}")
                raise optuna.TrialPruned()


class MatrixConvergenceCallback:
    """Callback to monitor transformation matrix convergence during optimization.
    
    This callback tracks the transformation matrices from recent trials and
    stops optimization when they converge (have low variance), indicating
    that the registration has stabilized. Useful for detecting when further
    optimization is unlikely to improve results.
    
    Parameters
    ----------
    window_size : int, default=10
        Number of recent trials to consider for convergence assessment.
    tolerance : float, default=0.01
        Maximum allowed variance in matrix elements for convergence.
        Lower values require tighter convergence.
    min_trials : int, default=20
        Minimum number of trials before convergence checking begins.
        
    Attributes
    ----------
    window_size : int
        Current window size for convergence assessment.
    tolerance : float
        Current tolerance for matrix element variance.
    min_trials : int
        Minimum trials before convergence checking.
    matrices : List[np.ndarray]
        List storing recent transformation matrices.
    """
    
    def __init__(self, window_size: int = 10, tolerance: float = 0.01, min_trials: int = 20):
        self.window_size = window_size
        self.tolerance = tolerance
        self.min_trials = min_trials
        self.matrices = []
    
    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Check transformation matrix convergence and stop if converged.
        
        This method monitors the transformation matrices from recent trials
        and stops the study when they show low variance, indicating convergence.
        
        Parameters
        ----------
        study : optuna.Study
            The Optuna study object.
        trial : optuna.Trial
            The just-completed trial containing the transformation matrix.
            
        Raises
        ------
        optuna.TrialPruned
            When matrix convergence is detected.
        """
        # Extract transformation matrix if available
        if 'M' in trial.user_attrs:
            matrix = trial.user_attrs['M']
            self.matrices.append(matrix)
            
            # Keep only recent matrices
            if len(self.matrices) > self.window_size:
                self.matrices = self.matrices[-self.window_size:]
            
            # Check convergence if we have enough trials
            if len(self.matrices) >= self.window_size and trial.number >= self.min_trials:
                # Calculate variance across recent matrices
                matrix_stack = np.stack(self.matrices)
                variance = np.var(matrix_stack, axis=0)
                max_variance = np.max(variance)
                
                if max_variance < self.tolerance:
                    print(f"Matrix convergence detected: max variance = {max_variance:.6f} < {self.tolerance}")
                    raise optuna.TrialPruned()
        