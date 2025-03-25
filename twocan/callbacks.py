from typing import Dict, Any, Optional
import optuna
import numpy as np
import pandas as pd


class SaveTrialsDFCallback:
    """Callback to save Optuna trials to a CSV file with additional annotations.
    
    Parameters
    ----------
    df_path : str
        Path where the trials DataFrame will be saved.
    anno_dict : Dict[str, Any], default={}
        Dictionary of additional annotations to add to each trial.
    """
    def __init__(self, df_path: str, anno_dict: Dict[str, Any] = {}):
        self.df_path = df_path
        self.anno_dict = anno_dict
    
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Save the current state of all trials to a CSV file.
        
        Parameters
        ----------
        study : optuna.study.Study
            The Optuna study object.
        trial : optuna.trial.FrozenTrial
            The current trial object.
        """
        study_df = study.trials_dataframe()
        #study_df['best_trial'] = study_df.number.isin([t.number for t in study.best_trials])
        study_df = study_df.assign(**self.anno_dict)
        study_df.to_csv(self.df_path, index=False)

class ThresholdReachedCallback:
    """Callback to stop optimization when quality thresholds are reached.
    
    Parameters
    ----------
    df_path : str
        Path to the trials DataFrame.
    and_threshold : int
        Minimum value for logical AND metric.
    corr_threshold : float
        Minimum value for correlation metric.
    iou_threshold : float
        Minimum value for IoU metric.
    """
    # stop early if a good registration is found
    def __init__(self, df_path: str, and_threshold: int, corr_threshold: float, iou_threshold: float):
        self.df_path = df_path
        self.and_threshold = and_threshold
        self.corr_threshold = corr_threshold
        self.iou_threshold = iou_threshold
    
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Check if quality thresholds are met and stop the study if they are.
        
        Parameters
        ----------
        study : optuna.study.Study
            The Optuna study object.
        trial : optuna.trial.FrozenTrial
            The current trial object.
        """
        study_df = study.trials_dataframe()
        corr_crit = study_df['user_attrs_image_max_corr'] >= self.corr_threshold  
        and_crit = study_df['user_attrs_logical_and'] >= self.and_threshold
        iou_crit = study_df['user_attrs_logical_iou'] >= self.iou_threshold
        if (corr_crit and and_crit and iou_crit): 
            study.stop()
        return 

class MatrixConvergenceCallback:
    """Callback to stop optimization when registration matrices converge.
    
    Stops the study when the best n trials (by correlation or IoU) 
    have similar transformation matrices.
    
    Parameters
    ----------
    df_path : str
        Path to the trials DataFrame.
    n_same : int, default=3
        Number of consecutive similar matrices required to stop.
    """
    # stop early if the best n registrations all agree on the same matrix
    def __init__(self, df_path: str, n_same: int = 3):
        self.df_path = df_path
        self.n_same = n_same
    
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Check if the best registration matrices have converged.
        
        Parameters
        ----------
        study : optuna.study.Study
            The Optuna study object.
        trial : optuna.trial.FrozenTrial
            The current trial object.
        """
        study_df = study.trials_dataframe()
        # Drop rows with NaN values in key columns
        study_df = study_df.dropna(subset=['user_attrs_image_max_corr', 'user_attrs_logical_iou', 'user_attrs_registration_matrix'])
        # arrange by max corr, take top n
        best_corr_df = study_df.sort_values(by='user_attrs_image_max_corr', ascending=False).head(self.n_same)
        # arrange by iou, take top n
        best_iou_df = study_df.sort_values(by='user_attrs_logical_iou', ascending=False).head(self.n_same)
        # concat the two and deduplicate
        best_df = pd.concat([best_corr_df, best_iou_df])
        print(best_df)
        # Get matrices from last 3 best trials
        matrices = [m for m in best_df['user_attrs_registration_matrix']]
        print(matrices)
        all_close = [np.allclose(matrices[i], matrices[i+1]) for i in range(len(matrices)-1)]
        print(all_close)
        if sum(all_close)>=self.n_same:
            study.stop()
        return
        