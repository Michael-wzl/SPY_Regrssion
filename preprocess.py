from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def to_logret(prices: pd.DataFrame) -> pd.DataFrame:
    float_prices = prices.astype(float)
    logret = np.log(float_prices).diff()
    logret.iloc[0] = 0.0
    return logret

def from_logret(logrets, init_prices, cumulative: bool = True):
    """
    Convert log returns back to prices.
    
    Args:
        logrets: Log returns (pd.DataFrame, pd.Series, or np.ndarray)
        init_prices: Initial prices for conversion (same type as logrets)
        cumulative: If True, treat logrets as sequential returns and compute cumulative prices.
                   If False, treat each logret independently: P_t = P_{t-1} * exp(logret_t)
    
    Returns:
        Prices in the same type as input logrets
    """
    # Handle numpy arrays
    if isinstance(logrets, np.ndarray):
        float_logrets = logrets.astype(float)
        float_init = np.asarray(init_prices).astype(float)
        if cumulative:
            return float_init * np.exp(np.cumsum(float_logrets))
        else:
            return float_init * np.exp(float_logrets)
    
    # Handle pandas DataFrame/Series
    float_logrets = logrets.astype(float)
    float_init = init_prices.astype(float) if hasattr(init_prices, 'astype') else float(init_prices)
    
    if cumulative:
        prices = float_init * np.exp(float_logrets.cumsum())
    else:
        prices = float_init * np.exp(float_logrets)
    return prices

class CovSelector:
    def __init__(self, topk: int = 150, method: str = 'spearman'):
        self.topk = topk
        self.method = method
        self.selected_features_: Optional[List[str]] = None
    
    def fit(self, dfs: pd.DataFrame, target: pd.Series) -> None:
        if self.method == 'spearman':
            corr_matrix = dfs.corrwith(target, method='spearman').abs()
        elif self.method == 'pearson':
            corr_matrix = dfs.corrwith(target, method='pearson').abs()
        else:
            raise ValueError("Unsupported correlation method. Use 'spearman' or 'pearson'.")
        self.selected_features_ = corr_matrix.nlargest(self.topk).index.tolist()
    
    def transform(self, dfs: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise ValueError("CovSelector must be fitted before calling transform.")
        return dfs[self.selected_features_]
    

class ZScoreScaler:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.means_: Optional[pd.Series] = None
        self.stds_: Optional[pd.Series] = None

    def fit(self, dfs: pd.DataFrame) -> None:
        self.means_ = dfs.mean()
        self.stds_ = dfs.std(ddof=0).replace(0, self.eps)

    def transform(self, dfs: pd.DataFrame) -> pd.DataFrame:
        if self.means_ is None or self.stds_ is None:
            raise ValueError("ZScoreScaler must be fitted before calling transform.")
        scaled = (dfs - self.means_) / (self.stds_ + self.eps)
        return scaled.add_suffix(f"_zscorescaled")
    
class Winsorizer:
    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.q_low_: Optional[pd.Series] = None
        self.q_high_: Optional[pd.Series] = None

    def fit(self, dfs: pd.DataFrame) -> None:
        self.q_low_ = dfs.quantile(self.lower_q)
        self.q_high_ = dfs.quantile(self.upper_q)

    def transform(self, dfs: pd.DataFrame) -> pd.DataFrame:
        if self.q_low_ is None or self.q_high_ is None:
            raise ValueError("Winsorizer must be fitted before calling transform.")
        winsorized = dfs.clip(lower=self.q_low_, upper=self.q_high_, axis=1).add_suffix(f"_winsorized")
        return winsorized
    
class PCACompressor:
    def __init__(self, n_components: int,rand_state: int = 42):
        self.n_components = n_components
        self.rand_state = rand_state
        self.pca_model: Optional[PCA] = None
    
    def fit(self, dfs: pd.DataFrame) -> None:
        self.pca_model = PCA(n_components=self.n_components, random_state=self.rand_state)
        self.pca_model.fit(dfs)

    def transform(self, dfs: pd.DataFrame) -> pd.DataFrame:
        if self.pca_model is None:
            raise ValueError("PCACompressor must be fitted before calling transform.")
        transformed_array = self.pca_model.transform(dfs)
        pca_columns = [f"pca_{i+1}" for i in range(self.n_components)]
        transformed_df = pd.DataFrame(transformed_array, index=dfs.index, columns=pca_columns)
        return transformed_df
    
class Preprocessor:
    def __init__(self, cfgs: Dict[str, Dict[str, Any]], steps: List[Tuple[str, str]], fit_dfs: pd.DataFrame, target: Optional[pd.Series] = None):
        """
        Initialize the preprocessor with configuration and steps.
        
        Args:
            cfgs: Configuration dictionary for each preprocessing method.
            steps: List of (method, operation) tuples. 'a' = append, 'o' = overwrite.
            fit_dfs: DataFrame to fit the preprocessor on (training features only).
            target: Target series, required if 'cov_select' is in steps.
        
        Note:
            Each preprocessing step fits on the data BEFORE any transformation from the current step.
            After fitting, the data is transformed for use by subsequent steps.
            This ensures:
            - cov_select: fits on original features
            - zscore: fits on selected features (after cov_select transform)
            - winsor: fits on selected features (same as zscore input, NOT on zscore-transformed data)
            - pca: fits on the output of prior transforms
        """
        self.cfgs = cfgs
        self.process_units: List[Tuple[str, str, Any]] = [] # (method, type, instance)
        
        # Track data at each stage for proper fitting
        current_data = fit_dfs.copy()
        
        for method, operation in steps:
            if method == 'cov_select':
                if target is None:
                    raise ValueError("CovSelector requires 'target' parameter in Preprocessor.__init__")
                unit = CovSelector(**cfgs[method])
                unit.fit(current_data, target)
            elif method == 'zscore':
                unit = ZScoreScaler(**cfgs[method])
                unit.fit(current_data)  # fit on current_data (before zscore transform)
            elif method == 'winsor':
                unit = Winsorizer(**cfgs[method])
                unit.fit(current_data)  # fit on current_data (before winsor transform)
            elif method == 'pca':
                unit = PCACompressor(**cfgs[method])
                unit.fit(current_data)
            else:
                raise ValueError(f"Unsupported preprocessing method: {method}")
            
            self.process_units.append((method, operation, unit))
            
            # Update current_data for next step's fitting
            if operation == 'a': # Append the new features to the original
                if method in ['cov_select']:
                    raise ValueError(f"Append operation not supported for method: {method}")
                current_data = pd.concat([current_data, unit.transform(current_data)], axis=1)
            elif operation == 'o': # Overwrite the original features
                current_data = unit.transform(current_data)
            else:
                raise ValueError(f"Unsupported operation type: {operation}")
    
    def transform(self, dfs: pd.DataFrame) -> pd.DataFrame:
        for method, operation, unit in self.process_units:
            if operation == 'a':
                dfs = pd.concat([dfs, unit.transform(dfs)], axis=1)
            elif operation == 'o':
                dfs = unit.transform(dfs)
            else:
                raise ValueError(f"Unsupported operation type: {operation}")
        return dfs
