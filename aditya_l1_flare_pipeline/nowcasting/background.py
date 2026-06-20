import pandas as pd
import numpy as np

def subtract_background(flux_series: pd.Series) -> pd.Series:
    """
    Quiet-Sun background subtraction on a flux series.
    Estimates quiet-Sun background baseline robustly using a rolling minimum of a rolling median,
    and handles zero, negative, and NaN values.
    """
    if flux_series.empty:
        return flux_series.copy()
        
    if flux_series.isna().all():
        return flux_series.copy()
        
    original_index = flux_series.index
    
    # 1. Clean the series for baseline calculation:
    filled = flux_series.interpolate(method='linear', limit_direction='both')
    filled = filled.ffill().bfill().fillna(0.0)
    
    # Clip negative values to 0.0 (physical flux cannot be negative)
    filled = filled.clip(lower=0.0)
    
    # If the series is constantly zero (or all non-NaN values <= 0), return zeros (preserving NaNs)
    if (filled == 0.0).all():
        return flux_series.where(flux_series.isna(), 0.0)
        
    n = len(flux_series)
    
    # 2. Robust Baseline Estimation:
    min_window = max(3, min(n, 360))
    median_window = max(3, min(n // 20, 15))
    if median_window % 2 == 0:
        median_window += 1  # Keep window size odd
        
    smoothed = filled.rolling(window=median_window, center=True, min_periods=1).median()
    baseline = smoothed.rolling(window=min_window, center=True, min_periods=1).min()
    
    subtracted = flux_series - baseline
    return subtracted
