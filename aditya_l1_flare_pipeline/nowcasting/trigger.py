import pandas as pd
import numpy as np

def detect_events(hel1os_deriv_series: pd.Series, threshold: float, window: int = 3, min_duration: float = 0.0) -> list:
    """
    Detect impulsive trigger events in the HEL1OS derivative series.
    Filters out high-frequency noise and merges triggers that are close in time.

    Parameters:
    -----------
    hel1os_deriv_series : pd.Series
        Series containing hard X-ray flux derivative.
    threshold : float
        Trigger threshold.
    window : int, default 3
        Rolling window size for noise smoothing.
    min_duration : float, default 0.0
        Minimum duration of a trigger event (in seconds if DatetimeIndex, else in index units)
        to filter out transient noise spikes.
    """
    # 1. Input Validation
    if not isinstance(hel1os_deriv_series, pd.Series):
        raise TypeError("hel1os_deriv_series must be a pandas Series")
        
    if threshold is None or np.isnan(threshold):
        raise ValueError("threshold must be a valid float")

    if hel1os_deriv_series.empty:
        return []
        
    # 2. Index Sorting and Cleaning
    # Clean index from NaNs/NaTs and ensure it is monotonic increasing
    series_cleaned = hel1os_deriv_series.copy()
    series_cleaned = series_cleaned[series_cleaned.index.notna()]
    if series_cleaned.empty:
        return []
        
    if not series_cleaned.index.is_monotonic_increasing:
        series_cleaned = series_cleaned.sort_index()
        
    deriv = series_cleaned.fillna(0.0)
    
    # 3. High-frequency noise filtering:
    if window < 1:
        raise ValueError("window must be at least 1")
    smoothed = deriv.rolling(window=window, center=True, min_periods=1).mean()
    
    # 4. Trigger thresholding:
    active_mask = smoothed > threshold
    if not active_mask.any():
        return []
        
    indices = series_cleaned.index
    is_datetime = isinstance(indices, pd.DatetimeIndex)
    
    # Define merge distance limit dynamically:
    if is_datetime:
        merge_limit = 120.0
    else:
        # Check median step size of the numeric index
        if len(indices) > 1:
            step_size = float(np.median(np.diff(indices)))
        else:
            step_size = 1.0
            
        # If step_size is large, the index represents actual seconds (e.g. 10.0)
        # If step_size is small, it represents step indices (e.g. 1.0)
        if step_size >= 5.0:
            merge_limit = 120.0
        else:
            merge_limit = 12.0
            
    def distance(t1, t2):
        if is_datetime:
            return (t2 - t1).total_seconds()
        else:
            return float(t2 - t1)
            
    # 5. Scan and extract contiguous trigger blocks
    raw_blocks = []
    in_trigger = False
    block_start = None
    
    for i, val in enumerate(active_mask):
        if val:
            if not in_trigger:
                block_start = indices[i]
                in_trigger = True
        else:
            if in_trigger:
                block_end = indices[i-1]
                raw_blocks.append((block_start, block_end))
                in_trigger = False
                
    if in_trigger:
        raw_blocks.append((block_start, indices[-1]))
        
    if not raw_blocks:
        return []
        
    # 6. Merge close triggers
    merged_blocks = []
    curr_start, curr_end = raw_blocks[0]
    for next_start, next_end in raw_blocks[1:]:
        if distance(curr_end, next_start) <= merge_limit:
            curr_end = next_end
        else:
            merged_blocks.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    merged_blocks.append((curr_start, curr_end))
    
    # 7. Build events list with peak determination and duration filtering
    events = []
    for start_t, end_t in merged_blocks:
        # Filter by minimum duration to reject short noise transients
        dur = distance(start_t, end_t)
        if dur < min_duration:
            continue
            
        sub_series = deriv.loc[start_t:end_t]
        if sub_series.empty:
            peak_t = start_t
        else:
            peak_t = sub_series.idxmax()
            
        events.append({
            'start': start_t,
            'peak': peak_t,
            'end': end_t
        })
        
    return events
