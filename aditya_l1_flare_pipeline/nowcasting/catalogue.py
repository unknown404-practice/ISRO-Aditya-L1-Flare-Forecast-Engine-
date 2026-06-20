import pandas as pd
import numpy as np

def get_goes_class(peak_flux: float) -> str:
    if peak_flux <= 0:
        return "A0.0"
        
    if peak_flux < 1e-7:
        prefix = "A"
        val = peak_flux / 1e-8
    elif peak_flux < 1e-6:
        prefix = "B"
        val = peak_flux / 1e-7
    elif peak_flux < 1e-5:
        prefix = "C"
        val = peak_flux / 1e-6
    elif peak_flux < 1e-4:
        prefix = "M"
        val = peak_flux / 1e-5
    else:
        prefix = "X"
        val = peak_flux / 1e-4
        
    val_str = f"{val:.1f}"
    if val_str == "10.0" and prefix != "X":
        # Promote to the next class due to float rounding boundaries (e.g. 9.96e-8 -> B1.0)
        promotions = {"A": "B", "B": "C", "C": "M", "M": "X"}
        prefix = promotions[prefix]
        val_str = "1.0"
        
    return f"{prefix}{val_str}"

def generate_catalogue(solexs_df: pd.DataFrame, events: list) -> pd.DataFrame:
    columns = ['start_time', 'peak_time', 'end_time', 'peak_flux', 'goes_class', 'active_region']
    
    if not events or solexs_df is None or solexs_df.empty:
        return pd.DataFrame(columns=columns)
        
    # Ensure index is sorted for range slicing and matching
    if not solexs_df.index.is_monotonic_increasing:
        solexs_df = solexs_df.sort_index()
        
    rows = []
    for event in events:
        if isinstance(event, dict):
            start_t = event.get('start')
            end_t = event.get('end')
        elif isinstance(event, (tuple, list)):
            if len(event) >= 3:
                start_t = event[0]
                end_t = event[2]
            elif len(event) == 2:
                start_t = event[0]
                end_t = event[1]
            else:
                continue
        else:
            continue
            
        # Protect against empty triggers/missing times
        if start_t is None or end_t is None:
            continue
            
        is_integer_slicing = False
        if isinstance(start_t, (int, np.integer)) and not isinstance(solexs_df.index, pd.RangeIndex) and not np.issubdtype(solexs_df.index.dtype, np.integer):
            is_integer_slicing = True
            
        try:
            if is_integer_slicing:
                sub_df = solexs_df.iloc[int(start_t):int(end_t)+1]
            elif isinstance(solexs_df.index, pd.MultiIndex):
                # Handle MultiIndex: Find the datetime-like level
                time_level = 0
                for idx, name in enumerate(solexs_df.index.names):
                    if name and ('time' in str(name).lower() or 'date' in str(name).lower()):
                        time_level = idx
                        break
                # Filter rows where time level is within range
                time_vals = solexs_df.index.get_level_values(time_level)
                sub_df = solexs_df[(time_vals >= start_t) & (time_vals <= end_t)]
            else:
                sub_df = solexs_df.loc[start_t:end_t]
        except Exception:
            continue
            
        # Robustly determine soft_xray column name
        flux_col = None
        for col in ['soft_xray', 'flux', 'rate', 'counts']:
            if col in sub_df.columns:
                flux_col = col
                break
        if flux_col is None:
            num_cols = sub_df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                flux_col = num_cols[0]
            else:
                flux_col = 'soft_xray'  # Fallback
                
        if flux_col not in sub_df.columns:
            continue
            
        sub_df_clean = sub_df.dropna(subset=[flux_col])
        if sub_df_clean.empty:
            continue
            
        peak_flux = sub_df_clean[flux_col].max()
        peak_time_idx = sub_df_clean[flux_col].idxmax()
        
        # In case of MultiIndex, idxmax returns a tuple (timestamp, active_region, etc.)
        # We need to extract the raw datetime/timestamp for the catalogue column
        if isinstance(peak_time_idx, tuple) and isinstance(solexs_df.index, pd.MultiIndex):
            # Extract datetime from tuple
            time_idx = 0
            for idx, name in enumerate(solexs_df.index.names):
                if name and ('time' in str(name).lower() or 'date' in str(name).lower()):
                    time_idx = idx
                    break
            for i, val in enumerate(peak_time_idx):
                if isinstance(val, (pd.Timestamp, np.datetime64)):
                    time_idx = i
                    break
            peak_time = peak_time_idx[time_idx]
        else:
            peak_time = peak_time_idx
            
        # Match active region using flexible candidate name list and levels
        active_region_names = ['active_region', 'ar', 'region', 'active_region_id']
        active_region = "Unknown"
        
        # Check columns first
        found_ar_col = None
        for col_name in active_region_names:
            if col_name in sub_df_clean.columns:
                found_ar_col = col_name
                break
                
        if found_ar_col is not None:
            active_region = sub_df_clean.loc[peak_time_idx, found_ar_col]
            if isinstance(active_region, pd.Series):
                active_region = active_region.iloc[0]
        elif isinstance(solexs_df.index, pd.MultiIndex):
            # If not in columns, check index level names
            found_ar_level = None
            for lvl_name in active_region_names:
                if lvl_name in solexs_df.index.names:
                    found_ar_level = lvl_name
                    break
            if found_ar_level is not None:
                ar_lvl_idx = solexs_df.index.names.index(found_ar_level)
                active_region = peak_time_idx[ar_lvl_idx]
                
        # Clean active region string
        if pd.isna(active_region) or str(active_region).strip().lower() in ['nan', 'none', '', 'nat']:
            active_region = "Unknown"
            
        goes_class = get_goes_class(peak_flux)
        
        rows.append({
            'start_time': start_t,
            'peak_time': peak_time,
            'end_time': end_t,
            'peak_flux': peak_flux,
            'goes_class': goes_class,
            'active_region': str(active_region)
        })
        
    return pd.DataFrame(rows, columns=columns)
