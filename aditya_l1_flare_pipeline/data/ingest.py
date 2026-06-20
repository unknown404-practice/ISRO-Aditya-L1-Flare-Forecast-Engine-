import numpy as np
import pandas as pd

def clean_zero_dropouts(series: pd.Series, epsilon_zero: float, k: int, theta_active: float) -> pd.Series:
    """
    Identifies zero dropouts and replaces them with NaN.
    """
    rolling_median = series.rolling(window=2*k+1, center=True, min_periods=1).median()
    is_dropout = (series <= epsilon_zero) & (rolling_median > theta_active)
    cleaned = series.copy()
    cleaned[is_dropout] = np.nan
    return cleaned

def clean_transient_spikes(series: pd.Series, z_threshold: float, k: int, epsilon: float = 1e-9) -> pd.Series:
    """
    Identifies transient spikes using MAD Z-score and replaces them with NaN.
    """
    rolling_median = series.rolling(window=2*k+1, center=True, min_periods=1).median()
    rolling_mad = series.rolling(window=2*k+1, center=True, min_periods=1).apply(
        lambda w: np.nanmedian(np.abs(w - np.nanmedian(w))),
        raw=True
    )
    z_mad = (series - rolling_median).abs() / (rolling_mad + epsilon)
    
    is_outlier = (z_mad > z_threshold).fillna(False)
    left_ok = ~is_outlier.shift(1, fill_value=False)
    right_ok = ~is_outlier.shift(-1, fill_value=False)
    
    is_spike = is_outlier & left_ok & right_ok
    cleaned = series.copy()
    cleaned[is_spike] = np.nan
    return cleaned

def correct_baseline_resets(series: pd.Series, z_step_threshold: float, theta_min_step: float, k: int, epsilon: float = 1e-9) -> pd.Series:
    """
    Corrects baseline resets by shifting subsequent values.
    """
    # 1. Temporarily interpolate the series containing NaNs to create a continuous signal y
    y = series.interpolate(method='linear', limit_direction='both')
    if y.isna().all():
        return series.copy()
        
    # 2. Compute first difference
    d = y.diff().fillna(0.0)
    
    # 3. Compute rolling difference Z-score
    rolling_median_d = d.rolling(window=2*k+1, center=True, min_periods=1).median()
    rolling_mad_d = d.rolling(window=2*k+1, center=True, min_periods=1).apply(
        lambda w: np.nanmedian(np.abs(w - np.nanmedian(w))),
        raw=True
    )
    z_diff = (d - rolling_median_d).abs() / (rolling_mad_d + epsilon)
    
    # 4. Identify step indices where Z_diff > threshold and |d_t| > theta_min_step
    is_step_candidate = (z_diff > z_step_threshold) & (d.abs() > theta_min_step)
    is_step_candidate = is_step_candidate.fillna(False)
    
    # 5. Differentiate Step from Spike
    next_d = d.shift(-1).fillna(0.0)
    is_not_spike = (d + next_d).abs() >= 0.3 * d.abs()
    is_step = is_step_candidate & is_not_spike
    
    # 6. Cumulative Correction
    delta = np.where(is_step, d, 0.0)
    C = np.cumsum(delta)
    
    # 7. Apply Correction
    return series - C

def interpolate_small_gaps(series: pd.Series, limit: int = 3) -> pd.Series:
    """
    Interpolates gaps of size <= limit and leaves larger gaps as NaN.
    """
    nan_mask = series.isna()
    if not nan_mask.any():
        return series
        
    # Group contiguous blocks of NaNs
    group = (nan_mask != nan_mask.shift()).cumsum()
    group_sizes = series.groupby(group).transform('size')
    
    # Interpolate using linear method
    interpolated = series.interpolate(method='linear', limit_direction='both')
    
    # Restore original NaNs for gaps larger than limit
    restore_mask = nan_mask & (group_sizes > limit)
    result = interpolated.where(~restore_mask, np.nan)
    return result

def _detect_file_type(path: str) -> str:
    import os
    if not os.path.exists(path):
        raise ValueError(f"File {path} does not exist")
    if os.path.getsize(path) == 0:
        raise ValueError(f"File {path} is empty")
        
    try:
        with open(path, 'rb') as f:
            first_8 = f.read(8)
    except Exception as e:
        raise ValueError(f"Failed to read file {path}: {str(e)}")
        
    if first_8 == b'SIMPLE  ':
        return 'fits'
        
    _, ext = os.path.splitext(path.lower())
    if ext in ['.fits', '.fit', '.fts']:
        return 'fits'
        
    return 'csv'


def _find_column(columns, candidates):
    col_map = {col.lower(): col for col in columns}
    for cand in candidates:
        if cand.lower() in col_map:
            return col_map[cand.lower()]
    return None


def get_header_value(hdul, table_hdu, key):
    val = table_hdu.header.get(key)
    if val is not None:
        return val
    if len(hdul) > 0:
        val = hdul[0].header.get(key)
        if val is not None:
            return val
    return None


def _parse_time_column(series: pd.Series, hdul=None, table_hdu=None) -> pd.Series:
    """
    Parses a time series (numeric or string) into pandas datetime Series.
    """
    # Decode byte strings if any
    if series.dtype == object or series.dtype.kind in ('S', 'V'):
        series = series.apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x))
        
    col_name = str(series.name).lower() if series.name is not None else ""
        
    if pd.api.types.is_numeric_dtype(series):
        valid_vals = series.dropna()
        if len(valid_vals) == 0:
            res = pd.to_datetime(series)
            if res.dt.tz is not None:
                res = res.dt.tz_convert('UTC').dt.tz_localize(None)
            return res
            
        # 1. Julian Date (JD) check
        is_jd = 'jd' in col_name and 'mjd' not in col_name
        if not is_jd and valid_vals.min() > 2400000:
            is_jd = True
            
        if is_jd:
            res = pd.to_datetime(series - 2400000.5, unit='D', origin='1858-11-17')
            if res.dt.tz is not None:
                res = res.dt.tz_convert('UTC').dt.tz_localize(None)
            return res
            
        # 2. Modified Julian Date (MJD) check
        mjdref_exists = False
        if hdul is not None and table_hdu is not None:
            if (get_header_value(hdul, table_hdu, 'MJDREF') is not None or
                get_header_value(hdul, table_hdu, 'MJDREFI') is not None or
                get_header_value(hdul, table_hdu, 'MJDREFF') is not None):
                mjdref_exists = True
                
        is_mjd = 'mjd' in col_name
        if not is_mjd and valid_vals.between(30000, 100000).all():
            if not mjdref_exists and 'sec' not in col_name and 'second' not in col_name:
                is_mjd = True
                
        if is_mjd:
            res = pd.to_datetime(series, unit='D', origin='1858-11-17')
            if res.dt.tz is not None:
                res = res.dt.tz_convert('UTC').dt.tz_localize(None)
            return res
        else:
            # MET seconds
            mjdref = None
            if hdul is not None and table_hdu is not None:
                mjdref_val = get_header_value(hdul, table_hdu, 'MJDREF')
                if mjdref_val is not None:
                    try:
                        mjdref = float(mjdref_val)
                    except ValueError:
                        pass
                else:
                    mjdrefi = get_header_value(hdul, table_hdu, 'MJDREFI')
                    mjdreff = get_header_value(hdul, table_hdu, 'MJDREFF')
                    if mjdrefi is not None:
                        try:
                            mjdref = float(mjdrefi) + float(mjdreff or 0.0)
                        except ValueError:
                            pass
            
            if mjdref is not None:
                epoch = pd.to_datetime(mjdref, unit='D', origin='1858-11-17')
            else:
                epoch = pd.to_datetime('2020-01-01T00:00:00')
                
            res = epoch + pd.to_timedelta(series, unit='s')
            if res.dt.tz is not None:
                res = res.dt.tz_convert('UTC').dt.tz_localize(None)
            return res
    else:
        res = pd.to_datetime(series)
        if res.dt.tz is not None:
            res = res.dt.tz_convert('UTC').dt.tz_localize(None)
        return res


def _load_fits_file(path: str, flux_col_name: str) -> pd.DataFrame:
    from astropy.io import fits
    
    try:
        hdul = fits.open(path)
    except Exception as e:
        raise ValueError(f"Failed to open FITS file {path}: {str(e)}")
        
    try:
        table_hdu = None
        for hdu in hdul:
            if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                table_hdu = hdu
                break
                
        if table_hdu is None:
            raise ValueError(f"No table HDU found in FITS file {path}")
            
        data = table_hdu.data
        if data is None or len(data) == 0:
            raise ValueError("FITS table data is empty")
            
        df = pd.DataFrame(np.array(data))
        if df.empty:
            raise ValueError("FITS table data is empty")
            
        for col in df.columns:
            series = df[col]
            if hasattr(series.dtype, 'byteorder') and series.dtype.byteorder not in ('=', '|'):
                try:
                    df[col] = series.astype(series.dtype.newbyteorder('='))
                except Exception:
                    pass
                    
        time_col = _find_column(df.columns, ['timestamp', 'time', 'time_utc', 'utc_time', 'mjd', 'jd', 'date_obs', 'time_sec', 'sec', 'utc'])
        flux_col = _find_column(df.columns, [flux_col_name] + ['flux', 'rate', 'count_rate', 'counts', 'counts_sec', 'counts_s', 'sf_flux', 'sf_rate', 'hd_flux', 'hd_rate', 'y_flux', 'intensity'])
        ar_col = _find_column(df.columns, ['active_region', 'ar', 'region', 'active_region_id'])
        
        if time_col is None:
            raise ValueError(f"Time column not found in FITS file {path}")
        if flux_col is None:
            raise ValueError(f"Flux column not found in FITS file {path}")
            
        time_series = _parse_time_column(df[time_col], hdul, table_hdu)
        
        active_region_val = 'Unknown'
        if ar_col is not None:
            ar_series = df[ar_col]
            if ar_series.dtype == object or ar_series.dtype.kind in ('S', 'V'):
                ar_series = ar_series.apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x))
        else:
            for hdr_key in ['ACTIVE_REGION', 'ACTIVE-REGION', 'AR', 'OBJECT', 'TARGET']:
                val = get_header_value(hdul, table_hdu, hdr_key)
                if val is not None:
                    if isinstance(val, bytes):
                        active_region_val = val.decode('utf-8')
                    else:
                        active_region_val = str(val)
                    break
            ar_series = pd.Series([active_region_val] * len(df))
            
        flux_series = df[flux_col]
        
        # Handle and drop NaT timestamps
        valid_mask = time_series.notna()
        time_series = time_series[valid_mask]
        flux_series = flux_series[valid_mask]
        ar_series = ar_series[valid_mask]
        if len(time_series) == 0:
            raise ValueError("File contains only invalid/NaT timestamps")
            
        new_df = pd.DataFrame({
            flux_col_name: flux_series.values,
            'active_region': ar_series.values
        }, index=time_series)
        new_df.index.name = 'timestamp'
        new_df = new_df.sort_index()
        return new_df
        
    finally:
        hdul.close()


def _load_csv_file(path: str, flux_col_name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to read file {path}: {str(e)}")
        
    if df.empty:
        raise ValueError(f"File {path} is empty")
        
    time_col = _find_column(df.columns, ['timestamp', 'time', 'time_utc', 'utc_time', 'mjd', 'jd', 'date_obs', 'time_sec', 'sec', 'utc'])
    flux_col = _find_column(df.columns, [flux_col_name] + ['flux', 'rate', 'count_rate', 'counts', 'counts_sec', 'counts_s', 'sf_flux', 'sf_rate', 'hd_flux', 'hd_rate', 'y_flux', 'intensity'])
    ar_col = _find_column(df.columns, ['active_region', 'ar', 'region', 'active_region_id'])
    
    if time_col is None:
        raise ValueError(f"File {path} is missing 'timestamp' column")
    if flux_col is None:
        raise ValueError(f"File {path} is missing flux column ('{flux_col_name}' or 'flux')")
    if ar_col is None:
        raise ValueError(f"File {path} is missing 'active_region' column")
        
    time_series = _parse_time_column(df[time_col])
    flux_series = df[flux_col]
    ar_series = df[ar_col]
    
    if ar_series.dtype == object or ar_series.dtype.kind in ('S', 'V'):
        ar_series = ar_series.apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x))
        
    # Handle and drop NaT timestamps
    valid_mask = time_series.notna()
    time_series = time_series[valid_mask]
    flux_series = flux_series[valid_mask]
    ar_series = ar_series[valid_mask]
    if len(time_series) == 0:
        raise ValueError("File contains only invalid/NaT timestamps")
        
    new_df = pd.DataFrame({
        flux_col_name: flux_series.values,
        'active_region': ar_series.values
    }, index=time_series)
    new_df.index.name = 'timestamp'
    new_df = new_df.sort_index()
    return new_df


def _load_raw_file(path: str, flux_col_name: str) -> pd.DataFrame:
    file_type = _detect_file_type(path)
    if file_type == 'fits':
        return _load_fits_file(path, flux_col_name)
    else:
        return _load_csv_file(path, flux_col_name)


def load_data(
    solexs_path: str,
    hel1os_path: str,
    solexs_params: dict = None,
    hel1os_params: dict = None
) -> pd.DataFrame:
    """
    Loads, cleans, aligns, and merges SoLEXS and HEL1OS data.
    """
    # 1. Load files
    solexs_df = _load_raw_file(solexs_path, 'soft_xray')
    hel1os_df = _load_raw_file(hel1os_path, 'hard_xray')
    
    if solexs_df.empty or hel1os_df.empty:
        raise ValueError("SoLEXS or HEL1OS data is empty")
        
    if solexs_df.index.max() < hel1os_df.index.min() or hel1os_df.index.max() < solexs_df.index.min():
        raise ValueError("SoLEXS and HEL1OS time ranges are completely disjoint")
        
    # 2. Get parameters
    s_params = {
        'epsilon_zero': 1e-5,
        'k': 15,
        'theta_active': 10.0,
        'z_threshold': 5.0,
        'z_step_threshold': 5.0,
        'theta_min_step': 5.0
    }
    if solexs_params:
        s_params.update(solexs_params)
        
    h_params = {
        'epsilon_zero': 1e-5,
        'k': 15,
        'theta_active': 1.0,
        'z_threshold': 5.0,
        'z_step_threshold': 5.0,
        'theta_min_step': 1.0
    }
    if hel1os_params:
        h_params.update(hel1os_params)
        
    # 3. Clean independently
    # SoLEXS
    s_flux = solexs_df['soft_xray']
    s_flux = clean_zero_dropouts(s_flux, s_params['epsilon_zero'], s_params['k'], s_params['theta_active'])
    s_flux = clean_transient_spikes(s_flux, s_params['z_threshold'], s_params['k'])
    s_flux = correct_baseline_resets(s_flux, s_params['z_step_threshold'], s_params['theta_min_step'], s_params['k'])
    solexs_df['soft_xray'] = s_flux
    
    # HEL1OS
    h_flux = hel1os_df['hard_xray']
    h_flux = clean_zero_dropouts(h_flux, h_params['epsilon_zero'], h_params['k'], h_params['theta_active'])
    h_flux = clean_transient_spikes(h_flux, h_params['z_threshold'], h_params['k'])
    h_flux = correct_baseline_resets(h_flux, h_params['z_step_threshold'], h_params['theta_min_step'], h_params['k'])
    hel1os_df['hard_xray'] = h_flux
    
    # 4. Temporal alignment
    t_start = min(solexs_df.index.min(), hel1os_df.index.min()).floor("10S")
    t_end = max(solexs_df.index.max(), hel1os_df.index.max()).ceil("10S")
    
    if (t_end - t_start) > pd.Timedelta(days=30):
        raise ValueError(f"Time range duration ({t_end - t_start}) is too large (exceeds 30 days), indicating potential epoch or file mismatch")
        
    target_index = pd.date_range(start=t_start, end=t_end, freq="10S")
    
    # Resample SoLEXS
    solexs_resampled = solexs_df['soft_xray'].resample('10S', closed='left', label='left').mean().reindex(target_index)
    solexs_ar = solexs_df['active_region'].resample('10S', closed='left', label='left').first().reindex(target_index)
    
    # Resample HEL1OS
    hel1os_resampled = hel1os_df['hard_xray'].resample('10S', closed='left', label='left').mean().reindex(target_index)
    hel1os_ar = hel1os_df['active_region'].resample('10S', closed='left', label='left').first().reindex(target_index)
    
    # 5. Handle small/large gaps for flux
    solexs_resampled = interpolate_small_gaps(solexs_resampled, limit=3)
    hel1os_resampled = interpolate_small_gaps(hel1os_resampled, limit=3)
    
    # 6. Active region combination and filling
    active_region = solexs_ar.combine_first(hel1os_ar).ffill().bfill()
    
    # 7. Merge into final DataFrame
    merged_df = pd.DataFrame({
        'soft_xray': solexs_resampled,
        'hard_xray': hel1os_resampled,
        'active_region': active_region
    })
    
    merged_df.index.name = 'timestamp'
    return merged_df
