import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from aditya_l1_flare_pipeline.data.simulator import (
    soft_xray_flare_profile_arr,
    hard_xray_flare_profile_arr,
    generate_mock_data
)
from aditya_l1_flare_pipeline.data.ingest import (
    load_data,
    clean_transient_spikes,
    correct_baseline_resets,
    interpolate_small_gaps
)

def test_double_exponential_flare_shape():
    """
    Verify that the double-exponential flare shape has its peak at the mathematically
    expected timestamp and its maximum amplitude is exactly the target amplitude.
    """
    t = np.linspace(0, 1000, 10001)
    t_start = 100.0
    tau_rise = 50.0
    tau_decay = 200.0
    A = 10.0
    
    # Analytical offset: (tau_rise * tau_decay) / (tau_decay - tau_rise) * ln(tau_decay / tau_rise)
    expected_peak_offset = (50.0 * 200.0) / 150.0 * np.log(4)
    expected_peak_t = t_start + expected_peak_offset
    
    profile = soft_xray_flare_profile_arr(t, t_start, tau_rise, tau_decay, A)
    
    peak_idx = np.argmax(profile)
    peak_t = t[peak_idx]
    
    assert np.abs(peak_t - expected_peak_t) < 0.2
    assert np.abs(np.max(profile) - A) < 1e-5

def test_neupert_effect_relation():
    """
    Verify that the simulated HEL1OS flare profile matches the derivative
    of the SoLEXS curve during the rise phase (correlation coefficient > 0.99).
    """
    t = np.linspace(0, 1000, 10001)
    t_start = 100.0
    tau_rise = 50.0
    tau_decay = 200.0
    A = 10.0
    K = 100.0
    rng = np.random.RandomState(42)
    
    t_peak_offset = (50.0 * 200.0) / 150.0 * np.log(4)
    t_peak = t_start + t_peak_offset
    
    # Compute hard X-ray profile without random bursts
    hard_profile = hard_xray_flare_profile_arr(t, t_start, tau_rise, tau_decay, A, K, rng, bursts=[])
    
    # Calculate numerical derivative of soft_profile
    soft_profile = soft_xray_flare_profile_arr(t, t_start, tau_rise, tau_decay, A)
    dt = t[1] - t[0]
    num_deriv = np.gradient(soft_profile, dt)
    
    # Limit to rise phase
    rise_mask = (t >= t_start) & (t <= t_peak)
    
    corr = np.corrcoef(hard_profile[rise_mask], num_deriv[rise_mask])[0, 1]
    assert corr > 0.99

def test_load_data_interface_contract():
    """
    Verify that load_data conforms to the interface contract of returning a DataFrame
    with specific DatetimeIndex, frequency, and columns.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        solexs_path = os.path.join(tmpdir, "solexs.csv")
        hel1os_path = os.path.join(tmpdir, "hel1os.csv")
        
        generate_mock_data(
            solexs_path,
            hel1os_path,
            start_time="2026-06-19T00:00:00",
            end_time="2026-06-19T01:00:00",
            seed=42
        )
        
        df = load_data(solexs_path, hel1os_path)
        
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        
        # Verify the cadence of index differences is exactly 10 seconds
        diffs = np.diff(df.index).astype('timedelta64[s]')
        assert np.all(diffs == np.timedelta64(10, 's'))
        
        assert list(df.columns) == ['soft_xray', 'hard_xray', 'active_region']

def test_cadence_resampling_irregular_to_regular():
    """
    Verify that after running load_data on irregular cadence raw data,
    the output timestamps are perfectly rounded to 10-second boundaries.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        solexs_path = os.path.join(tmpdir, "solexs.csv")
        hel1os_path = os.path.join(tmpdir, "hel1os.csv")
        
        generate_mock_data(
            solexs_path,
            hel1os_path,
            start_time="2026-06-19T00:00:00",
            end_time="2026-06-19T00:30:00",
            seed=42
        )
        
        df = load_data(solexs_path, hel1os_path)
        
        for ts in df.index:
            assert ts.second % 10 == 0
            assert ts.microsecond == 0
            assert ts.nanosecond == 0

def test_empty_files_handling():
    """
    Verify that load_data raises a ValueError when loading empty files.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        solexs_path = os.path.join(tmpdir, "solexs_empty.csv")
        hel1os_path = os.path.join(tmpdir, "hel1os_empty.csv")
        
        with open(solexs_path, 'w') as f:
            f.write("timestamp,soft_xray,active_region\n")
        with open(hel1os_path, 'w') as f:
            f.write("timestamp,hard_xray,active_region\n")
            
        with pytest.raises(ValueError) as excinfo:
            load_data(solexs_path, hel1os_path)
        
        assert "empty" in str(excinfo.value).lower()

def test_all_nan_series():
    """
    Verify that the pipeline handles all-NaN series correctly without crashing
    and preserves active region metadata.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        solexs_path = os.path.join(tmpdir, "solexs_nan.csv")
        hel1os_path = os.path.join(tmpdir, "hel1os_nan.csv")
        
        t_index = pd.date_range("2026-06-19T00:00:00", "2026-06-19T00:10:00", freq="1S")
        solexs_df = pd.DataFrame({
            'timestamp': t_index,
            'soft_xray': [np.nan] * len(t_index),
            'active_region': ["AR12734"] * len(t_index)
        })
        hel1os_df = pd.DataFrame({
            'timestamp': t_index,
            'hard_xray': [np.nan] * len(t_index),
            'active_region': ["AR12734"] * len(t_index)
        })
        
        solexs_df.to_csv(solexs_path, index=False)
        hel1os_df.to_csv(hel1os_path, index=False)
        
        df = load_data(solexs_path, hel1os_path)
        
        assert df['soft_xray'].isna().all()
        assert df['hard_xray'].isna().all()
        assert (df['active_region'] == "AR12734").all()

def test_huge_baseline_step():
    """
    Verify that baseline shift correction successfully removes massive resets
    and restores baseline back to quiet-Sun levels.
    """
    rng = np.random.RandomState(42)
    n_points = 500
    noise = rng.normal(0.0, 0.05, size=n_points)
    base = 1.0
    
    # Step change at t=250
    step = np.zeros(n_points)
    step[250:] = 999.0
    
    raw_series = pd.Series(base + noise + step)
    
    corrected = correct_baseline_resets(raw_series, z_step_threshold=5.0, theta_min_step=5.0, k=15)
    
    mean_before = corrected.iloc[:250].mean()
    mean_after = corrected.iloc[250:].mean()
    
    assert np.abs(mean_before - 1.0) < 0.02
    assert np.abs(mean_after - 1.0) < 0.02
    assert np.abs(corrected.std() - 0.05) < 0.01

def test_transient_spike_rejection_during_flare_rise():
    """
    Verify that transient spikes on a rising profile are rejected
    and interpolated back near their true smooth values.
    """
    t = np.linspace(0, 100, 101)
    smooth_rise = np.exp(t / 10.0)
    
    raw = smooth_rise.copy()
    raw[50] += 500.0
    
    series = pd.Series(raw)
    cleaned = clean_transient_spikes(series, z_threshold=5.0, k=10)
    
    assert np.isnan(cleaned.iloc[50])
    assert not np.isnan(cleaned.iloc[49])
    assert not np.isnan(cleaned.iloc[51])
    
    interpolated = interpolate_small_gaps(cleaned, limit=3)
    expected_val = (smooth_rise[49] + smooth_rise[51]) / 2.0
    assert np.abs(interpolated.iloc[50] - expected_val) < 1.0

def test_multiple_consecutive_baseline_resets():
    """
    Verify that cumulative reset corrector aligns multiple steps.
    """
    rng = np.random.RandomState(42)
    n_points = 500
    noise = rng.normal(0.0, 0.05, size=n_points)
    base = 100.0
    
    resets = np.zeros(n_points)
    resets[100:] += 50.0
    resets[200:] -= 80.0
    resets[300:] += 40.0
    
    raw_series = pd.Series(base + noise + resets)
    corrected = correct_baseline_resets(raw_series, z_step_threshold=5.0, theta_min_step=5.0, k=15)
    
    assert corrected.std() < 0.15
    for start, end in [(0, 95), (105, 195), (205, 295), (305, 495)]:
        assert np.abs(corrected.iloc[start:end].mean() - base) < 0.1

def test_large_gap_non_interpolation():
    """
    Verify that small gaps <= 3 bins are interpolated, but large gaps > 3 bins
    are not interpolated and remain NaN.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        solexs_path = os.path.join(tmpdir, "solexs_gap.csv")
        hel1os_path = os.path.join(tmpdir, "hel1os_gap.csv")
        
        times_part1 = pd.date_range("2026-06-19T00:00:00", "2026-06-19T00:30:00", freq="1.2S")
        times_part2 = pd.date_range("2026-06-19T00:45:00", "2026-06-19T01:30:00", freq="1.2S")
        solexs_timestamps = times_part1.union(times_part2)
        
        times_h1 = pd.date_range("2026-06-19T00:00:00", "2026-06-19T00:30:00", freq="0.8S")
        times_h2 = pd.date_range("2026-06-19T00:45:00", "2026-06-19T01:30:00", freq="0.8S")
        hel1os_timestamps = times_h1.union(times_h2)
        
        solexs_df = pd.DataFrame({
            'timestamp': solexs_timestamps,
            'soft_xray': np.random.normal(100.0, 0.5, size=len(solexs_timestamps)),
            'active_region': ["AR12734"] * len(solexs_timestamps)
        })
        
        hel1os_df = pd.DataFrame({
            'timestamp': hel1os_timestamps,
            'hard_xray': np.random.normal(5.0, 0.1, size=len(hel1os_timestamps)),
            'active_region': ["AR12734"] * len(hel1os_timestamps)
        })
        
        solexs_df.to_csv(solexs_path, index=False)
        hel1os_df.to_csv(hel1os_path, index=False)
        
        df = load_data(solexs_path, hel1os_path)
        
        # Gap is from 00:30:00 to 00:45:00, which is 15 minutes.
        # This is a large gap (> 30s) and should remain NaN.
        gap_slice = df.loc["2026-06-19T00:31:00":"2026-06-19T00:44:00"]
        
        assert gap_slice['soft_xray'].isna().all()
        assert gap_slice['hard_xray'].isna().all()
        assert (gap_slice['active_region'] == "AR12734").all()


def write_mock_fits(df, path, fits_type, has_ar=True, epoch_headers=True):
    """
    Helper to write mock FITS files for testing.
    """
    from astropy.io import fits
    import numpy as np
    import pandas as pd
    
    cols = []
    
    # Handle time column
    if isinstance(df.index, pd.DatetimeIndex):
        epoch = pd.Timestamp('2020-01-01')
        time_vals = (df.index - epoch).total_seconds().values
    elif 'timestamp' in df.columns:
        epoch = pd.Timestamp('2020-01-01')
        time_vals = (pd.to_datetime(df['timestamp']) - epoch).dt.total_seconds().values
    elif 'time' in df.columns:
        epoch = pd.Timestamp('2020-01-01')
        time_vals = (pd.to_datetime(df['time']) - epoch).dt.total_seconds().values
    else:
        time_vals = np.array(range(len(df)), dtype=float)
        
    cols.append(fits.Column(name='time_sec', format='D', array=time_vals))
    
    # Handle flux column
    flux_src_col = None
    for c in ['soft_xray', 'hard_xray', 'flux', 'rate']:
        if c in df.columns:
            flux_src_col = c
            break
            
    if flux_src_col is not None:
        flux_vals = df[flux_src_col].values
    else:
        flux_vals = np.zeros(len(df), dtype=float)
        
    cols.append(fits.Column(name='rate', format='D', array=flux_vals))
    
    # Handle active region column
    if has_ar:
        ar_src_col = None
        for c in ['active_region', 'ar', 'region', 'active_region_id']:
            if c in df.columns:
                ar_src_col = c
                break
        if ar_src_col is not None:
            ar_vals = df[ar_src_col].values
        else:
            ar_vals = np.array(['AR12734'] * len(df))
            
        max_len = max(len(str(x)) for x in ar_vals) if len(ar_vals) > 0 else 1
        cols.append(fits.Column(name='active_region_id', format=f'{max_len}A', array=ar_vals.astype(f'S{max_len}')))
        
    # Create the HDU
    hdu = fits.BinTableHDU.from_columns(cols)
    
    # Write epoch headers if requested (Aditya-L1 epoch: 2020-01-01, which is MJD 58849.0)
    if epoch_headers:
        hdu.header['MJDREF'] = 58849.0
        hdu.header['MJDREFI'] = 58849
        hdu.header['MJDREFF'] = 0.0
        
    # If has_ar is False, write the active region to header to test the fallback
    if not has_ar:
        hdu.header['ACTIVE_REGION'] = 'AR12734_HEADER'
        
    primary_hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([primary_hdu, hdu])
    hdul.writeto(path, overwrite=True)


def test_load_fits_data():
    """
    Verify that load_data successfully loads FITS files and integrates them.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        solexs_path = os.path.join(tmpdir, "solexs.fits")
        hel1os_path = os.path.join(tmpdir, "hel1os.fits")
        
        times = pd.date_range("2026-06-19T00:00:00", "2026-06-19T00:10:00", freq="1S")
        
        solexs_df = pd.DataFrame({
            'soft_xray': np.random.normal(10.0, 0.1, size=len(times)),
            'active_region': ['AR12734'] * len(times)
        }, index=times)
        
        hel1os_df = pd.DataFrame({
            'hard_xray': np.random.normal(2.0, 0.05, size=len(times)),
            'active_region': ['AR12734'] * len(times)
        }, index=times)
        
        write_mock_fits(solexs_df, solexs_path, 'solexs', has_ar=True)
        write_mock_fits(hel1os_df, hel1os_path, 'hel1os', has_ar=True)
        
        df = load_data(solexs_path, hel1os_path)
        
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert list(df.columns) == ['soft_xray', 'hard_xray', 'active_region']
        assert not df['soft_xray'].isna().all()
        assert not df['hard_xray'].isna().all()
        assert (df['active_region'] == "AR12734").all()
        
        for ts in df.index:
            assert ts.second % 10 == 0


def test_fits_missing_active_region():
    """
    Verify that when FITS table lacks active_region column,
    it falls back to reading active region from the FITS header.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        solexs_path = os.path.join(tmpdir, "solexs_no_ar.fits")
        hel1os_path = os.path.join(tmpdir, "hel1os_no_ar.fits")
        
        times = pd.date_range("2026-06-19T00:00:00", "2026-06-19T00:10:00", freq="1S")
        solexs_df = pd.DataFrame({
            'soft_xray': np.random.normal(10.0, 0.1, size=len(times))
        }, index=times)
        hel1os_df = pd.DataFrame({
            'hard_xray': np.random.normal(2.0, 0.05, size=len(times))
        }, index=times)
        
        write_mock_fits(solexs_df, solexs_path, 'solexs', has_ar=False)
        write_mock_fits(hel1os_df, hel1os_path, 'hel1os', has_ar=False)
        
        df = load_data(solexs_path, hel1os_path)
        
        assert (df['active_region'] == "AR12734_HEADER").all()


def test_fits_empty_file_handling():
    """
    Verify that load_data raises ValueError containing 'empty' when reading empty FITS files.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        solexs_empty_path = os.path.join(tmpdir, "solexs_empty.fits")
        hel1os_empty_path = os.path.join(tmpdir, "hel1os_empty.fits")
        
        with open(solexs_empty_path, 'wb') as f:
            pass
        with open(hel1os_empty_path, 'wb') as f:
            pass
            
        with pytest.raises(ValueError) as excinfo:
            load_data(solexs_empty_path, hel1os_empty_path)
        assert "empty" in str(excinfo.value).lower()


def test_timezone_aware_string_parsing():
    from aditya_l1_flare_pipeline.data.ingest import _parse_time_column
    s = pd.Series(["2026-06-19T10:42:36+05:30", "2026-06-19T10:42:46+05:30"])
    parsed = _parse_time_column(s)
    # Verify it converted timezone-aware datetimes to UTC before stripping the timezone
    assert parsed.dt.tz is None
    expected_0 = pd.Timestamp("2026-06-19T10:42:36+05:30").tz_convert("UTC").tz_localize(None)
    expected_1 = pd.Timestamp("2026-06-19T10:42:46+05:30").tz_convert("UTC").tz_localize(None)
    assert parsed.iloc[0] == expected_0
    assert parsed.iloc[1] == expected_1


def test_julian_date_column_parsing():
    from aditya_l1_flare_pipeline.data.ingest import _parse_time_column
    # Case 1: Column name has 'jd' and not 'mjd'
    s1 = pd.Series([2461212.5, 2461212.6], name='time_jd')
    parsed1 = _parse_time_column(s1)
    # 2461212.5 - 2400000.5 = 61212.0 MJD
    # 61212 MJD = 2026-06-19
    expected_0 = pd.to_datetime(61212.0, unit='D', origin='1858-11-17')
    assert parsed1.iloc[0] == expected_0
    
    # Case 2: Column name doesn't have 'jd' but values are > 2400000
    s2 = pd.Series([2461212.5, 2461212.6], name='some_numeric_time')
    parsed2 = _parse_time_column(s2)
    assert parsed2.iloc[0] == expected_0


def test_disjoint_time_range_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        solexs_path = os.path.join(tmpdir, "solexs_disjoint.csv")
        hel1os_path = os.path.join(tmpdir, "hel1os_disjoint.csv")
        
        # SoLEXS is from 2026-06-19 00:00 to 01:00
        solexs_times = pd.date_range("2026-06-19T00:00:00", "2026-06-19T01:00:00", freq="10S")
        # HEL1OS is from 2026-06-19 02:00 to 03:00
        hel1os_times = pd.date_range("2026-06-19T02:00:00", "2026-06-19T03:00:00", freq="10S")
        
        solexs_df = pd.DataFrame({
            'timestamp': solexs_times,
            'soft_xray': np.random.normal(10.0, 0.1, size=len(solexs_times)),
            'active_region': ['AR12734'] * len(solexs_times)
        })
        hel1os_df = pd.DataFrame({
            'timestamp': hel1os_times,
            'hard_xray': np.random.normal(2.0, 0.05, size=len(hel1os_times)),
            'active_region': ['AR12734'] * len(hel1os_times)
        })
        
        solexs_df.to_csv(solexs_path, index=False)
        hel1os_df.to_csv(hel1os_path, index=False)
        
        with pytest.raises(ValueError) as excinfo:
            load_data(solexs_path, hel1os_path)
        assert "disjoint" in str(excinfo.value).lower()


def test_epoch_mismatch_oom_protection():
    with tempfile.TemporaryDirectory() as tmpdir:
        solexs_path = os.path.join(tmpdir, "solexs_epoch_mismatch.csv")
        hel1os_path = os.path.join(tmpdir, "hel1os_epoch_mismatch.csv")
        
        # SoLEXS starts at 2020-01-01
        solexs_times = pd.date_range("2020-01-01T00:00:00", "2020-01-01T01:00:00", freq="10S")
        # HEL1OS starts at 2026-06-19
        hel1os_times = pd.date_range("2026-06-19T00:00:00", "2026-06-19T01:00:00", freq="10S")
        
        solexs_df = pd.DataFrame({
            'timestamp': solexs_times,
            'soft_xray': np.random.normal(10.0, 0.1, size=len(solexs_times)),
            'active_region': ['AR12734'] * len(solexs_times)
        })
        hel1os_df = pd.DataFrame({
            'timestamp': hel1os_times,
            'hard_xray': np.random.normal(2.0, 0.05, size=len(hel1os_times)),
            'active_region': ['AR12734'] * len(hel1os_times)
        })
        
        solexs_df.to_csv(solexs_path, index=False)
        hel1os_df.to_csv(hel1os_path, index=False)
        
        with pytest.raises(ValueError) as excinfo:
            load_data(solexs_path, hel1os_path)
        assert "too large" in str(excinfo.value).lower() or "exceeds 30 days" in str(excinfo.value).lower()


def test_nat_timestamp_handling():
    with tempfile.TemporaryDirectory() as tmpdir:
        solexs_path = os.path.join(tmpdir, "solexs_nat.csv")
        hel1os_path = os.path.join(tmpdir, "hel1os_nat.csv")
        
        # Generate time index, insert some NaT strings
        solexs_times = ["2026-06-19T00:00:00", "NaT", "2026-06-19T00:00:20"]
        hel1os_times = ["2026-06-19T00:00:00", "2026-06-19T00:00:10", "NaT"]
        
        solexs_df = pd.DataFrame({
            'timestamp': solexs_times,
            'soft_xray': [10.0, 999.0, 10.2],
            'active_region': ['AR12734'] * 3
        })
        hel1os_df = pd.DataFrame({
            'timestamp': hel1os_times,
            'hard_xray': [2.0, 2.1, 999.0],
            'active_region': ['AR12734'] * 3
        })
        
        solexs_df.to_csv(solexs_path, index=False)
        hel1os_df.to_csv(hel1os_path, index=False)
        
        # This should succeed by dropping NaT rows
        df = load_data(solexs_path, hel1os_path)
        assert isinstance(df, pd.DataFrame)
        # The index should be aligned to 10s: 00:00:00, 00:00:10, 00:00:20
        assert len(df) == 3
        assert df.index[0] == pd.Timestamp("2026-06-19T00:00:00")
        assert df.index[1] == pd.Timestamp("2026-06-19T00:00:10")
        assert df.index[2] == pd.Timestamp("2026-06-19T00:00:20")
        
        # Now test file containing only NaT timestamps
        solexs_only_nat_path = os.path.join(tmpdir, "solexs_only_nat.csv")
        solexs_only_nat_df = pd.DataFrame({
            'timestamp': ["NaT", "NaT"],
            'soft_xray': [10.0, 10.2],
            'active_region': ['AR12734'] * 2
        })
        solexs_only_nat_df.to_csv(solexs_only_nat_path, index=False)
        
        with pytest.raises(ValueError) as excinfo:
            load_data(solexs_only_nat_path, hel1os_path)
        assert "only invalid/nat timestamps" in str(excinfo.value).lower()


