import numpy as np
import pandas as pd

def soft_xray_flare_profile_arr(t, t_start_f, tau_rise, tau_decay, A):
    """
    Computes double-exponential soft X-ray flare profile.
    """
    t_peak_offset = (tau_rise * tau_decay) / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
    denom = np.exp(-t_peak_offset / tau_decay) - np.exp(-t_peak_offset / tau_rise)
    
    mask = t >= t_start_f
    val = np.zeros_like(t, dtype=float)
    dt = t[mask] - t_start_f
    val[mask] = A * (np.exp(-dt / tau_decay) - np.exp(-dt / tau_rise)) / denom
    return val

def soft_xray_flare_derivative_arr(t, t_start_f, tau_rise, tau_decay, A):
    """
    Computes analytical derivative of the soft X-ray flare profile.
    """
    t_peak_offset = (tau_rise * tau_decay) / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
    denom = np.exp(-t_peak_offset / tau_decay) - np.exp(-t_peak_offset / tau_rise)
    
    mask = t >= t_start_f
    val = np.zeros_like(t, dtype=float)
    dt = t[mask] - t_start_f
    val[mask] = A * ((-1.0 / tau_decay) * np.exp(-dt / tau_decay) + (1.0 / tau_rise) * np.exp(-dt / tau_rise)) / denom
    return val

def hard_xray_flare_profile_arr(t, t_start_f, tau_rise, tau_decay, A, K, rng, bursts=None):
    """
    Computes hard X-ray flare profile using the Neupert effect and short bursts.
    """
    deriv = soft_xray_flare_derivative_arr(t, t_start_f, tau_rise, tau_decay, A)
    neupert = K * np.maximum(0.0, deriv)
    
    t_peak_offset = (tau_rise * tau_decay) / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
    t_peak = t_start_f + t_peak_offset
    
    if bursts is None:
        num_bursts = rng.randint(3, 6)
        bursts = []
        max_neupert = np.max(neupert) if np.max(neupert) > 0 else 1.0
        for _ in range(num_bursts):
            t_H = rng.uniform(t_start_f, t_peak)
            sigma_H = rng.uniform(2.0, 10.0)
            amp_H = rng.uniform(0.1, 0.5) * max_neupert
            bursts.append((t_H, sigma_H, amp_H))
            
    burst_sum = np.zeros_like(t, dtype=float)
    for t_H, sigma_H, amp_H in bursts:
        burst_sum += amp_H * np.exp(-((t - t_H) ** 2) / (2 * (sigma_H ** 2)))
        
    return neupert + burst_sum

def get_active_region(t, total_seconds):
    if t <= 0.4 * total_seconds:
        return "AR12734"
    elif t <= 0.8 * total_seconds:
        return "AR12735"
    else:
        return "AR12736"

def generate_mock_data(
    solexs_path: str,
    hel1os_path: str,
    start_time: str = "2026-06-19T00:00:00",
    end_time: str = "2026-06-19T04:00:00",
    seed: int = 42,
    flare_parameters: list = None
):
    """
    Generates mock Level-1 SoLEXS and HEL1OS data with physical curves, background, and anomalies.
    """
    t_start_dt = pd.to_datetime(start_time)
    t_end_dt = pd.to_datetime(end_time)
    total_seconds = (t_end_dt - t_start_dt).total_seconds()
    
    rng = np.random.RandomState(seed)
    
    # 1. Generate timestamps (SoLEXS: ~1.2s cadence, HEL1OS: ~0.8s cadence)
    solexs_times = []
    curr_t = 0.0
    while curr_t <= total_seconds:
        solexs_times.append(curr_t)
        curr_t += rng.normal(1.2, 0.1)
    solexs_times = np.array(solexs_times)
    
    hel1os_times = []
    curr_t = 0.0
    while curr_t <= total_seconds:
        hel1os_times.append(curr_t)
        curr_t += rng.normal(0.8, 0.08)
    hel1os_times = np.array(hel1os_times)
    
    solexs_datetimes = t_start_dt + pd.to_timedelta(solexs_times, unit='s')
    hel1os_datetimes = t_start_dt + pd.to_timedelta(hel1os_times, unit='s')
    
    # 2. Flare parameters
    if flare_parameters is None:
        flare_params = [
            {
                'start_time_rel': 3600.0,
                'tau_rise': 120.0,
                'tau_decay': 600.0,
                'amplitude': 500.0
            },
            {
                'start_time_rel': 9000.0,
                'tau_rise': 180.0,
                'tau_decay': 900.0,
                'amplitude': 800.0
            }
        ]
    else:
        flare_params = []
        for fp in flare_parameters:
            if isinstance(fp.get('start_time'), str):
                start_f_dt = pd.to_datetime(fp['start_time'])
                start_time_rel = (start_f_dt - t_start_dt).total_seconds()
            else:
                start_time_rel = fp.get('start_time_rel', fp.get('start_time', 3600.0))
            
            flare_params.append({
                'start_time_rel': start_time_rel,
                'tau_rise': fp.get('tau_rise', 120.0),
                'tau_decay': fp.get('tau_decay', 600.0),
                'amplitude': fp.get('amplitude', 500.0)
            })
            
    # 3. Generate SoLEXS
    # Quiet-Sun Background soft
    # B_soft(t) = B0 + B1 * sin(2*pi*t / 86400) + eta_red(t)
    solexs_diffs = np.diff(solexs_times, prepend=0.0)
    solexs_red_steps = rng.normal(0.0, 0.05 * np.sqrt(solexs_diffs))
    solexs_eta_red = np.cumsum(solexs_red_steps)
    solexs_bg = 100.0 + 10.0 * np.sin(2.0 * np.pi * solexs_times / 86400.0) + solexs_eta_red
    
    # Flares
    solexs_flares = np.zeros_like(solexs_times)
    for fp in flare_params:
        solexs_flares += soft_xray_flare_profile_arr(
            solexs_times,
            fp['start_time_rel'],
            fp['tau_rise'],
            fp['tau_decay'],
            fp['amplitude']
        )
    solexs_clean = solexs_bg + solexs_flares
    
    # Add Gaussian noise
    solexs_flux = solexs_clean + rng.normal(0.0, 0.5, size=len(solexs_clean))
    
    # Resets (Step changes)
    resets_solexs = rng.uniform(0, 1, size=len(solexs_flux)) < 0.0001
    reset_shifts_solexs = np.zeros_like(solexs_flux)
    curr_shift = 0.0
    for idx in range(len(solexs_flux)):
        if resets_solexs[idx]:
            curr_shift += rng.uniform(-50.0, 200.0)
        reset_shifts_solexs[idx] = curr_shift
    solexs_flux += reset_shifts_solexs
    
    # Spikes
    spikes_solexs = rng.uniform(0, 1, size=len(solexs_flux)) < 0.001
    solexs_flux[spikes_solexs] += rng.uniform(500.0, 2000.0, size=np.sum(spikes_solexs))
    
    # Zero dropouts
    idx = 0
    while idx < len(solexs_flux):
        if rng.uniform(0, 1) < 0.002:
            dropout_len = rng.randint(1, 6)
            solexs_flux[idx : idx + dropout_len] = 0.0
            idx += dropout_len
        else:
            idx += 1
            
    # NaN gaps
    idx = 0
    while idx < len(solexs_flux):
        if rng.uniform(0, 1) < 0.0005:
            gap_len = rng.randint(10, 101)
            solexs_flux[idx : idx + gap_len] = np.nan
            idx += gap_len
        else:
            idx += 1
            
    solexs_ar = [get_active_region(t, total_seconds) for t in solexs_times]
    
    # 4. Generate HEL1OS
    # Quiet-Sun Background hard
    hel1os_bg = 5.0 + rng.normal(0.0, 0.1, size=len(hel1os_times))
    
    # Resets (Step changes)
    resets_hel1os = rng.uniform(0, 1, size=len(hel1os_times)) < 0.0001
    reset_shifts_hel1os = np.zeros_like(hel1os_times)
    curr_shift = 0.0
    for idx in range(len(hel1os_times)):
        if resets_hel1os[idx]:
            curr_shift += rng.uniform(-2.5, 10.0)
        reset_shifts_hel1os[idx] = curr_shift
    hel1os_bg += reset_shifts_hel1os
    
    # Flares
    hel1os_flares = np.zeros_like(hel1os_times)
    for fp in flare_params:
        hel1os_flares += hard_xray_flare_profile_arr(
            hel1os_times,
            fp['start_time_rel'],
            fp['tau_rise'],
            fp['tau_decay'],
            fp['amplitude'],
            100.0,
            rng
        )
    hel1os_flux = hel1os_bg + hel1os_flares
    
    # Spikes
    spikes_hel1os = rng.uniform(0, 1, size=len(hel1os_flux)) < 0.001
    hel1os_flux[spikes_hel1os] += rng.uniform(50.0, 200.0, size=np.sum(spikes_hel1os))
    
    # Zero dropouts
    idx = 0
    while idx < len(hel1os_flux):
        if rng.uniform(0, 1) < 0.002:
            dropout_len = rng.randint(1, 6)
            hel1os_flux[idx : idx + dropout_len] = 0.0
            idx += dropout_len
        else:
            idx += 1
            
    # NaN gaps
    idx = 0
    while idx < len(hel1os_flux):
        if rng.uniform(0, 1) < 0.0005:
            gap_len = rng.randint(10, 101)
            hel1os_flux[idx : idx + gap_len] = np.nan
            idx += gap_len
        else:
            idx += 1
            
    hel1os_ar = [get_active_region(t, total_seconds) for t in hel1os_times]
    
    # Save files
    solexs_df = pd.DataFrame({
        'timestamp': solexs_datetimes,
        'soft_xray': solexs_flux,
        'active_region': solexs_ar
    })
    
    hel1os_df = pd.DataFrame({
        'timestamp': hel1os_datetimes,
        'hard_xray': hel1os_flux,
        'active_region': hel1os_ar
    })
    
    solexs_df.to_csv(solexs_path, index=False)
    hel1os_df.to_csv(hel1os_path, index=False)
