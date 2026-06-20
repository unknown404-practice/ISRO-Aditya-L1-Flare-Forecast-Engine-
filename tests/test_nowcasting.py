import numpy as np
import pandas as pd
import pytest
from aditya_l1_flare_pipeline.nowcasting.background import subtract_background
from aditya_l1_flare_pipeline.nowcasting.trigger import detect_events
from aditya_l1_flare_pipeline.nowcasting.catalogue import generate_catalogue, get_goes_class

def test_subtract_background_basic():
    flux = pd.Series([1.0, 1.0, 1.0, 5.0, 1.0, 1.0])
    subtracted = subtract_background(flux)
    assert len(subtracted) == len(flux)
    assert subtracted.iloc[3] > 0.0

def test_subtract_background_const_zero():
    flux = pd.Series([0.0] * 10)
    subtracted = subtract_background(flux)
    assert (subtracted == 0.0).all()

def test_subtract_background_nans():
    flux = pd.Series([1.0, np.nan, 1.0, 10.0, 1.0, np.nan])
    subtracted = subtract_background(flux)
    assert np.isnan(subtracted.iloc[1])
    assert np.isnan(subtracted.iloc[5])
    assert subtracted.iloc[3] > 0.0

def test_detect_events_basic():
    deriv = pd.Series([0.0, 0.0, 5.0, 0.0, 0.0])
    events = detect_events(deriv, threshold=1.0)
    assert len(events) == 1
    assert events[0]['peak'] == 2

def test_detect_events_noise():
    deriv = pd.Series([0.01, -0.02, 0.05, -0.04, 0.03] * 100)
    events = detect_events(deriv, threshold=1.0)
    assert len(events) == 0

def test_detect_events_merge():
    deriv = pd.Series([0.0] * 30)
    deriv.iloc[5] = 5.0
    deriv.iloc[10] = 4.0
    events = detect_events(deriv, threshold=1.0)
    assert len(events) == 1

def test_goes_classification():
    assert get_goes_class(5e-8).startswith("A")
    assert get_goes_class(5e-7).startswith("B")
    assert get_goes_class(5e-6).startswith("C")
    assert get_goes_class(5e-5).startswith("M")
    assert get_goes_class(5e-4).startswith("X")
    assert get_goes_class(1e-4) == "X1.0"
    assert get_goes_class(1e-5) == "M1.0"

def test_generate_catalogue_empty():
    df = pd.DataFrame(columns=['soft_xray', 'active_region'])
    cat = generate_catalogue(df, [])
    assert cat.empty
    assert 'goes_class' in cat.columns

def test_subtract_background_const_zero_with_nans():
    flux = pd.Series([0.0, np.nan, 0.0, np.nan, 0.0])
    subtracted = subtract_background(flux)
    assert len(subtracted) == len(flux)
    assert np.isnan(subtracted.iloc[1])
    assert np.isnan(subtracted.iloc[3])
    assert subtracted.iloc[0] == 0.0
    assert subtracted.iloc[2] == 0.0
    assert subtracted.iloc[4] == 0.0

def test_detect_events_unsorted_index():
    index = [3, 1, 4, 2]
    deriv = pd.Series([0.0, 5.0, 0.0, 0.0], index=index)
    events = detect_events(deriv, threshold=1.0)
    assert len(events) == 1
    assert events[0]['peak'] == 1

def test_detect_events_numeric_index_seconds():
    index = [0.0, 10.0, 20.0, 30.0, 100.0, 110.0, 120.0]
    deriv = pd.Series([0.0, 5.0, 0.0, 0.0, 5.0, 0.0, 0.0], index=index)
    events = detect_events(deriv, threshold=1.0)
    assert len(events) == 1
    
    index2 = [0.0, 10.0, 20.0, 30.0, 200.0, 210.0, 220.0]
    deriv2 = pd.Series([0.0, 5.0, 0.0, 0.0, 5.0, 0.0, 0.0], index=index2)
    events2 = detect_events(deriv2, threshold=1.0)
    assert len(events2) == 2

def test_get_goes_class_boundary_rounding():
    assert get_goes_class(9.96e-8) == 'B1.0'
    assert get_goes_class(9.96e-7) == 'C1.0'
    assert get_goes_class(9.96e-6) == 'M1.0'
    assert get_goes_class(9.96e-5) == 'X1.0'

def test_generate_catalogue_empty_and_missing_times():
    times = pd.date_range("2026-06-19 10:00:00", periods=10, freq="10S")
    solexs_df = pd.DataFrame({"soft_xray": [1e-6] * 10, "active_region": ["AR1"] * 10}, index=times)
    events = [
        {},
        {"start": None, "end": times[4]},
        {"start": times[1], "end": None},
        {"start": times[1], "peak": times[2], "end": times[3]}
    ]
    cat = generate_catalogue(solexs_df, events)
    assert len(cat) == 1
    assert cat.iloc[0]['start_time'] == times[1]
    assert cat.iloc[0]['end_time'] == times[3]

def test_generate_catalogue_multiindex():
    times = pd.date_range("2026-06-19 10:00:00", periods=5, freq="10S")
    ar_names = ["AR1", "AR1", "AR1", "AR1", "AR1"]
    index = pd.MultiIndex.from_arrays([times, ar_names], names=["time", "active_region"])
    
    solexs_df = pd.DataFrame({
        "soft_xray": [1e-7, 2e-7, 1e-6, 5e-7, 1e-7]
    }, index=index)
    
    events = [{"start": times[0], "end": times[4]}]
    
    cat = generate_catalogue(solexs_df, events)
    assert len(cat) == 1
    assert cat.iloc[0]['start_time'] == times[0]
    assert cat.iloc[0]['peak_time'] == times[2]
    assert cat.iloc[0]['end_time'] == times[4]
    assert cat.iloc[0]['peak_flux'] == 1e-6
    assert cat.iloc[0]['goes_class'] == "C1.0"
    assert cat.iloc[0]['active_region'] == "AR1"
