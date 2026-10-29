"""
test_pipeline.py — Unit & Integration Tests
Indonesian Food Commodity Price Intelligence Platform
"""

import os
import sys
import pandas as pd
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data_loader import get_clean_data, clean_data, read_html_xls
from src.analyzer import (
    compute_pivot,
    compute_inflation_yoy,
    compute_cagr,
    compute_volatility,
    perform_kmeans_clustering,
    perform_price_forecasting
)


def test_data_ingestion_and_cleaning():
    """Test 1: Memverifikasi integritas pemuatan dan pembersihan dataset."""
    df = get_clean_data(save_csv=False)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 33, f"Expected 33 rows, got {len(df)}"
    assert set(df.columns) == {"periode_data", "komoditas", "harga_rata_rata"}
    
    # Check 11 unique commodities
    assert df["komoditas"].nunique() == 11
    
    # Check 3 years
    assert sorted(df["periode_data"].unique()) == [2023, 2024, 2025]
    
    # Check typo correction
    assert "CABAI RAWIT MERHA" not in df["komoditas"].values
    assert "CABAI RAWIT MERAH" in df["komoditas"].values
    
    # Check prices are positive floats
    assert (df["harga_rata_rata"] > 0).all()
    assert df["harga_rata_rata"].isnull().sum() == 0


def test_pivot_matrix_shape():
    """Test 2: Memverifikasi bentuk matriks pivot komoditas x tahun."""
    df = get_clean_data(save_csv=False)
    pivot = compute_pivot(df)
    
    assert pivot.shape == (11, 3)
    assert list(pivot.columns) == [2023, 2024, 2025]
    assert "BERAS IR I" in pivot.index
    assert "DAGING SAPI" in pivot.index


def test_inflation_yoy_calculation():
    """Test 3: Memverifikasi akurasi matematis inflasi Year-over-Year."""
    df = get_clean_data(save_csv=False)
    pivot = compute_pivot(df)
    yoy = compute_inflation_yoy(pivot)
    
    assert "YoY_2023_2024_pct" in yoy.columns
    assert "YoY_2024_2025_pct" in yoy.columns
    assert len(yoy) == 11
    
    # Sample check Beras IR I: 12,759.81 in 2023 -> 14,734.00 in 2024
    expected_yoy = round(((14734.0 - 12759.81) / 12759.81) * 100, 2)
    actual_yoy = yoy.loc["BERAS IR I", "YoY_2023_2024_pct"]
    assert np.isclose(actual_yoy, expected_yoy, atol=0.01)


def test_cagr_calculation():
    """Test 4: Memverifikasi kalkulasi Compound Annual Growth Rate."""
    df = get_clean_data(save_csv=False)
    pivot = compute_pivot(df)
    cagr = compute_cagr(pivot)
    
    assert len(cagr) == 11
    # Bawang Merah should have highest CAGR
    assert cagr.idxmax() == "BAWANG MERAH"
    assert cagr["BAWANG MERAH"] > 10.0


def test_kmeans_clustering():
    """Test 5: Memverifikasi clustering K-Means dan evaluasi silhouette."""
    df = get_clean_data(save_csv=False)
    pivot = compute_pivot(df)
    res = perform_kmeans_clustering(pivot, max_k=5)
    
    assert "best_k" in res
    assert res["best_k"] in [2, 3, 4]
    assert res["silhouette_scores"][res["best_k"]] > 0.50
    assert "segment_name" in res["clustered_df"].columns
    assert len(res["clustered_df"]) == 11


def test_price_forecasting():
    """Test 6: Memverifikasi model peramalan linear harga tahun 2026."""
    df = get_clean_data(save_csv=False)
    forecast = perform_price_forecasting(df, target_year=2026)
    
    assert len(forecast) == 11
    assert "prediksi_2026" in forecast.columns
    assert "r2" in forecast.columns
    assert "mape_pct" in forecast.columns
    
    # All predictions must be positive
    assert (forecast["prediksi_2026"] > 0).all()
    # Mean MAPE should be under 5%
    assert forecast["mape_pct"].mean() < 5.0
