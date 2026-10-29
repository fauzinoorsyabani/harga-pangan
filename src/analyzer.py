"""
analyzer.py — Modul Analisis Statistik & Machine Learning
Indonesian Food Commodity Price Intelligence Platform

Menyediakan kalkulasi:
1. Inflasi Year-over-Year (YoY)
2. Compound Annual Growth Rate (CAGR 2023-2025)
3. Volatilitas & Koefisien Variasi
4. K-Means Clustering & Evaluasi Silhouette
5. Time Series Forecasting dengan Linear Regression & Error Metrics (MAE, RMSE, R2, MAPE)
"""

import os
import sys
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Ensure utf-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def compute_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Membuat pivot table komoditas vs tahun."""
    pivot = df.pivot_table(
        index="komoditas",
        columns="periode_data",
        values="harga_rata_rata"
    )
    pivot.columns = [int(c) for c in pivot.columns]
    return pivot


def compute_inflation_yoy(pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung laju inflasi Year-over-Year (YoY) dalam persen.
    YoY % = ((Harga_t - Harga_{t-1}) / Harga_{t-1}) * 100
    """
    yoy_df = pd.DataFrame(index=pivot.index)
    years = sorted(pivot.columns)
    for i in range(1, len(years)):
        prev_yr = years[i - 1]
        curr_yr = years[i]
        col_name = f"YoY_{prev_yr}_{curr_yr}_pct"
        yoy_df[col_name] = ((pivot[curr_yr] - pivot[prev_yr]) / pivot[prev_yr] * 100).round(2)
    return yoy_df


def compute_cagr(pivot: pd.DataFrame) -> pd.Series:
    """
    Menghitung Compound Annual Growth Rate (CAGR) 2023 - 2025.
    CAGR = (Harga_2025 / Harga_2023)^(1 / n_years) - 1
    """
    first_year = pivot.columns.min()
    last_year = pivot.columns.max()
    n_years = last_year - first_year
    cagr = ((pivot[last_year] / pivot[first_year]) ** (1.0 / n_years) - 1.0) * 100
    return cagr.round(2).rename("cagr_pct")


def compute_volatility(pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung rata-rata, standar deviasi, dan Koefisien Variasi (CV %).
    CV = (std / mean) * 100
    """
    stats = pd.DataFrame(index=pivot.index)
    stats["mean_price"] = pivot.mean(axis=1).round(2)
    stats["std_price"] = pivot.std(axis=1).round(2)
    stats["cv_pct"] = ((stats["std_price"] / stats["mean_price"]) * 100).round(2)
    stats["min_price"] = pivot.min(axis=1).round(2)
    stats["max_price"] = pivot.max(axis=1).round(2)
    stats["range_price"] = stats["max_price"] - stats["min_price"]
    return stats


def perform_kmeans_clustering(pivot: pd.DataFrame, max_k: int = 5):
    """
    Melakukan clustering K-Means dengan eksplorasi K=2 hingga max_k
    menggunakan Silhouette Analysis.
    Fitur: [mean_price, cv_pct, total_growth_pct]
    """
    vol = compute_volatility(pivot)
    cagr = compute_cagr(pivot)
    
    first_year = pivot.columns.min()
    last_year = pivot.columns.max()
    total_growth = ((pivot[last_year] - pivot[first_year]) / pivot[first_year] * 100).round(2)
    
    feature_df = pd.DataFrame({
        "mean_price": vol["mean_price"],
        "cv_pct": vol["cv_pct"],
        "total_growth": total_growth
    }, index=pivot.index)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)
    
    silhouette_dict = {}
    for k in range(2, min(max_k + 1, len(pivot))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_dict[k] = round(score, 4)
        
    best_k = max(silhouette_dict, key=silhouette_dict.get)
    
    # Model terbaik
    optimal_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = optimal_km.fit_predict(X_scaled)
    
    clustered_df = feature_df.copy()
    clustered_df["cluster"] = final_labels
    
    # Buat label bisnis untuk tiap cluster berdasarkan mean_price
    cluster_means = clustered_df.groupby("cluster")["mean_price"].mean().sort_values()
    tier_names = ["Tier Terjangkau", "Tier Menengah", "Tier Premium", "Tier Khusus"]
    label_map = {}
    for rank, (c_id, _) in enumerate(cluster_means.items()):
        label_map[c_id] = tier_names[rank] if rank < len(tier_names) else f"Tier {rank+1}"
        
    clustered_df["segment_name"] = clustered_df["cluster"].map(label_map)
    
    return {
        "silhouette_scores": silhouette_dict,
        "best_k": best_k,
        "clustered_df": clustered_df,
        "scaler": scaler,
        "model": optimal_km,
        "scaled_data": X_scaled
    }


def perform_price_forecasting(df: pd.DataFrame, target_year: int = 2026) -> pd.DataFrame:
    """
    Melakukan peramalan harga untuk tiap komoditas ke tahun target menggunakan Linear Regression.
    Menghitung evaluasi in-sample (MAE, RMSE, R2, MAPE).
    """
    results = []
    commodities = sorted(df["komoditas"].unique())
    
    for comm in commodities:
        sub = df[df["komoditas"] == comm].sort_values("periode_data")
        X = sub[["periode_data"]].values
        y = sub["harga_rata_rata"].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred) if len(y) > 2 else 1.0
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        
        pred_future = model.predict([[target_year]])[0]
        curr_2025 = sub[sub["periode_data"] == 2025]["harga_rata_rata"].values[0]
        delta_val = pred_future - curr_2025
        delta_pct = (delta_val / curr_2025) * 100
        
        results.append({
            "komoditas": comm,
            "harga_2023": sub[sub["periode_data"] == 2023]["harga_rata_rata"].values[0],
            "harga_2024": sub[sub["periode_data"] == 2024]["harga_rata_rata"].values[0],
            "harga_2025": curr_2025,
            f"prediksi_{target_year}": round(pred_future, 2),
            "delta_nominal": round(delta_val, 2),
            "delta_pct": round(delta_pct, 2),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "mape_pct": round(mape, 2)
        })
        
    forecast_df = pd.DataFrame(results)
    return forecast_df


if __name__ == "__main__":
    try:
        from data_loader import get_clean_data
    except ImportError:
        from src.data_loader import get_clean_data
    df = get_clean_data(save_csv=False)
    pivot = compute_pivot(df)
    
    print("\n" + "="*60)
    print("📊 PIVOT HARGA RATA-RATA KOMODITAS (2023-2025)")
    print("="*60)
    print(pivot)
    
    print("\n" + "="*60)
    print("📈 INFLASI YEAR-OVER-YEAR (YoY %)")
    print("="*60)
    yoy = compute_inflation_yoy(pivot)
    print(yoy)
    
    print("\n" + "="*60)
    print("⚡ COMPOUND ANNUAL GROWTH RATE (CAGR 2Y %)")
    print("="*60)
    cagr = compute_cagr(pivot)
    print(cagr)
    
    print("\n" + "="*60)
    print("🎯 HASIL K-MEANS CLUSTERING KOMODITAS")
    print("="*60)
    cl_res = perform_kmeans_clustering(pivot)
    print(f"Optimal K: {cl_res['best_k']} (Silhouette Score: {cl_res['silhouette_scores'][cl_res['best_k']]})")
    print(cl_res["clustered_df"][["mean_price", "cv_pct", "total_growth", "segment_name"]])
    
    print("\n" + "="*60)
    print("🔮 PROYEKSI HARGA PANGAN TAHUN 2026")
    print("="*60)
    forecast = perform_price_forecasting(df, target_year=2026)
    print(forecast[["komoditas", "harga_2025", "prediksi_2026", "delta_pct", "r2", "mape_pct"]])
