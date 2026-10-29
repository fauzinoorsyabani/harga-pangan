"""
visualizer.py — Generator Visualisasi Tingkat Publikasi (300 DPI)
Indonesian Food Commodity Price Intelligence Platform

Menghasilkan 8+ visualisasi grafis publication-grade untuk laporan, notebook,
dan aset portofolio web Fauzi Noorsyabani.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_loader import get_clean_data
from analyzer import (
    compute_pivot,
    compute_inflation_yoy,
    compute_cagr,
    compute_volatility,
    perform_kmeans_clustering,
    perform_price_forecasting
)

# Output directory for publication assets
ASSETS_DIR = os.path.join(_PROJECT_ROOT, "portfolio_assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# Styling setup
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#EAEAEA"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7

COLOR_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def save_chart(fig, filename: str):
    """Menyimpan gambar dengan resolusi publikasi 300 DPI."""
    filepath = os.path.join(ASSETS_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f" Saved: {filename}")
    return filepath


def plot_01_commodity_prices_overview(df: pd.DataFrame):
    """01: Grouped bar chart harga seluruh komoditas tahun 2023-2025."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    pivot = df.pivot_table(index="komoditas", columns="periode_data", values="harga_rata_rata")
    pivot = pivot.sort_values(by=2025, ascending=True)
    
    x = np.arange(len(pivot.index))
    width = 0.27
    
    rects1 = ax.barh(x - width, pivot[2023], width, label="2023", color="#4575b4", alpha=0.9)
    rects2 = ax.barh(x, pivot[2024], width, label="2024", color="#fdae61", alpha=0.9)
    rects3 = ax.barh(x + width, pivot[2025], width, label="2025", color="#d73027", alpha=0.9)
    
    ax.set_yticks(x)
    ax.set_yticklabels(pivot.index, fontweight="bold", fontsize=10)
    ax.set_xlabel("Harga Rata-Rata Nasional (IDR/kg atau liter)", fontsize=11, fontweight="bold")
    ax.set_title("Perbandingan Harga 11 Komoditas Pangan Pokok Indonesia (2023–2025)\nNational Average Food Commodity Prices Benchmark", fontsize=14, fontweight="bold", pad=15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"Rp {int(val):,}"))
    ax.legend(title="Tahun Periode", loc="lower right", frameon=True, facecolor="white", edgecolor="#ccc")
    
    # Annotate top commodity
    max_val = pivot[2025].max()
    ax.set_xlim(0, max_val * 1.15)
    
    return save_chart(fig, "01_commodity_prices_overview.png")


def plot_02_yoy_inflation_comparison(pivot: pd.DataFrame):
    """02: Bar chart perbandingan laju inflasi YoY (2023-2024 vs 2024-2025)."""
    yoy = compute_inflation_yoy(pivot)
    yoy_sorted = yoy.sort_values(by="YoY_2024_2025_pct", ascending=True)
    
    fig, ax = plt.subplots(figsize=(13, 7))
    y = np.arange(len(yoy_sorted))
    height = 0.35
    
    ax.barh(y - height/2, yoy_sorted["YoY_2023_2024_pct"], height, label="YoY 2023→2024 (%)", color="#3182bd", alpha=0.85)
    ax.barh(y + height/2, yoy_sorted["YoY_2024_2025_pct"], height, label="YoY 2024→2025 (%)", color="#e6550d", alpha=0.85)
    
    ax.axvline(0, color="black", linestyle="-", linewidth=0.9, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(yoy_sorted.index, fontweight="bold", fontsize=10)
    ax.set_xlabel("Laju Inflasi Komoditas (%)", fontsize=11, fontweight="bold")
    ax.set_title("Dinamika Inflasi Tahunan Komoditas Pangan Pokok (YoY %)\nAnnual Commodity Inflation Dynamics: 2023-2024 vs 2024-2025", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#ccc")
    
    # Value labels
    for idx, (idx_name, row) in enumerate(yoy_sorted.iterrows()):
        ax.text(row["YoY_2024_2025_pct"] + (0.5 if row["YoY_2024_2025_pct"] >= 0 else -1.5), idx + height/2, f"{row['YoY_2024_2025_pct']:.1f}%", va="center", fontsize=8, fontweight="bold", color="#a63603")
        
    return save_chart(fig, "02_yoy_inflation_comparison.png")


def plot_03_price_matrix_heatmap(pivot: pd.DataFrame):
    """03: Heatmap matriks harga komoditas pangan."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort by 2025 price
    sorted_pivot = pivot.sort_values(by=2025, ascending=False)
    
    # Normalize for color intensity (log scale or relative)
    sns.heatmap(
        sorted_pivot,
        annot=True,
        fmt=",.0f",
        cmap="YlOrRd",
        linewidths=1,
        linecolor="#fff",
        cbar_kws={"label": "Harga Rata-Rata (IDR)"},
        ax=ax,
        annot_kws={"fontweight": "bold", "fontsize": 10}
    )
    
    ax.set_title("Matriks Harga Rata-Rata Komoditas Pangan Indonesia (2023–2025)\nFood Commodity Price Heatmap Matrix", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Tahun Periode Data", fontsize=11, fontweight="bold")
    ax.set_ylabel("Komoditas", fontsize=11, fontweight="bold")
    
    return save_chart(fig, "03_price_matrix_heatmap.png")


def plot_04_longitudinal_trends(df: pd.DataFrame):
    """04: Line chart tren longitudinal harga pangan."""
    fig, ax = plt.subplots(figsize=(13, 7))
    
    commodities = sorted(df["komoditas"].unique())
    palette = sns.color_palette("tab20", len(commodities))
    
    for i, comm in enumerate(commodities):
        sub = df[df["komoditas"] == comm].sort_values("periode_data")
        ax.plot(
            sub["periode_data"],
            sub["harga_rata_rata"],
            marker="o",
            linewidth=2.2,
            label=comm,
            color=palette[i],
            alpha=0.9
        )
        
    ax.set_xticks([2023, 2024, 2025])
    ax.set_xlabel("Tahun Pengamatan", fontsize=11, fontweight="bold")
    ax.set_ylabel("Harga Rata-Rata (IDR)", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"Rp {int(val):,}"))
    ax.set_title("Lintasan Tren Longitudinal Harga 11 Komoditas Pangan (2023–2025)\nLongitudinal Price Trajectories Across Staple Commodities", fontsize=14, fontweight="bold", pad=15)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, facecolor="white", edgecolor="#ccc", fontsize=9)
    
    return save_chart(fig, "04_longitudinal_trends.png")


def plot_05_kmeans_commodity_clusters(pivot: pd.DataFrame):
    """05: Scatter plot 2D PCA & K-Means clustering segmen harga pangan."""
    cl_res = perform_kmeans_clustering(pivot)
    clustered_df = cl_res["clustered_df"]
    X_scaled = cl_res["scaled_data"]
    
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    clustered_df["pca_1"] = coords[:, 0]
    clustered_df["pca_2"] = coords[:, 1]
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    colors = {"Tier Terjangkau": "#2b83ba", "Tier Menengah": "#fdae61", "Tier Premium": "#d7191c"}
    
    for segment, group in clustered_df.groupby("segment_name"):
        color = colors.get(segment, "#7570b3")
        ax.scatter(
            group["pca_1"],
            group["pca_2"],
            s=group["mean_price"] / 150,
            label=f"{segment} (n={len(group)})",
            color=color,
            alpha=0.8,
            edgecolors="black",
            linewidth=1.2
        )
        for _, row in group.iterrows():
            ax.annotate(
                row.name,
                (row["pca_1"], row["pca_2"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold"
            )
            
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Varians)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Varians)", fontsize=10, fontweight="bold")
    ax.set_title(f"Segmentasi Komoditas Pangan via K-Means Clustering (K={cl_res['best_k']})\nCommodity Market Segmentation: Silhouette Score = {cl_res['silhouette_scores'][cl_res['best_k']]:.4f}", fontsize=13, fontweight="bold", pad=15)
    ax.legend(title="Segmen Pasar (Ukuran = Rata-Rata Harga)", loc="upper left", frameon=True, facecolor="white")
    
    return save_chart(fig, "05_kmeans_commodity_clusters.png")


def plot_06_cagr_growth_ranking(pivot: pd.DataFrame):
    """06: Horizontal bar chart peringkat CAGR 2-tahun (2023-2025)."""
    cagr = compute_cagr(pivot).sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 6.5))
    
    colors = ["#d73027" if val > 10 else "#fee08b" if val > 5 else "#1a9850" if val >= 0 else "#4575b4" for val in cagr.values]
    
    bars = ax.barh(cagr.index, cagr.values, color=colors, height=0.6, edgecolor="#333", linewidth=0.8)
    
    ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Compound Annual Growth Rate 2-Tahun (CAGR %)", fontsize=11, fontweight="bold")
    ax.set_title("Peringkat Laju Pertumbuhan Tahunan Majemuk (CAGR 2023–2025)\nTwo-Year Compound Annual Growth Rate per Commodity", fontsize=13, fontweight="bold", pad=15)
    
    for bar in bars:
        val = bar.get_width()
        offset = 0.3 if val >= 0 else -1.2
        ax.text(val + offset, bar.get_y() + bar.get_height()/2, f"{val:.2f}%", va="center", fontsize=9, fontweight="bold")
        
    ax.set_xlim(min(cagr.values) - 2, max(cagr.values) + 2)
    return save_chart(fig, "06_cagr_growth_ranking.png")


def plot_07_forecasting_2026(df: pd.DataFrame):
    """07: Perbandingan harga aktual 2025 vs proyeksi 2026."""
    forecast = perform_price_forecasting(df, target_year=2026)
    forecast = forecast.sort_values(by="harga_2025", ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 7.5))
    
    y = np.arange(len(forecast))
    height = 0.38
    
    ax.barh(y - height/2, forecast["harga_2025"], height, label="Aktual 2025", color="#2b83ba", alpha=0.9)
    ax.barh(y + height/2, forecast["prediksi_2026"], height, label="Proyeksi 2026 (Linear Regression)", color="#d7191c", alpha=0.9)
    
    ax.set_yticks(y)
    ax.set_yticklabels(forecast["komoditas"], fontweight="bold", fontsize=10)
    ax.set_xlabel("Harga Pangan (IDR)", fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"Rp {int(val):,}"))
    ax.set_title("Proyeksi Harga Pangan Tahun 2026 Menggunakan Model Linear Regression\nFood Commodity Price Projections for 2026 (Avg R² = 0.774)", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#ccc")
    
    for idx, (_, row) in enumerate(forecast.iterrows()):
        d_pct = row["delta_pct"]
        sign = "+" if d_pct > 0 else ""
        ax.text(row["prediksi_2026"] + 1500, idx + height/2, f"{sign}{d_pct:.1f}%", va="center", fontsize=8, fontweight="bold", color="#990000")
        
    return save_chart(fig, "07_forecasting_2026_projections.png")


def plot_08_volatility_coefficient_variation(pivot: pd.DataFrame):
    """08: Koefisien Variasi (CV %) sebagai indikator stabilitas harga."""
    vol = compute_volatility(pivot).sort_values(by="cv_pct", ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 6.5))
    
    # Colors: green for stable (CV < 5%), yellow for medium, red for volatile (CV > 10%)
    colors = ["#1a9850" if val < 5 else "#fdae61" if val <= 10 else "#d73027" for val in vol["cv_pct"]]
    
    bars = ax.barh(vol.index, vol["cv_pct"], color=colors, height=0.6, edgecolor="#333", linewidth=0.8)
    
    ax.axvline(5, color="#1a9850", linestyle="--", linewidth=1, label="Batas Stabil (<5%)")
    ax.axvline(10, color="#d73027", linestyle="--", linewidth=1, label="Batas Volatil (>10%)")
    
    ax.set_xlabel("Koefisien Variasi Harga / Coefficient of Variation (%)", fontsize=11, fontweight="bold")
    ax.set_title("Tingkat Volatilitas & Stabilitas Harga Komoditas Pangan (2023–2025)\nPrice Volatility Classification: Stable vs Volatile Commodities", fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, facecolor="white")
    
    for bar in bars:
        val = bar.get_width()
        ax.text(val + 0.2, bar.get_y() + bar.get_height()/2, f"{val:.2f}%", va="center", fontsize=9, fontweight="bold")
        
    ax.set_xlim(0, max(vol["cv_pct"]) + 2.5)
    return save_chart(fig, "08_volatility_coefficient_variation.png")


def generate_all_visualizations():
    """Menjalankan seluruh pipeline pembuatan visualisasi."""
    print("=" * 60)
    print("🎨 MEMBUAT VISUALISASI PUBLIKASI (300 DPI)")
    print("=" * 60)
    
    df = get_clean_data(save_csv=True)
    pivot = compute_pivot(df)
    
    p1 = plot_01_commodity_prices_overview(df)
    p2 = plot_02_yoy_inflation_comparison(pivot)
    p3 = plot_03_price_matrix_heatmap(pivot)
    p4 = plot_04_longitudinal_trends(df)
    p5 = plot_05_kmeans_commodity_clusters(pivot)
    p6 = plot_06_cagr_growth_ranking(pivot)
    p7 = plot_07_forecasting_2026(df)
    p8 = plot_08_volatility_coefficient_variation(pivot)
    
    print("=" * 60)
    print(f"✅ Seluruh 8 artefak grafis 300 DPI berhasil disimpan di: {ASSETS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_visualizations()
