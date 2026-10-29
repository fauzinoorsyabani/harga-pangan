"""
generate_portfolio_assets.py
Generate high-fidelity web-optimized WebP assets for Fauzi Noorsyabani's Portfolio Website.
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
PORTFOLIO_PUBLIC = r"D:\Code\fauzi-noorsyabani-portfolio-web\client\public"
PORTFOLIO_DOC = os.path.join(PORTFOLIO_PUBLIC, "documentation")

os.makedirs(PORTFOLIO_DOC, exist_ok=True)
ASSETS_DIR = os.path.join(_PROJECT_ROOT, "portfolio_assets")

csv_path = os.path.join(_PROJECT_ROOT, "data", "harga_pangan_clean.csv")
df = pd.read_csv(csv_path)
pivot = df.pivot_table(index="komoditas", columns="periode_data", values="harga_rata_rata")

# -------------------------------------------------------------
# 1. Main Project Card Hero Asset: food-price-intelligence.webp (1200x750)
# -------------------------------------------------------------
fig = plt.figure(figsize=(15, 9.5), dpi=150, facecolor="#0B0F19")
gs = gridspec.GridSpec(2, 2, height_ratios=[1.1, 1], hspace=0.35, wspace=0.25, left=0.08, right=0.95, top=0.88, bottom=0.08)

# Title banner
fig.text(0.08, 0.94, "HARGA PANGAN INDONESIA · FOOD PRICE INTELLIGENCE", color="#38BDF8", fontsize=15, fontweight="bold", family="sans-serif")
fig.text(0.08, 0.905, "Komparasi Harga 11 Komoditas Pokok & Proyeksi Inflasi Nasional (2023–2025)", color="#94A3B8", fontsize=11, family="sans-serif")
fig.text(0.95, 0.94, "BPS / KEMENDAG BENCHMARK", color="#10B981", fontsize=11, fontweight="bold", ha="right", family="sans-serif")

# Subplot 1: Price by Commodity & Year
ax1 = fig.add_subplot(gs[0, 0], facecolor="#111827")
sorted_p = pivot.sort_values(by=2025, ascending=True)
y_pos = np.arange(len(sorted_p))
h = 0.28
ax1.barh(y_pos - h, sorted_p[2023], h, label="2023", color="#38BDF8", alpha=0.9)
ax1.barh(y_pos, sorted_p[2024], h, label="2024", color="#F59E0B", alpha=0.9)
ax1.barh(y_pos + h, sorted_p[2025], h, label="2025", color="#EF4444", alpha=0.9)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(sorted_p.index, color="#E2E8F0", fontsize=8, fontweight="bold")
ax1.tick_params(axis="x", colors="#94A3B8", labelsize=8)
ax1.set_title("Distribusi Harga Komoditas (IDR/kg)", color="#F1F5F9", fontsize=10, fontweight="bold", pad=8)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"Rp {int(v/1000)}k"))
ax1.legend(loc="lower right", facecolor="#1F2937", edgecolor="#374151", labelcolor="#F1F5F9", fontsize=8)
ax1.grid(color="#1F2937", linestyle="--", alpha=0.6)

# Subplot 2: Longitudinal trajectories
ax2 = fig.add_subplot(gs[0, 1], facecolor="#111827")
colors = ["#38BDF8", "#F59E0B", "#10B981", "#EF4444", "#A855F7", "#EC4899", "#6366F1", "#14B8A6", "#F97316", "#84CC16", "#EAB308"]
for i, comm in enumerate(sorted(df["komoditas"].unique())):
    sub = df[df["komoditas"] == comm].sort_values("periode_data")
    ax2.plot(sub["periode_data"], sub["harga_rata_rata"], marker="o", linewidth=1.8, markersize=5, color=colors[i % len(colors)], label=comm)
ax2.set_xticks([2023, 2024, 2025])
ax2.tick_params(colors="#94A3B8", labelsize=8)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"Rp {int(v/1000)}k"))
ax2.set_title("Lintasan Tren Multi-Tahun", color="#F1F5F9", fontsize=10, fontweight="bold", pad=8)
ax2.grid(color="#1F2937", linestyle="--", alpha=0.6)
ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", facecolor="#1F2937", edgecolor="#374151", labelcolor="#E2E8F0", fontsize=6.5)

# Subplot 3: 2-Year CAGR
ax3 = fig.add_subplot(gs[1, 0], facecolor="#111827")
cagr = (((pivot[2025] / pivot[2023]) ** 0.5) - 1) * 100
cagr_sorted = cagr.sort_values(ascending=True)
c_colors = ["#EF4444" if v > 10 else "#F59E0B" if v > 5 else "#10B981" if v >= 0 else "#38BDF8" for v in cagr_sorted.values]
bars = ax3.barh(cagr_sorted.index, cagr_sorted.values, color=c_colors, height=0.6)
ax3.axvline(0, color="#94A3B8", linewidth=0.8)
ax3.set_yticks(np.arange(len(cagr_sorted)))
ax3.set_yticklabels(cagr_sorted.index, color="#E2E8F0", fontsize=8, fontweight="bold")
ax3.tick_params(axis="x", colors="#94A3B8", labelsize=8)
ax3.set_title("Laju Pertumbuhan Majemuk (CAGR % 2023→2025)", color="#F1F5F9", fontsize=10, fontweight="bold", pad=8)
for bar in bars:
    val = bar.get_width()
    ax3.text(val + (0.3 if val >= 0 else -1.8), bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va="center", color="#F8FAFC", fontsize=7.5, fontweight="bold")
ax3.grid(color="#1F2937", linestyle="--", alpha=0.6)

# Subplot 4: K-Means Segmentation & Forecast Highlight
ax4 = fig.add_subplot(gs[1, 1], facecolor="#111827")
yoy_last = ((pivot[2025] - pivot[2024]) / pivot[2024]) * 100
scatter_colors = ["#EF4444" if "CABAI" in c or "BAWANG" in c else "#10B981" if "SAPI" in c or "AYAM" in c or "TELUR" in c else "#38BDF8" for c in pivot.index]
scatter = ax4.scatter(pivot.mean(axis=1), yoy_last, s=180, c=scatter_colors, edgecolors="#F8FAFC", linewidth=1.2, alpha=0.85)
for c in pivot.index:
    ax4.annotate(c, (pivot.mean(axis=1)[c], yoy_last[c]), xytext=(5, 5), textcoords="offset points", color="#E2E8F0", fontsize=7, fontweight="bold")
ax4.set_xlabel("Rata-Rata Harga (IDR)", color="#94A3B8", fontsize=8, fontweight="bold")
ax4.set_ylabel("Inflasi YoY 2025 (%)", color="#94A3B8", fontsize=8, fontweight="bold")
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"Rp {int(v/1000)}k"))
ax4.tick_params(colors="#94A3B8", labelsize=8)
ax4.set_title("Segmentasi Risiko & Volatilitas Komoditas", color="#F1F5F9", fontsize=10, fontweight="bold", pad=8)
ax4.grid(color="#1F2937", linestyle="--", alpha=0.6)

temp_hero_png = os.path.join(ASSETS_DIR, "hero_food_price_intelligence_raw.png")
fig.savefig(temp_hero_png, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)

# Convert to WebP
img = Image.open(temp_hero_png)
webp_hero_path = os.path.join(PORTFOLIO_PUBLIC, "food-price-intelligence.webp")
img.save(webp_hero_path, "WEBP", quality=90)
img.save(os.path.join(ASSETS_DIR, "food-price-intelligence.webp"), "WEBP", quality=90)
print(f"✅ Generated hero card: {webp_hero_path}")

# -------------------------------------------------------------
# 2. Documentation Carousel Asset: food-price-analytics.webp (800x550)
# -------------------------------------------------------------
# Use chart 01 or 05 converted to WebP
source_chart = os.path.join(ASSETS_DIR, "05_kmeans_commodity_clusters.png")
if os.path.exists(source_chart):
    doc_img = Image.open(source_chart)
    doc_webp_path = os.path.join(PORTFOLIO_DOC, "food-price-analytics.webp")
    doc_img.save(doc_webp_path, "WEBP", quality=88)
    doc_img.save(os.path.join(ASSETS_DIR, "food-price-analytics.webp"), "WEBP", quality=88)
    print(f"✅ Generated documentation card: {doc_webp_path}")

print("✨ All portfolio web assets generated successfully!")
