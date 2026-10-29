"""
Harga Pangan Indonesia — Streamlit Interactive Dashboard
=========================================================
Interactive analytics dashboard for Indonesian food commodity prices (2023-2025).
Menampilkan analisis harga rata-rata 11 komoditas pangan pokok Indonesia.

Author: Fauzi Noorsyabani
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Harga Pangan Indonesia — Food Price Intelligence",
    page_icon="🍚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data Loading ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and clean the food price data."""
    csv_path = os.path.join(os.path.dirname(__file__), "data", "harga_pangan_clean.csv")
    xls_path = os.path.join(os.path.dirname(__file__), "harga-rata-rata-bahan-pangan-komponen-data.xls")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Parse HTML-as-XLS directly
        from bs4 import BeautifulSoup
        with open(xls_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "lxml")
        rows = []
        for tr in soup.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) == 3:
                rows.append(cells)
        df = pd.DataFrame(rows, columns=["periode_data", "komoditas", "harga_rata_rata"])
        df["periode_data"] = df["periode_data"].astype(int)
        df["harga_rata_rata"] = df["harga_rata_rata"].str.replace(",", "").astype(float)
        df["komoditas"] = df["komoditas"].str.strip().str.upper()
        df["komoditas"] = df["komoditas"].replace({
            "CABAI RAWIT MERHA": "CABAI RAWIT MERAH",
        })

    return df


def compute_yoy(df):
    """Compute Year-over-Year inflation per commodity."""
    pivot = df.pivot_table(index="komoditas", columns="periode_data", values="harga_rata_rata")
    yoy = pd.DataFrame(index=pivot.index)
    for i in range(1, len(pivot.columns)):
        col_prev = pivot.columns[i - 1]
        col_curr = pivot.columns[i]
        yoy[f"{col_prev}→{col_curr} (%)"] = ((pivot[col_curr] - pivot[col_prev]) / pivot[col_prev] * 100).round(2)
    return yoy, pivot


def compute_cagr(pivot):
    """Compute 2-year CAGR per commodity."""
    first_year = pivot.columns.min()
    last_year = pivot.columns.max()
    n_years = last_year - first_year
    cagr = ((pivot[last_year] / pivot[first_year]) ** (1 / n_years) - 1) * 100
    return cagr.round(2).sort_values(ascending=False)


# ── Load Data ───────────────────────────────────────────────────────
df = load_data()
yoy_df, pivot_df = compute_yoy(df)
cagr_series = compute_cagr(pivot_df)

# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/sheaf-of-rice.png", width=64)
    st.title("🍚 Harga Pangan Indonesia")
    st.caption("Food Price Intelligence Platform")
    st.markdown("---")
    st.markdown("**Data Period**: 2023 — 2025")
    st.markdown("**Commodities**: 11 Staple Items")
    st.markdown("**Source**: BPS / Kemendag")
    st.markdown("---")

    selected_commodities = st.multiselect(
        "Filter Komoditas",
        options=sorted(df["komoditas"].unique()),
        default=sorted(df["komoditas"].unique()),
    )

    selected_years = st.multiselect(
        "Filter Tahun",
        options=sorted(df["periode_data"].unique()),
        default=sorted(df["periode_data"].unique()),
    )

    st.markdown("---")
    st.markdown("**Built by** [Fauzi Noorsyabani](https://fauzinoorsyabani.my.id)")
    st.caption("Full Stack AI Engineer")

# ── Filter Data ─────────────────────────────────────────────────────
filtered_df = df[
    (df["komoditas"].isin(selected_commodities)) &
    (df["periode_data"].isin(selected_years))
]

# ── Header ──────────────────────────────────────────────────────────
st.markdown("# 🇮🇩 Harga Pangan Indonesia")
st.markdown("### Platform Analitik Kecerdasan Harga Pangan — Food Price Intelligence Dashboard")
st.markdown("Analisis komprehensif harga rata-rata **11 komoditas pangan pokok** Indonesia periode **2023–2025**.")

# ── KPI Cards ───────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_price = filtered_df["harga_rata_rata"].mean()
    st.metric("Rata-rata Harga", f"Rp {avg_price:,.0f}")

with col2:
    max_commodity = filtered_df.loc[filtered_df["harga_rata_rata"].idxmax()]
    st.metric("Termahal", max_commodity["komoditas"].title(), f"Rp {max_commodity['harga_rata_rata']:,.0f}")

with col3:
    min_commodity = filtered_df.loc[filtered_df["harga_rata_rata"].idxmin()]
    st.metric("Termurah", min_commodity["komoditas"].title(), f"Rp {min_commodity['harga_rata_rata']:,.0f}")

with col4:
    avg_cagr = cagr_series.mean()
    st.metric("Avg CAGR (2Y)", f"{avg_cagr:.1f}%", delta=f"{'↑' if avg_cagr > 0 else '↓'} inflasi")

st.markdown("---")

# ── Tab Layout ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Perbandingan Harga", "📈 Tren & Inflasi", "🎯 Clustering", "🔮 Forecasting"
])

# ── Tab 1: Price Comparison ─────────────────────────────────────────
with tab1:
    st.subheader("Perbandingan Harga Komoditas per Tahun")

    fig1 = px.bar(
        filtered_df,
        x="komoditas",
        y="harga_rata_rata",
        color="periode_data",
        barmode="group",
        title="Harga Rata-rata Komoditas Pangan Indonesia (2023-2025)",
        labels={
            "harga_rata_rata": "Harga (IDR/kg)",
            "komoditas": "Komoditas",
            "periode_data": "Tahun",
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig1.update_layout(
        xaxis_tickangle=-45,
        height=500,
        font=dict(size=12),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Heatmap
    st.subheader("Heatmap Harga Komoditas")
    heatmap_data = filtered_df.pivot_table(
        index="komoditas", columns="periode_data", values="harga_rata_rata"
    )
    fig_heat = px.imshow(
        heatmap_data,
        text_auto=".0f",
        aspect="auto",
        color_continuous_scale="YlOrRd",
        title="Matriks Harga (Komoditas × Tahun)",
        labels=dict(color="Harga (IDR)"),
    )
    fig_heat.update_layout(height=450)
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Tab 2: Trends & Inflation ───────────────────────────────────────
with tab2:
    st.subheader("Tren Harga & Inflasi Year-over-Year")

    # Line chart
    fig2 = px.line(
        filtered_df,
        x="periode_data",
        y="harga_rata_rata",
        color="komoditas",
        markers=True,
        title="Tren Harga Komoditas Pangan (2023-2025)",
        labels={
            "harga_rata_rata": "Harga (IDR/kg)",
            "periode_data": "Tahun",
            "komoditas": "Komoditas",
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig2.update_layout(height=500, xaxis=dict(dtick=1))
    st.plotly_chart(fig2, use_container_width=True)

    # YoY Inflation table
    st.subheader("Inflasi Year-over-Year (%)")
    styled_yoy = yoy_df.style.background_gradient(cmap="RdYlGn_r", axis=None).format("{:.2f}%")
    st.dataframe(styled_yoy, use_container_width=True)

    # CAGR Bar
    st.subheader("CAGR 2 Tahun (2023→2025)")
    cagr_df = cagr_series.reset_index()
    cagr_df.columns = ["Komoditas", "CAGR (%)"]
    fig_cagr = px.bar(
        cagr_df,
        x="CAGR (%)",
        y="Komoditas",
        orientation="h",
        title="Compound Annual Growth Rate per Komoditas",
        color="CAGR (%)",
        color_continuous_scale="RdYlGn_r",
    )
    fig_cagr.update_layout(height=450, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_cagr, use_container_width=True)

# ── Tab 3: Clustering ───────────────────────────────────────────────
with tab3:
    st.subheader("Segmentasi Komoditas dengan K-Means Clustering")

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    # Feature engineering
    features = pivot_df.copy()
    features["volatility"] = features.std(axis=1)
    features["mean_price"] = features.mean(axis=1)
    features["growth"] = ((features[2025] - features[2023]) / features[2023] * 100)

    X = features[["mean_price", "volatility", "growth"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Silhouette analysis
    from sklearn.metrics import silhouette_score
    sil_scores = {}
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil_scores[k] = silhouette_score(X_scaled, labels)

    best_k = max(sil_scores, key=sil_scores.get)

    col_sil1, col_sil2 = st.columns([1, 2])
    with col_sil1:
        st.markdown(f"**Optimal K**: {best_k} clusters")
        sil_df = pd.DataFrame(list(sil_scores.items()), columns=["K", "Silhouette"])
        st.dataframe(sil_df.set_index("K"))

    with col_sil2:
        fig_sil = px.line(
            x=list(sil_scores.keys()),
            y=list(sil_scores.values()),
            markers=True,
            title="Silhouette Score vs Number of Clusters",
            labels={"x": "K", "y": "Silhouette Score"},
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    # Final clustering
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    features["cluster"] = km_final.fit_predict(X_scaled)
    cluster_labels = {i: f"Tier {i+1}" for i in range(best_k)}
    features["segment"] = features["cluster"].map(cluster_labels)

    fig_cluster = px.scatter(
        features.reset_index(),
        x="mean_price",
        y="growth",
        size="volatility",
        color="segment",
        hover_name="komoditas",
        title="Commodity Segmentation (Mean Price vs Growth Rate)",
        labels={
            "mean_price": "Harga Rata-rata (IDR)",
            "growth": "Pertumbuhan 2Y (%)",
            "volatility": "Volatilitas",
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_cluster.update_layout(height=500)
    st.plotly_chart(fig_cluster, use_container_width=True)

    # Cluster profiles
    st.subheader("Profil Cluster")
    profile = features.groupby("segment")[["mean_price", "volatility", "growth"]].mean().round(2)
    profile.columns = ["Avg Price (IDR)", "Volatility", "Growth (%)"]
    st.dataframe(profile, use_container_width=True)

# ── Tab 4: Forecasting ─────────────────────────────────────────────
with tab4:
    st.subheader("Proyeksi Harga 2026 — Linear Regression")

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    forecasts = []
    for commodity in sorted(df["komoditas"].unique()):
        sub = df[df["komoditas"] == commodity].sort_values("periode_data")
        X_train = sub["periode_data"].values.reshape(-1, 1)
        y_train = sub["harga_rata_rata"].values
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_train)
        mae = mean_absolute_error(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        r2 = r2_score(y_train, y_pred)

        forecast_2026 = model.predict([[2026]])[0]
        forecasts.append({
            "Komoditas": commodity,
            "Harga 2025 (IDR)": y_train[-1],
            "Prediksi 2026 (IDR)": round(forecast_2026, 0),
            "Δ (IDR)": round(forecast_2026 - y_train[-1], 0),
            "Δ (%)": round((forecast_2026 - y_train[-1]) / y_train[-1] * 100, 2),
            "MAE": round(mae, 0),
            "RMSE": round(rmse, 0),
            "R²": round(r2, 4),
        })

    forecast_df = pd.DataFrame(forecasts)
    st.dataframe(forecast_df.set_index("Komoditas"), use_container_width=True)

    # Forecast visualization
    fig_forecast = go.Figure()
    for _, row in forecast_df.iterrows():
        fig_forecast.add_trace(go.Bar(
            name=row["Komoditas"],
            x=[row["Komoditas"]],
            y=[row["Prediksi 2026 (IDR)"]],
            marker_color="rgba(55, 128, 191, 0.7)",
        ))
        fig_forecast.add_trace(go.Bar(
            name=f"{row['Komoditas']} (2025)",
            x=[row["Komoditas"]],
            y=[row["Harga 2025 (IDR)"]],
            marker_color="rgba(219, 64, 82, 0.5)",
        ))

    fig_forecast = px.bar(
        forecast_df,
        x="Komoditas",
        y=["Harga 2025 (IDR)", "Prediksi 2026 (IDR)"],
        barmode="group",
        title="Harga Aktual 2025 vs Prediksi 2026",
        color_discrete_sequence=["#3780BF", "#DB4052"],
    )
    fig_forecast.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Model performance summary
    st.subheader("Ringkasan Performa Model")
    avg_r2 = forecast_df["R²"].mean()
    avg_mae = forecast_df["MAE"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg R²", f"{avg_r2:.4f}")
    c2.metric("Avg MAE", f"Rp {avg_mae:,.0f}")
    c3.metric("Commodities Forecasted", f"{len(forecast_df)}")

# ── Footer ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.85em;'>
        <strong>Harga Pangan Indonesia</strong> — Food Price Intelligence Platform<br/>
        Built by <a href='https://fauzinoorsyabani.my.id' target='_blank'>Fauzi Noorsyabani</a> · 
        Data: BPS/Kemendag · October 2026
    </div>
    """,
    unsafe_allow_html=True,
)

