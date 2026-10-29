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
import os
import requests
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# -- Helper Functions --------------------------------------------------
def sanitize_plain_text(text: str) -> str:
    """Menghilangkan karakter asterisk dan format bold agar tidak memunculkan icon bintang."""
    if not text:
        return ""
    return text.replace("**", "").replace("*", "").strip()


# -- Page Configuration ------------------------------------------------
st.set_page_config(
    page_title="Harga Pangan Indonesia — Food Price Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Data Loading ------------------------------------------------------
@st.cache_data
def load_data():
    """Load and clean the food price data."""
    csv_path = os.path.join(os.path.dirname(__file__), "data", "harga_pangan_clean.csv")
    xls_path = os.path.join(os.path.dirname(__file__), "harga-rata-rata-bahan-pangan-komponen-data.xls")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
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
        df["harga_rata_rata"] = df["harga_rata_rata"].astype(str).str.replace(",", "").astype(float)
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
        yoy[f"{col_prev}-{col_curr} (%)"] = ((pivot[col_curr] - pivot[col_prev]) / pivot[col_prev] * 100).round(2)
    return yoy, pivot


def compute_cagr(pivot):
    """Compute 2-year CAGR per commodity."""
    first_year = pivot.columns.min()
    last_year = pivot.columns.max()
    n_years = last_year - first_year
    cagr = ((pivot[last_year] / pivot[first_year]).pow(1 / n_years) - 1) * 100
    return cagr.round(2).sort_values(ascending=False)


# -- Load Data ---------------------------------------------------------
df = load_data()
yoy_df, pivot_df = compute_yoy(df)
cagr_series = compute_cagr(pivot_df)

# Compute forecasting metrics for all commodities
all_commodities = sorted(df["komoditas"].unique())
forecast_summary = {}
for c in all_commodities:
    sub = df[df["komoditas"] == c].sort_values("periode_data")
    X_t = sub["periode_data"].values.reshape(-1, 1)
    y_t = sub["harga_rata_rata"].values
    lr = LinearRegression()
    lr.fit(X_t, y_t)
    f26 = lr.predict([[2026]])[0]
    p25 = y_t[-1]
    pct = ((f26 - p25) / p25) * 100
    forecast_summary[c] = {
        "price_2023": sub[sub["periode_data"] == 2023]["harga_rata_rata"].values[0] if 2023 in sub["periode_data"].values else 0,
        "price_2024": sub[sub["periode_data"] == 2024]["harga_rata_rata"].values[0] if 2024 in sub["periode_data"].values else 0,
        "price_2025": p25,
        "forecast_2026": f26,
        "forecast_pct": pct,
        "cagr": cagr_series.get(c, 0.0),
        "yoy_24_25": yoy_df.loc[c, "2024-2025 (%)"] if "2024-2025 (%)" in yoy_df.columns else 0.0,
    }

# -- Sidebar -----------------------------------------------------------
with st.sidebar:
    st.title("Harga Pangan Indonesia")
    st.caption("Food Price Intelligence Platform")
    st.markdown("---")
    st.markdown("Periode Data: 2023 - 2025")
    st.markdown("Komoditas: 11 Bahan Pangan Pokok")
    st.markdown("Sumber Data: BPS / Kementerian Perdagangan RI")
    st.markdown("---")

    selected_commodities = st.multiselect(
        "Pilih Komoditas",
        options=all_commodities,
        default=all_commodities,
    )

    selected_years = st.multiselect(
        "Pilih Tahun",
        options=sorted(df["periode_data"].unique()),
        default=sorted(df["periode_data"].unique()),
    )

    # Look for API key from st.secrets or os.getenv
    default_key = ""
    try:
        default_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass
    if not default_key:
        default_key = os.getenv("GEMINI_API_KEY", "")

    st.markdown("---")
    st.markdown("### Integrasi AI Studio (Gemini)")
    gemini_key_input = st.text_input(
        "Google Gemini API Key",
        type="password",
        value=default_key,
        help="Dapatkan API Key gratis di Google AI Studio (aistudio.google.com) untuk mengaktifkan bot tanya jawab pada setiap bagian.",
    )
    if gemini_key_input:
        st.caption("Status API: Terhubung ke Google AI Studio")

    st.markdown("---")
    st.markdown("Pengembang: [Fauzi Noorsyabani](https://fauzinoorsyabani.my.id)")
    st.caption("Full Stack AI & Data Science Engineer")

# -- Filter Data -------------------------------------------------------
filtered_df = df[
    (df["komoditas"].isin(selected_commodities)) &
    (df["periode_data"].isin(selected_years))
]

# -- Header ------------------------------------------------------------
st.markdown("# Harga Pangan Indonesia")
st.markdown("### Platform Analitik Kecerdasan Harga Pangan Pokok Nasional")
st.markdown("Analisis komprehensif harga rata-rata 11 komoditas pangan pokok Indonesia periode 2023 - 2025.")

# -- KPI Cards ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_price = filtered_df["harga_rata_rata"].mean() if not filtered_df.empty else 0
    st.metric("Rata-rata Harga", f"Rp {avg_price:,.0f}")

with col2:
    if not filtered_df.empty:
        max_commodity = filtered_df.loc[filtered_df["harga_rata_rata"].idxmax()]
        st.metric("Komoditas Termahal", max_commodity["komoditas"].title(), f"Rp {max_commodity['harga_rata_rata']:,.0f}")
    else:
        st.metric("Komoditas Termahal", "-")

with col3:
    if not filtered_df.empty:
        min_commodity = filtered_df.loc[filtered_df["harga_rata_rata"].idxmin()]
        st.metric("Komoditas Termurah", min_commodity["komoditas"].title(), f"Rp {min_commodity['harga_rata_rata']:,.0f}")
    else:
        st.metric("Komoditas Termurah", "-")

with col4:
    avg_cagr = cagr_series.mean()
    st.metric("Rata-rata CAGR (2-Tahun)", f"{avg_cagr:+.2f}%")

st.markdown("---")

# -- Gemini Interactive Q&A Function -----------------------------------
def ask_gemini_bot(api_key: str, section_title: str, context_text: str, user_question: str) -> str:
    """Interactive AI Q&A bot for each dashboard section with strict plain text rules."""
    headers = {"Content-Type": "application/json"}
    prompt = f"""Anda adalah Analis Senior Intelijen Pangan Nasional untuk Badan Pangan Nasional (Bapanas) dan Kementerian Perdagangan RI.
Anda sedang berdialog langsung dengan pengguna pada bagian dashboard: {section_title}.

KONTEKS DATA PADA BAGIAN INI:
{context_text}

PERTANYAAN PENGGUNA:
{user_question}

PERATURAN FORMAT JAWABAN SANGAT KETAT:
1. DILARANG MENGGUNAKAN SIMBOL BINTANG (*) MAUPUN FORMAT BOLD (**) SAMA SEKALI. JANGAN PERNAH MENULISKAN TANDA ASTERISK.
2. DILARANG MENGGUNAKAN SIMBOL EMOJI APAPUN.
3. Gunakan teks polos yang rapi dan profesional. Untuk penekanan kata atau judul, gunakan HURUF KAPITAL (UPPERCASE).
4. Gunakan angka (1., 2., 3.) atau tanda strip (-) untuk membuat poin-poin penjelasan.
5. Jawab pertanyaan pengguna secara ringkas, lugas, berbasis data di atas, dan berikan rekomendasi actionable.
"""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1000,
        },
    }

    candidate_models = ["gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-3.8-flash"]
    for model_name in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                return sanitize_plain_text(raw_text)
            elif resp.status_code == 404:
                continue
        except Exception:
            continue

    return "Layanan Analis AI sedang sibuk atau tidak dapat dijangkau. Silakan periksa kembali API Key Anda."


def render_interactive_bot(section_id: str, section_title: str, context_str: str, api_key: str, sample_questions: list):
    """Menampilkan kotak bot tanya jawab interaktif untuk setiap bagian dashboard."""
    st.markdown("---")
    st.subheader(f"Tanya Analis AI: {section_title}")
    st.caption(f"Konsultasikan analisis mendalam dan ajukan pertanyaan mengenai data pada bagian {section_title}.")

    history_key = f"chat_history_{section_id}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    st.markdown("Pilihan Pertanyaan Cepat:")
    cols = st.columns(len(sample_questions))
    selected_quick_q = None
    for idx, q_text in enumerate(sample_questions):
        if cols[idx].button(q_text, key=f"quick_{section_id}_{idx}", width="stretch"):
            selected_quick_q = q_text

    with st.form(key=f"form_{section_id}", clear_on_submit=True):
        custom_question = st.text_input(
            "Ajukan pertanyaan Anda mengenai data di atas:",
            value=selected_quick_q if selected_quick_q else "",
            placeholder="Contoh: Mengapa harga komoditas ini naik drastis?",
            key=f"input_{section_id}",
        )
        submit_btn = st.form_submit_button("Kirim Pertanyaan ke Analis AI")

    active_question = selected_quick_q if selected_quick_q else (custom_question if submit_btn else None)

    if active_question and active_question.strip():
        if not api_key:
            st.warning("Silakan masukkan Google Gemini API Key di sidebar untuk mengaktifkan dialog dengan Analis AI.")
        else:
            with st.spinner("Analis AI sedang memproses data dan menyusun jawaban..."):
                answer = ask_gemini_bot(api_key, section_title, context_str, active_question)
                st.session_state[history_key].append({"q": active_question, "a": answer})

    if st.session_state[history_key]:
        for item in reversed(st.session_state[history_key]):
            st.markdown(f"PERTANYAAN: {item['q']}")
            st.info(item["a"])
            st.markdown("")
        if st.button("Hapus Riwayat Tanya Jawab", key=f"clear_{section_id}"):
            st.session_state[history_key] = []
            st.rerun()


# -- Tab Layout --------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Perbandingan Harga",
    "Tren dan Inflasi",
    "Clustering dan Segmentasi",
    "Forecasting 2026",
    "Rekomendasi Kebijakan AI",
])

# -- Tab 1: Price Comparison -------------------------------------------
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
    st.plotly_chart(fig1, width="stretch")

    st.subheader("Matriks Harga (Komoditas vs Tahun)")
    if not filtered_df.empty:
        heatmap_data = filtered_df.pivot_table(
            index="komoditas", columns="periode_data", values="harga_rata_rata"
        )
        fig_heat = px.imshow(
            heatmap_data,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale="YlOrRd",
            title="Matriks Harga Pangan Rata-rata (IDR)",
            labels=dict(color="Harga (IDR)"),
        )
        fig_heat.update_layout(height=450)
        st.plotly_chart(fig_heat, width="stretch")

    # Context and Bot for Tab 1
    context_tab1 = f"""Tabel Harga Pivot Komoditas x Tahun:
{pivot_df.to_string()}
Rata-rata Harga Keseluruhan: Rp {avg_price:,.0f}
Komoditas Termahal: {max_commodity['komoditas']} (Rp {max_commodity['harga_rata_rata']:,.0f})
Komoditas Termurah: {min_commodity['komoditas']} (Rp {min_commodity['harga_rata_rata']:,.0f})
"""
    sample_q_tab1 = [
        "Komoditas apa yang memiliki disparitas harga paling mencolok?",
        "Bagaimana perbandingan harga Beras IR I dengan Beras IR III?",
        "Komoditas apa yang harganya paling stabil selama 3 tahun?",
    ]
    render_interactive_bot("tab1", "Perbandingan Harga", context_tab1, gemini_key_input, sample_q_tab1)

# -- Tab 2: Trends & Inflation -----------------------------------------
with tab2:
    st.subheader("Tren Harga dan Laju Inflasi Year-over-Year")

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
    st.plotly_chart(fig2, width="stretch")

    st.subheader("Inflasi Year-over-Year (%)")
    styled_yoy = yoy_df.style.background_gradient(cmap="RdYlGn_r", axis=None).format("{:.2f}%")
    st.dataframe(styled_yoy, width="stretch")

    st.subheader("Compound Annual Growth Rate (CAGR 2-Tahun: 2023-2025)")
    cagr_df = cagr_series.reset_index()
    cagr_df.columns = ["Komoditas", "CAGR (%)"]
    fig_cagr = px.bar(
        cagr_df,
        x="CAGR (%)",
        y="Komoditas",
        orientation="h",
        title="Laju Pertumbuhan Tahunan Majemuk per Komoditas",
        color="CAGR (%)",
        color_continuous_scale="RdYlGn_r",
    )
    fig_cagr.update_layout(height=450, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_cagr, width="stretch")

    # Context and Bot for Tab 2
    context_tab2 = f"""Data Pertumbuhan dan Inflasi Pangan:
Compound Annual Growth Rate (CAGR 2-Tahun):
{cagr_series.to_string()}
Laju Inflasi YoY:
{yoy_df.to_string()}
Catatan Kunci: Bawang Merah mengalami CAGR tertinggi (+12.43%), disusul Minyak Goreng (+11.61%) dan Gula Pasir (+11.01%). Daging Sapi mengalami sedikit deflasi (-0.69%).
"""
    sample_q_tab2 = [
        "Mengapa inflasi Bawang Merah dan Minyak Goreng mencapai dua digit?",
        "Mengapa harga Daging Sapi cenderung stagnan atau sedikit deflasi?",
        "Apa implikasi kenaikan harga minyak goreng terhadap biaya hidup rumah tangga?",
    ]
    render_interactive_bot("tab2", "Tren dan Inflasi", context_tab2, gemini_key_input, sample_q_tab2)

# -- Tab 3: Clustering -------------------------------------------------
with tab3:
    st.subheader("Segmentasi Pasar Komoditas dengan K-Means Clustering")

    features = pivot_df.copy()
    features["volatility"] = features.std(axis=1)
    features["mean_price"] = features.mean(axis=1)
    features["growth"] = ((features[2025] - features[2023]) / features[2023] * 100)

    X = features[["mean_price", "volatility", "growth"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sil_scores = {}
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil_scores[k] = silhouette_score(X_scaled, labels)

    best_k = max(sil_scores, key=sil_scores.get)

    col_sil1, col_sil2 = st.columns([1, 2])
    with col_sil1:
        st.markdown(f"Klaster Optimal Berdasarkan Evaluasi Silhouette: K = {best_k}")
        sil_df = pd.DataFrame(list(sil_scores.items()), columns=["K (Jumlah Klaster)", "Skor Silhouette"])
        st.dataframe(sil_df.set_index("K (Jumlah Klaster)"))

    with col_sil2:
        fig_sil = px.line(
            x=list(sil_scores.keys()),
            y=list(sil_scores.values()),
            markers=True,
            title="Analisis Skor Silhouette vs Jumlah Klaster",
            labels={"x": "Jumlah Klaster (K)", "y": "Skor Silhouette"},
        )
        st.plotly_chart(fig_sil, width="stretch")

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    features["cluster"] = km_final.fit_predict(X_scaled)
    cluster_labels = {
        0: "Tier 1: Terjangkau & Volatilitas Tinggi",
        1: "Tier 2: Stabil Terkendali",
        2: "Tier 3: Premium & Bernilai Tinggi",
    }
    features["segment"] = features["cluster"].map(lambda c: cluster_labels.get(c, f"Tier {c+1}"))

    for c in features.index:
        forecast_summary[c]["cluster"] = features.loc[c, "segment"]

    fig_cluster = px.scatter(
        features.reset_index(),
        x="mean_price",
        y="growth",
        size="volatility",
        color="segment",
        hover_name="komoditas",
        title="Segmentasi Pasar: Harga Rata-rata vs Pertumbuhan 2-Tahun",
        labels={
            "mean_price": "Harga Rata-rata (IDR)",
            "growth": "Pertumbuhan 2-Tahun (%)",
            "volatility": "Volatilitas Standar Deviasi",
            "segment": "Segmen Pasar",
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_cluster.update_layout(height=500)
    st.plotly_chart(fig_cluster, width="stretch")

    st.subheader("Profil Rata-rata Klaster Komoditas")
    profile = features.groupby("segment")[["mean_price", "volatility", "growth"]].mean().round(2)
    profile.columns = ["Rata-rata Harga (IDR)", "Rata-rata Volatilitas", "Pertumbuhan (%)"]
    st.dataframe(profile, width="stretch")

    # Context and Bot for Tab 3
    context_tab3 = f"""Hasil Clustering K-Means Komoditas Pangan:
Jumlah Klaster Optimal: K = {best_k} (Skor Silhouette = {sil_scores[best_k]:.4f})
Profil Rata-rata Klaster:
{profile.to_string()}
Daftar Anggota Klaster:
{features[['segment', 'mean_price', 'volatility', 'growth']].to_string()}
"""
    sample_q_tab3 = [
        "Mengapa model memilih 3 klaster sebagai segmentasi terbaik?",
        "Apa saja karakteristik komoditas pada Tier 1?",
        "Apa langkah penanganan khusus untuk komoditas pada masing-masing klaster?",
    ]
    render_interactive_bot("tab3", "Clustering dan Segmentasi", context_tab3, gemini_key_input, sample_q_tab3)

# -- Tab 4: Forecasting ------------------------------------------------
with tab4:
    st.subheader("Proyeksi Harga Pangan Tahun 2026 — Model Regresi Linier")

    forecasts = []
    for commodity in all_commodities:
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
            "Selisih (IDR)": round(forecast_2026 - y_train[-1], 0),
            "Perubahan (%)": round((forecast_2026 - y_train[-1]) / y_train[-1] * 100, 2),
            "MAE": round(mae, 0),
            "RMSE": round(rmse, 0),
            "R2": round(r2, 4),
        })

    forecast_df = pd.DataFrame(forecasts)
    st.dataframe(forecast_df.set_index("Komoditas"), width="stretch")

    fig_forecast = px.bar(
        forecast_df,
        x="Komoditas",
        y=["Harga 2025 (IDR)", "Prediksi 2026 (IDR)"],
        barmode="group",
        title="Harga Aktual 2025 vs Estimasi Prediksi 2026",
        color_discrete_sequence=["#3780BF", "#DB4052"],
        labels={"value": "Harga (IDR)", "variable": "Kategori Data"},
    )
    fig_forecast.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig_forecast, width="stretch")

    st.subheader("Ringkasan Metrik Evaluasi Model")
    avg_r2 = forecast_df["R2"].mean()
    avg_mae = forecast_df["MAE"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Rata-rata R2", f"{avg_r2:.4f}")
    c2.metric("Rata-rata MAE", f"Rp {avg_mae:,.0f}")
    c3.metric("Total Komoditas Terprediksi", f"{len(forecast_df)}")

    # Context and Bot for Tab 4
    context_tab4 = f"""Hasil Peramalan Harga Pangan Tahun 2026 (Regresi Linier):
Rata-rata Akurasi R2: {avg_r2:.4f}
Rata-rata MAE: Rp {avg_mae:,.0f}
Tabel Proyeksi Harga 2026:
{forecast_df.to_string()}
"""
    sample_q_tab4 = [
        "Komoditas mana yang diprediksi mengalami lonjakan nominal tertinggi di 2026?",
        "Seberapa andal akurasi model regresi linier dalam peramalan harga ini?",
        "Langkah mitigasi apa yang perlu disiapkan sebelum tahun 2026?",
    ]
    render_interactive_bot("tab4", "Forecasting 2026", context_tab4, gemini_key_input, sample_q_tab4)

# -- Tab 5: AI Policy Recommendations (Google AI Studio) --------------
with tab5:
    st.subheader("Rekomendasi Kebijakan dan Strategi Pasar Pangan Berbasis AI")
    st.markdown(
        """
        Sistem Rekomendasi Kebijakan Cerdas memanfaatkan model Google Gemini AI dari Google AI Studio.
        Modul ini mengonversi metrik statistik harga, volatilitas, hasil clustering, dan estimasi peramalan 2026
        menjadi rekomendasi kebijakan terstruktur bagi pembuat keputusan (Bapanas, Kementerian Perdagangan) 
        serta panduan mitigasi risiko bagi pelaku usaha dan konsumen.
        """
    )

    selected_ai_commodity = st.selectbox(
        "Pilih Komoditas untuk Dianalisis oleh AI:",
        options=all_commodities,
        index=0,
    )

    m = forecast_summary.get(selected_ai_commodity, {})

    # Display KPI metrics for selected commodity
    ai_col1, ai_col2, ai_col3, ai_col4 = st.columns(4)
    ai_col1.metric("Harga Aktual 2025", f"Rp {m.get('price_2025', 0):,.0f}")
    ai_col2.metric("Proyeksi 2026", f"Rp {m.get('forecast_2026', 0):,.0f}", delta=f"{m.get('forecast_pct', 0):+.2f}%")
    ai_col3.metric("CAGR 2-Tahun", f"{m.get('cagr', 0):+.2f}%")
    ai_col4.metric("Klaster Pasar", m.get("cluster", "Tier Standar").split(":")[0])

    st.markdown("---")

    # Recommendation Engine Logic
    def generate_gemini_intelligence(api_key, commodity, metrics):
        headers = {"Content-Type": "application/json"}
        prompt = f"""
Anda adalah Senior Food Security & Agricultural Policy Advisor untuk Badan Pangan Nasional (Bapanas) dan Kementerian Perdagangan RI.
Analisis data intelijen harga komoditas pangan pokok Indonesia berikut:

- Komoditas: {commodity}
- Harga Rata-rata 2023: Rp {metrics.get('price_2023', 0):,.0f}
- Harga Rata-rata 2024: Rp {metrics.get('price_2024', 0):,.0f}
- Harga Rata-rata 2025: Rp {metrics.get('price_2025', 0):,.0f}
- Laju Inflasi YoY (2024 ke 2025): {metrics.get('yoy_24_25', 0):.2f}%
- Laju Pertumbuhan Majemuk (CAGR 2-Tahun): {metrics.get('cagr', 0):.2f}%
- Segmen Klaster Pasar: {metrics.get('cluster', 'N/A')}
- Estimasi Model Prediksi Tahun 2026: Rp {metrics.get('forecast_2026', 0):,.0f} (Perubahan: {metrics.get('forecast_pct', 0):+.2f}%)

Berikan dokumen analisis strategis eksekutif yang lugas, terstruktur, berbasis bukti kuantitatif di atas, dan TANPA menggunakan emoji atau simbol bintang (asterisk) apapun. DILARANG MENGGUNAKAN FORMAT BOLD.
Format laporan mencakup 4 bagian:

1. DIAGNOSIS STRUKTUR PASAR DAN VOLATILITAS
   - Evaluasi laju kenaikan harga dan faktor fundamental pasokan/permintaan (musim panen, disparitas logistik antarpulau, ongkos input produksi/pakan/pupuk).
2. TINGKAT RISIKO TERHADAP STABILITAS INFLASI
   - Penilaian dampak volatilitas harga komoditas ini terhadap daya beli masyarakat berpenghasilan menengah ke bawah.
3. REKOMENDASI KEBIJAKAN REGULATOR (BAPANAS / KEMENDAG / KEMENTAN)
   - Tiga langkah intervensi konkret (contoh: Operasi Pasar Terfokus, Fasilitasi Distribusi Pangan/FDP, Manajemen Cadangan Pangan Pemerintah/CPP, atau Penyesuaian Harga Acuan Pembelian/Penjualan).
4. REKOMENDASI STRATEGIS UNTUK PELAKU USAHA DAN KONSUMEN
   - Rekomendasi mitigasi persediaan (safety stock / forward purchase) untuk pelaku usaha dan panduan belanja cerdas atau diversifikasi pangan untuk rumah tangga.
"""
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1500,
            },
        }

        candidate_models = ["gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-3.8-flash"]
        last_error = ""

        for model_name in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=25)
                if resp.status_code == 200:
                    data = resp.json()
                    raw = data["candidates"][0]["content"]["parts"][0]["text"]
                    return sanitize_plain_text(raw)
                elif resp.status_code == 404:
                    continue
                else:
                    last_error = f"Status {resp.status_code}: {resp.text}"
            except Exception as e:
                last_error = str(e)

        return f"Gagal menghubungi model Google AI Studio: {last_error}"

    # Heuristic Rule-Based Fallback
    def generate_heuristic_recommendation(commodity, metrics):
        cagr = metrics.get("cagr", 0.0)
        pct_change = metrics.get("forecast_pct", 0.0)
        p25 = metrics.get("price_2025", 0.0)
        f26 = metrics.get("forecast_2026", 0.0)

        if cagr > 10.0:
            risk = "TINGGI (Peringatan Inflasi Kritis)"
            diag = f"{commodity} mengalami akselerasi kenaikan harga yang sangat cepat dengan CAGR 2-tahun mencapai {cagr:.2f}%. Lonjakan harga ini didorong oleh keterbatasan pasokan sentra produksi, peningkatan ongkos distribusi antar wilayah, dan biaya input pertanian."
            reg = [
                "Pelaksanaan Gerakan Pangan Murah (GPM) dan Operasi Pasar Khusus secara intensif di sentra-sentra konsumen utama.",
                "Fasilitasi Distribusi Pangan (FDP) dengan subsidi ongkos angkut dari wilayah surplus ke wilayah defisit pasokan.",
                "Penguatan sinergi pengawasan jalur tata niaga bersama Satgas Pangan untuk mencegah penahanan pasokan oleh spekulan.",
            ]
            biz = [
                "Pelaku industri dan pedagang besar disarankan menerapkan kontrak pasokan berjangka untuk memitigasi volatilitas harga.",
                "Konsumen rumah tangga disarankan mengoptimalkan efisiensi konsumsi dan mempertimbangkan alternatif substitusi olahan.",
            ]
        elif cagr > 3.0:
            risk = "SEDANG (Fluktuasi Moderat)"
            diag = f"{commodity} menunjukkan tren kenaikan harga moderat (CAGR {cagr:.2f}%). Fluktuasi harga lebih banyak dipengaruhi oleh siklus musiman dan iklim cuaca berkala."
            reg = [
                "Optimalisasi penyerapan komoditas lokal saat masa panen raya sebagai Cadangan Pangan Pemerintah (CPP).",
                "Penyediaan cold storage atau fasilitas pascapanen modern di sentra produksi guna memperpanjang masa simpan produk.",
                "Pemantauan dini sistem neraca pangan wilayah secara harian.",
            ]
            biz = [
                "Pelaku usaha disarankan menjaga tingkat persediaan penyangga (buffer stock) minimum 2 hingga 3 minggu.",
                "Konsumen dapat melakukan pembelian secara terjadwal mengikuti kalender masa panen.",
            ]
        else:
            risk = "RENDAH / STABIL (Harga Terkendali)"
            diag = f"{commodity} mencatatkan pergerakan harga yang sangat stabil dengan CAGR {cagr:.2f}%. Pasokan di pasar nasional memadai dan distribusi berjalan dengan lancar."
            reg = [
                "Menjaga perlindungan harga di tingkat petani/peternak agar tidak anjlok di bawah Harga Acuan Pembelian (HAP).",
                "Mempertahankan kelancaran arus logistik tol laut dan jalur distribusi antar pulau.",
                "Mendorong hilirisasi produk pangan guna meningkatkan nilai tambah produksi lokal.",
            ]
            biz = [
                "Pelaku usaha dapat menerapkan pengadaan Just-In-Time (JIT) karena risiko guncangan harga tergolong rendah.",
                "Konsumen menikmati stabilitas harga tanpa kekhawatiran lonjakan mendadak.",
            ]

        output = f"""### Laporan Rekomendasi Intelijen Pasar: {commodity}

Tingkat Risiko Pasar: {risk}  
Ringkasan Prediksi: Harga 2025 tercatat Rp {p25:,.0f} dan diperkirakan mencapai Rp {f26:,.0f} pada 2026 ({pct_change:+.2f}%).

1. DIAGNOSIS STRUKTUR PASAR DAN VOLATILITAS
{diag}

2. TINGKAT RISIKO TERHADAP STABILITAS INFLASI
Komoditas ini memberikan kontribusi yang terukur terhadap Indeks Harga Konsumen (IHK). Pada level pertumbuhan saat ini, diperlukan pengawasan rutin pada rantai distribusi agar tidak menciptakan efek domino terhadap biaya hidup masyarakat.

3. REKOMENDASI KEBIJAKAN REGULATOR (BAPANAS / KEMENDAG / KEMENTAN)
- {reg[0]}
- {reg[1]}
- {reg[2]}

4. REKOMENDASI STRATEGIS UNTUK PELAKU USAHA DAN KONSUMEN
- {biz[0]}
- {biz[1]}
"""
        return sanitize_plain_text(output)

    if gemini_key_input:
        st.info("API Key Google AI Studio terdeteksi. Anda dapat menghasilkan analisis komprehensif menggunakan model Gemini.")
        if st.button("Generate Rekomendasi Strategis via Gemini AI"):
            with st.spinner("Sedang memproses analisis ekonomi komoditas menggunakan Google Gemini AI..."):
                try:
                    ai_result = generate_gemini_intelligence(gemini_key_input, selected_ai_commodity, m)
                    st.markdown(ai_result)
                except Exception as ex:
                    st.error(f"Terjadi kesalahan saat memanggil API: {str(ex)}")
    else:
        st.markdown(
            """
            > Catatan Penggunaan: Masukkan Google Gemini API Key pada sidebar untuk mengaktifkan penalaran generatif langsung dari Google AI Studio.
            > Di bawah ini disajikan Laporan Rekomendasi Ekonometrika Standar berbasis algoritma rule-based internal.
            """
        )
        heuristic_text = generate_heuristic_recommendation(selected_ai_commodity, m)
        st.markdown(heuristic_text)

    # Interactive Bot for Tab 5
    context_tab5 = f"""Kebijakan Pangan Nasional dan Analisis Komoditas Terpilih:
Komoditas: {selected_ai_commodity}
Harga 2023-2025: Rp {m.get('price_2023', 0):,.0f} -> Rp {m.get('price_2024', 0):,.0f} -> Rp {m.get('price_2025', 0):,.0f}
Laju CAGR 2-Tahun: {m.get('cagr', 0):.2f}%
Prediksi Harga 2026: Rp {m.get('forecast_2026', 0):,.0f}
Segmen Klaster: {m.get('cluster', 'N/A')}
"""
    sample_q_tab5 = [
        "Apa rekomendasi intervensi pasar terfokus untuk Bapanas?",
        "Bagaimana koordinasi subsidi logistik antarpulau untuk menekan harga cabai dan bawang?",
        "Apa panduan manajemen stok bahan baku untuk pelaku usaha kuliner dan ritel?",
    ]
    render_interactive_bot("tab5", "Rekomendasi Kebijakan AI", context_tab5, gemini_key_input, sample_q_tab5)

# -- Footer ------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.85em;'>
        Harga Pangan Indonesia — Food Price Intelligence Platform<br/>
        Author: <a href='https://fauzinoorsyabani.my.id' target='_blank'>Fauzi Noorsyabani</a> | 
        Data: BPS / Kementerian Perdagangan RI | 2026
    </div>
    """,
    unsafe_allow_html=True,
)
