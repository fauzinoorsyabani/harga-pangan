# 🍚 Indonesian Food Commodity Price Intelligence Platform
### Platform Analitik & Kecerdasan Harga Pangan Pokok Nasional (2023–2025)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

Platform analitik kecerdasan data dan pemodelan prediktif untuk memantau dinamika harga rata-rata **11 komoditas pangan pokok Indonesia** (Beras IR I, Beras IR III, Cabai Rawit Merah, Cabai Merah Keriting, Bawang Merah, Bawang Putih, Daging Ayam, Telur Ayam, Daging Sapi, Gula Pasir, dan Minyak Goreng) periode **2023–2025** berdasarkan data pemantauan harga BPS dan Kementerian Perdagangan RI.

---

## 📸 Tinjauan Visual Sistem

![Food Price Intelligence Platform](portfolio_assets/01_commodity_prices_overview.png)

---

## 🌟 Fitur Utama (Core Features)

1. **Ingesti & Normalisasi Data Otomatis (`src/data_loader.py`)**:
   - Parsing berkas XLS berstruktur tabel HTML murni menggunakan BeautifulSoup.
   - Penanganan pemisah ribuan koma khas format BPS dan koreksi inkonsistensi string typo (`CABAI RAWIT MERHA` ➔ `CABAI RAWIT MERAH`).
   - Ekspor otomatis ke berkas bersih terstandarisasi `data/harga_pangan_clean.csv`.

2. **Analisis Statistik & Ekonometrika Pangan (`src/analyzer.py`)**:
   - Komputasi laju inflasi Year-over-Year (YoY %) periode 2023→2024 dan 2024→2025.
   - Komputasi Compound Annual Growth Rate (CAGR 2-Tahun).
   - Pengukuran volatilitas dan Koefisien Variasi (CV %).

3. **Segmentasi Pasar via Unsupervised Machine Learning (`notebooks/03_clustering_komoditas.ipynb`)**:
   - Rekayasa fitur multi-dimensi (`mean_price`, `cv_pct`, `total_growth_pct`) dengan `StandardScaler`.
   - Optimasi $K$ berbasis **Silhouette Score Analysis** ($K=3$, skor **0.6038**).
   - Proyeksi sebaran pasar dalam ruang 2D menggunakan Principal Component Analysis (PCA).

4. **Peramalan Harga Pangan Runtun Waktu Tahun 2026 (`notebooks/04_forecasting_harga.ipynb`)**:
   - Pemodelan regresi linier per komoditas dengan rata-rata **$R^2 = 0.7744$** dan **MAPE = 2.07%**.
   - Estimasi harga komparatif 2025 vs proyeksi 2026.

5. **Visualisasi Tingkat Publikasi 300 DPI (`src/visualizer.py`)**:
   - Menghasilkan 8 grafik analitis beresolusi tinggi di folder `portfolio_assets/`.

6. **Dashboard Interaktif Streamlit (`app.py`)**:
   - KPI cards, grafik interaktif Plotly, multi-filter komoditas & tahun, dan 4 tab analitis mendalam.

---

## 📁 Struktur Direktori

```
harga-pangan/
├── data/
│   └── harga_pangan_clean.csv             # Data bersih hasil parsing
├── docs/
│   └── agents/                            # Dokumentasi triage & agen
├── notebooks/
│   ├── 01_eksplorasi_data.ipynb           # EDA & profiling data
│   ├── 02_analisis_tren.ipynb             # Tren & inflasi YoY
│   ├── 03_clustering_komoditas.ipynb      # K-Means clustering & PCA
│   ├── 04_forecasting_harga.ipynb         # Model regresi linear 2026
│   └── build_notebooks.py                 # Generator notebook otomatis
├── portfolio_assets/                      # 8 Visualisasi 300 DPI & web assets
├── scripts/
│   └── generate_portfolio_assets.py       # Generator WebP aset portofolio
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     # Pipeline ingesti & pembersihan
│   ├── analyzer.py                        # Komputasi statistik & ML
│   └── visualizer.py                      # Generator visualisasi 300 DPI
├── tests/
│   └── test_pipeline.py                   # 6 Unit & integration tests (100% Pass)
├── app.py                                 # Streamlit Interactive Dashboard
├── GLOSSARY.md                            # Terminologi domain proyek
├── PANDUAN_DAN_KESIMPULAN_PROYEK.md       # Panduan eksekusi & analisis 5W
├── requirements.txt                       # Daftar dependensi Python
└── README.md
```

---

## 🚀 Panduan Memulai Cepat (Quickstart)

### 1. Kloning Repositori & Pasang Dependensi
```bash
git clone https://github.com/fauzinoorsyabani/harga-pangan.git
cd harga-pangan
pip install -r requirements.txt
```

### 2. Jalankan Pipeline Lengkap
```bash
# Ingesti & bersihkan data
python src/data_loader.py

# Komputasi statistik & ML
python src/analyzer.py

# Buat seluruh visualisasi 300 DPI
python src/visualizer.py

# Bangun seluruh notebook
python notebooks/build_notebooks.py

# Jalankan pengujian otomatis
python -c "from tests.test_pipeline import *; test_data_ingestion_and_cleaning(); test_pivot_matrix_shape(); test_inflation_yoy_calculation(); test_cagr_calculation(); test_kmeans_clustering(); test_price_forecasting(); print('ALL TESTS PASSED!')"
```

### 3. Jalankan Dashboard Interaktif
```bash
streamlit run app.py
```

---

## 📊 Ringkasan Temuan Kunci

| Komoditas | Harga 2023 (IDR) | Harga 2025 (IDR) | CAGR 2Y (%) | Prediksi 2026 (IDR) | Karakteristik |
|:---|:---:|:---:|:---:|:---:|:---|
| **Bawang Merah** | Rp 39.974 | Rp 50.531 | **+12.43%** | Rp 55.809 | Inflasi tertinggi, sangat fluktuatif |
| **Minyak Goreng** | Rp 15.778 | Rp 19.652 | **+11.61%** | Rp 21.590 | Kenaikan persisten dua digit |
| **Gula Pasir** | Rp 14.955 | Rp 18.430 | **+11.01%** | Rp 20.168 | Kenaikan stabil berkelanjutan |
| **Beras IR III** | Rp 11.239 | Rp 13.599 | **+10.00%** | Rp 14.779 | Lonjakan tinggi di 2024, melandai di 2025 |
| **Beras IR I** | Rp 12.760 | Rp 15.169 | **+9.03%** | Rp 16.374 | Bahan pangan primer pokok |
| **Cabai Rawit Merah**| Rp 62.417 | Rp 70.942 | **+6.61%** | Rp 75.204 | Volatilitas musiman cuaca |
| **Daging Ayam** | Rp 40.143 | Rp 41.071 | **+1.15%** | Rp 41.535 | Sangat stabil terkendali |
| **Telur Ayam** | Rp 29.006 | Rp 29.297 | **+0.50%** | Rp 29.443 | Stabilitas tinggi, guncangan minimal |
| **Daging Sapi** | Rp 140.288 | Rp 138.353 | **-0.69%** | Rp 137.385 | Premium tier, tren sedikit deflasi |

---

## 👨‍💻 Pengembang

**Fauzi Noorsyabani**  
- Portfolio: [fauzinoorsyabani.my.id](https://fauzinoorsyabani.my.id)  
- GitHub: [@fauzinoorsyabani](https://github.com/fauzinoorsyabani)  
- LinkedIn: [linkedin.com/in/fauzinoorsyabani](https://linkedin.com/in/fauzinoorsyabani)  
- Email: fauzinoorsyabani05@gmail.com  

Lisensi: [MIT License](LICENSE)
