"""
data_loader.py — Modul Pemuatan & Pembersihan Data / Data Loading & Cleaning Module

Membaca file XLS (format HTML table) berisi data harga rata-rata bahan pangan Indonesia
dan mengembalikan DataFrame yang bersih dan siap analisis.

Reads an XLS file (actually an HTML table) containing Indonesian food commodity
average prices and returns a clean, analysis-ready DataFrame.

Komoditas / Commodities (11 items):
    BERAS IR I, BERAS IR III, CABAI RAWIT MERAH, CABAI MERAH KERITING,
    BAWANG MERAH, BAWANG PUTIH, DAGING AYAM, TELUR AYAM,
    DAGING SAPI, GULA PASIR, MINYAK GORENG

Periode / Period: 2023, 2024, 2025
"""

import os
import sys
import pandas as pd
from bs4 import BeautifulSoup

# Ensure utf-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================================
# Konfigurasi Path / Path Configuration
# ============================================================================
# Mendukung eksekusi dari root project maupun dari folder src/
# Supports execution from project root or from src/ folder
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

# Path default file sumber dan output
# Default source file and output paths
DEFAULT_XLS_PATH = os.path.join(
    _PROJECT_ROOT, "harga-rata-rata-bahan-pangan-komponen-data.xls"
)
DEFAULT_CSV_OUTPUT = os.path.join(_PROJECT_ROOT, "data", "harga_pangan_clean.csv")


# ============================================================================
# Pemetaan Normalisasi Nama Komoditas / Commodity Name Normalization Map
# ============================================================================
# Memperbaiki typo dan inkonsistensi penamaan
# Fixes typos and naming inconsistencies
COMMODITY_NAME_MAP = {
    "CABAI RAWIT MERHA": "CABAI RAWIT MERAH",  # Typo pada data 2023 & 2024
}


def read_html_xls(filepath: str = DEFAULT_XLS_PATH) -> pd.DataFrame:
    """
    Membaca file XLS yang sebenarnya berformat HTML table.
    Reads an XLS file that is actually an HTML table.

    Parameters
    ----------
    filepath : str
        Path ke file XLS / Path to the XLS file

    Returns
    -------
    pd.DataFrame
        DataFrame mentah dari tabel HTML / Raw DataFrame from HTML table
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File tidak ditemukan / File not found: {filepath}"
        )

    # Baca file sebagai HTML / Read file as HTML
    with open(filepath, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Parse langsung dari HTML Table menggunakan BeautifulSoup (sangat stabil & deterministik)
    # Direct parsing from HTML Table using BeautifulSoup (highly robust & deterministic)
    soup = BeautifulSoup(html_content, "html.parser")
    table = soup.find("table")
    if not table:
        raise ValueError(
            "Tidak ada tabel ditemukan dalam file / No tables found in file"
        )

    all_rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if cells:
            all_rows.append(cells)

    if len(all_rows) < 2:
        raise ValueError("Tabel tidak memiliki data / Table has no data")

    headers = [h.strip() for h in all_rows[0]]
    df = pd.DataFrame(all_rows[1:], columns=headers)
    print(f"✅ Data mentah dimuat: {len(df)} baris × {len(df.columns)} kolom")
    print(f"   Raw data loaded: {len(df)} rows × {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan dan menormalisasi DataFrame harga pangan.
    Cleans and normalizes the food price DataFrame.

    Langkah-langkah / Steps:
    1. Normalisasi nama kolom (lowercase, strip)
    2. Perbaiki typo nama komoditas
    3. Konversi harga dari string ke float (handle comma separator)
    4. Konversi periode ke integer
    5. Urutkan berdasarkan periode dan komoditas
    6. Reset index

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mentah / Raw DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame bersih / Clean DataFrame
    """
    df = df.copy()

    # --- Langkah 1: Normalisasi nama kolom / Step 1: Normalize column names ---
    df.columns = [col.strip().lower() for col in df.columns]

    # Pastikan kolom yang dibutuhkan ada / Ensure required columns exist
    required_cols = {"periode_data", "komoditas", "harga_rata_rata"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Kolom tidak ditemukan / Missing columns: {missing}"
        )

    # --- Langkah 2: Perbaiki typo komoditas / Step 2: Fix commodity typos ---
    df["komoditas"] = df["komoditas"].str.strip().str.upper()
    df["komoditas"] = df["komoditas"].replace(COMMODITY_NAME_MAP)
    n_fixed = sum(
        df["komoditas"].value_counts().get(v, 0)
        for v in COMMODITY_NAME_MAP.values()
    )
    print(f"🔧 Nama komoditas dinormalisasi / Commodity names normalized")

    # --- Langkah 3: Konversi harga ke float / Step 3: Convert price to float ---
    # Handle format Indonesia: "12,759.81" (comma = thousands, dot = decimal)
    df["harga_rata_rata"] = (
        df["harga_rata_rata"]
        .astype(str)
        .str.replace(",", "", regex=False)  # Hapus pemisah ribuan / Remove thousands sep
        .astype(float)
    )

    # --- Langkah 4: Konversi periode ke int / Step 4: Convert period to int ---
    df["periode_data"] = df["periode_data"].astype(int)

    # --- Langkah 5: Urutkan data / Step 5: Sort data ---
    df = df.sort_values(
        ["periode_data", "komoditas"], ascending=[True, True]
    ).reset_index(drop=True)

    # --- Validasi akhir / Final validation ---
    n_commodities = df["komoditas"].nunique()
    n_years = df["periode_data"].nunique()
    print(f"✅ Data bersih: {len(df)} baris, {n_commodities} komoditas, {n_years} tahun")
    print(f"   Clean data: {len(df)} rows, {n_commodities} commodities, {n_years} years")
    print(f"   Periode / Period: {sorted(df['periode_data'].unique())}")
    print(f"   Rentang harga / Price range: Rp {df['harga_rata_rata'].min():,.2f} — Rp {df['harga_rata_rata'].max():,.2f}")

    return df


def get_clean_data(
    filepath: str = DEFAULT_XLS_PATH,
    save_csv: bool = True,
    csv_output: str = DEFAULT_CSV_OUTPUT,
) -> pd.DataFrame:
    """
    Pipeline utama: muat, bersihkan, dan (opsional) simpan data.
    Main pipeline: load, clean, and (optionally) save data.

    Parameters
    ----------
    filepath : str
        Path ke file XLS sumber / Path to source XLS file
    save_csv : bool
        Simpan CSV output atau tidak / Whether to save CSV output
    csv_output : str
        Path file CSV output / Output CSV file path

    Returns
    -------
    pd.DataFrame
        DataFrame bersih siap analisis / Clean analysis-ready DataFrame
    """
    print("=" * 60)
    print("📦 PIPELINE PEMUATAN DATA / DATA LOADING PIPELINE")
    print("=" * 60)

    # Muat data mentah / Load raw data
    df_raw = read_html_xls(filepath)

    # Bersihkan data / Clean data
    df_clean = clean_data(df_raw)

    # Simpan ke CSV / Save to CSV
    if save_csv:
        os.makedirs(os.path.dirname(csv_output), exist_ok=True)
        df_clean.to_csv(csv_output, index=False)
        print(f"💾 Data tersimpan di / Data saved to: {csv_output}")

    print("=" * 60)
    return df_clean


# ============================================================================
# Eksekusi langsung / Direct execution
# ============================================================================
if __name__ == "__main__":
    df = get_clean_data()
    print("\n📊 Preview Data:")
    print(df.to_string(index=False))
    print(f"\n📋 Info DataFrame:")
    print(df.dtypes)
    print(f"\n📈 Statistik Deskriptif / Descriptive Statistics:")
    print(df.describe())
