# GLOSSARY — Harga Pangan Indonesia Intelligence Platform

Shared domain vocabulary for the Indonesian Food Commodity Price Intelligence Platform (`d:\Code\harga-pangan`).

| Term | Definition | Code Representation |
| :--- | :--- | :--- |
| **Komoditas (Commodity)** | One of 11 staple food items tracked by the Indonesian government price monitoring system: Beras IR I, Beras IR III, Cabai Rawit Merah, Cabai Merah Keriting, Bawang Merah, Bawang Putih, Daging Ayam, Telur Ayam, Daging Sapi, Gula Pasir, Minyak Goreng. | `komoditas` column |
| **Harga Rata-Rata (Average Price)** | The national average retail price per kilogram (or per liter for Minyak Goreng) in Indonesian Rupiah (IDR), sourced from BPS/Kemendag monitoring. | `harga_rata_rata` column |
| **Periode Data (Data Period)** | The calendar year (2023, 2024, 2025) for which the average price is reported. Annual granularity. | `periode_data` column |
| **Inflasi Komoditas (Commodity Inflation)** | Year-over-year percentage price change: $\Delta\% = \frac{P_{t} - P_{t-1}}{P_{t-1}} \times 100$. Positive = price increase, negative = price decrease. | `inflasi_yoy` computed column |
| **Komoditas Volatil (Volatile Commodity)** | A commodity whose YoY price change exceeds ±15% in any observed period. Primarily horticultural items (cabai, bawang). | Flagged in clustering analysis |
| **Komoditas Stabil (Stable Commodity)** | A commodity whose YoY price change stays within ±5% across all observed periods. Primarily protein items (daging sapi, telur ayam). | Flagged in clustering analysis |
| **Segmentasi Harga (Price Segmentation)** | K-Means clustering of commodities into price tiers (Premium, Menengah, Terjangkau) based on absolute price level and volatility. | `price_segment` computed column |
| **CAGR (Compound Annual Growth Rate)** | Two-year compound growth: $\text{CAGR} = \left(\frac{P_{2025}}{P_{2023}}\right)^{1/2} - 1$. Measures sustained price pressure. | `cagr` computed column |
| **BPS** | Badan Pusat Statistik — Indonesia's Central Statistics Agency. Primary source for commodity price monitoring data. | Data provenance |
| **Kemendag** | Kementerian Perdagangan — Ministry of Trade. Co-publishes food price monitoring data alongside BPS. | Data provenance |

