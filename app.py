"""
Aplikasi Streamlit — Segmentasi Nasabah Kartu Kredit dengan K-Means Clustering
Proyek: Implementasi Clustering untuk Menemukan Pola pada Data (Metodologi CRISP-DM)
Mahasiswa: Sekar — Sistem Informasi, Universitas Gunadarma

Cara menjalankan secara lokal:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --------------------------------------------------------------------------
# Konfigurasi halaman
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Segmentasi Nasabah Kartu Kredit",
    page_icon="💳",
    layout="wide",
)

# --------------------------------------------------------------------------
# Memuat model & artefak hasil training (dari notebook CRISP-DM)
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    imputer = joblib.load("imputer.pkl")
    feature_names = joblib.load("feature_names.pkl")
    median_values = joblib.load("median_values.pkl")
    return kmeans, scaler, imputer, feature_names, median_values


kmeans, scaler, imputer, feature_names, median_values = load_artifacts()

SEGMENT_LABELS = {
    0: "Low Activity / Nasabah Pasif",
    1: "Revolver / Pengguna Cash Advance",
    2: "Heavy Spender / Nasabah Premium",
    3: "Transactor / Pembayar Disiplin",
}

SEGMENT_DESC = {
    "Low Activity / Nasabah Pasif": "Saldo dan aktivitas transaksi rendah. Rekomendasi: kampanye edukasi & insentif penggunaan kartu.",
    "Revolver / Pengguna Cash Advance": "Sering menggunakan cash advance dan jarang melakukan full payment. Rekomendasi: program restrukturisasi / cicilan bunga rendah.",
    "Heavy Spender / Nasabah Premium": "Nilai pembelian dan limit kartu tinggi. Rekomendasi: program loyalty/reward & penawaran kenaikan limit.",
    "Transactor / Pembayar Disiplin": "Rasio full payment tinggi, saldo terkendali. Rekomendasi: tawarkan produk tabungan/investasi (cross-sell).",
}

# --------------------------------------------------------------------------
# Sidebar navigasi
# --------------------------------------------------------------------------
st.sidebar.title("💳 Menu")
page = st.sidebar.radio(
    "Navigasi",
    ["Prediksi Segmen Nasabah", "Prediksi Massal (Upload CSV)", "Tentang Proyek"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Proyek Data Science — Implementasi Clustering dengan Metodologi CRISP-DM\n\n"
    "Dataset: Credit Card Dataset for Clustering (Kaggle)"
)

# --------------------------------------------------------------------------
# Halaman 1: Prediksi Segmen untuk 1 nasabah (input manual)
# --------------------------------------------------------------------------
if page == "Prediksi Segmen Nasabah":
    st.title("💳 Prediksi Segmen Nasabah Kartu Kredit")
    st.write(
        "Masukkan data perilaku transaksi seorang nasabah untuk mengetahui "
        "termasuk ke dalam segmen (cluster) mana nasabah tersebut, berdasarkan "
        "model **K-Means Clustering** yang telah dilatih pada notebook CRISP-DM."
    )

    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            balance = st.number_input("Balance (saldo)", min_value=0.0, value=float(median_values["BALANCE"]))
            balance_freq = st.slider("Balance Frequency", 0.0, 1.0, float(median_values["BALANCE_FREQUENCY"]))
            purchases = st.number_input("Purchases (total pembelian)", min_value=0.0, value=float(median_values["PURCHASES"]))
            oneoff_purchases = st.number_input("One-off Purchases", min_value=0.0, value=float(median_values["ONEOFF_PURCHASES"]))
            installments_purchases = st.number_input("Installments Purchases", min_value=0.0, value=float(median_values["INSTALLMENTS_PURCHASES"]))
            cash_advance = st.number_input("Cash Advance", min_value=0.0, value=float(median_values["CASH_ADVANCE"]))

        with col2:
            purchases_freq = st.slider("Purchases Frequency", 0.0, 1.0, float(median_values["PURCHASES_FREQUENCY"]))
            oneoff_freq = st.slider("One-off Purchases Frequency", 0.0, 1.0, float(median_values["ONEOFF_PURCHASES_FREQUENCY"]))
            installments_freq = st.slider("Purchases Installments Frequency", 0.0, 1.0, float(median_values["PURCHASES_INSTALLMENTS_FREQUENCY"]))
            cash_advance_freq = st.slider("Cash Advance Frequency", 0.0, 1.5, float(median_values["CASH_ADVANCE_FREQUENCY"]))
            cash_advance_trx = st.number_input("Cash Advance TRX (jumlah transaksi)", min_value=0, value=int(median_values["CASH_ADVANCE_TRX"]))
            purchases_trx = st.number_input("Purchases TRX (jumlah transaksi)", min_value=0, value=int(median_values["PURCHASES_TRX"]))

        with col3:
            credit_limit = st.number_input("Credit Limit", min_value=0.0, value=float(median_values["CREDIT_LIMIT"]))
            payments = st.number_input("Payments (total pembayaran)", min_value=0.0, value=float(median_values["PAYMENTS"]))
            minimum_payments = st.number_input("Minimum Payments", min_value=0.0, value=float(median_values["MINIMUM_PAYMENTS"]))
            prc_full_payment = st.slider("Percent Full Payment", 0.0, 1.0, float(median_values["PRC_FULL_PAYMENT"]))
            tenure = st.number_input("Tenure (bulan)", min_value=6, max_value=12, value=int(median_values["TENURE"]))

        submitted = st.form_submit_button("🔍 Prediksi Segmen", use_container_width=True)

    if submitted:
        input_dict = {
            "BALANCE": balance,
            "BALANCE_FREQUENCY": balance_freq,
            "PURCHASES": purchases,
            "ONEOFF_PURCHASES": oneoff_purchases,
            "INSTALLMENTS_PURCHASES": installments_purchases,
            "CASH_ADVANCE": cash_advance,
            "PURCHASES_FREQUENCY": purchases_freq,
            "ONEOFF_PURCHASES_FREQUENCY": oneoff_freq,
            "PURCHASES_INSTALLMENTS_FREQUENCY": installments_freq,
            "CASH_ADVANCE_FREQUENCY": cash_advance_freq,
            "CASH_ADVANCE_TRX": cash_advance_trx,
            "PURCHASES_TRX": purchases_trx,
            "CREDIT_LIMIT": credit_limit,
            "PAYMENTS": payments,
            "MINIMUM_PAYMENTS": minimum_payments,
            "PRC_FULL_PAYMENT": prc_full_payment,
            "TENURE": tenure,
        }

        input_df = pd.DataFrame([input_dict])[feature_names]
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)
        cluster_pred = int(kmeans.predict(input_scaled)[0])
        segment_name = SEGMENT_LABELS.get(cluster_pred, f"Cluster {cluster_pred}")

        st.success(f"### Hasil Prediksi: Cluster {cluster_pred} — **{segment_name}**")
        st.info(SEGMENT_DESC.get(segment_name, ""))

        distances = kmeans.transform(input_scaled)[0]
        dist_df = pd.DataFrame({
            "Cluster": [f"Cluster {i} ({SEGMENT_LABELS.get(i, '')})" for i in range(len(distances))],
            "Jarak ke Centroid": distances,
        }).sort_values("Jarak ke Centroid")
        st.write("**Jarak nasabah ke tiap centroid cluster** (semakin kecil = semakin mirip):")
        st.dataframe(dist_df, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# Halaman 2: Prediksi massal via upload CSV
# --------------------------------------------------------------------------
elif page == "Prediksi Massal (Upload CSV)":
    st.title("📂 Prediksi Segmen Massal dari File CSV")
    st.write(
        "Unggah file CSV berisi data nasabah dengan struktur kolom yang sama "
        "dengan dataset *Credit Card Dataset for Clustering* (boleh menyertakan "
        "kolom `CUST_ID`, kolom tersebut akan diabaikan pada proses clustering)."
    )

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Pratinjau data yang diunggah:")
        st.dataframe(data.head(), use_container_width=True)

        missing_cols = [c for c in feature_names if c not in data.columns]
        if missing_cols:
            st.error(f"Kolom berikut tidak ditemukan pada file: {missing_cols}")
        else:
            X = data[feature_names]
            X_imputed = imputer.transform(X)
            X_scaled = scaler.transform(X_imputed)
            preds = kmeans.predict(X_scaled)

            result = data.copy()
            result["Cluster"] = preds
            result["Segmen"] = result["Cluster"].map(SEGMENT_LABELS)

            st.success(f"Prediksi berhasil untuk {len(result)} baris data.")
            st.dataframe(result, use_container_width=True)

            st.write("**Distribusi jumlah nasabah per segmen:**")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(data=result, x="Segmen", palette="Set2", ax=ax,
                          order=result["Segmen"].value_counts().index)
            ax.set_xlabel("")
            ax.set_ylabel("Jumlah Nasabah")
            plt.xticks(rotation=15)
            st.pyplot(fig)

            csv_out = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Unduh Hasil Prediksi (CSV)",
                data=csv_out,
                file_name="hasil_segmentasi_nasabah.csv",
                mime="text/csv",
            )

# --------------------------------------------------------------------------
# Halaman 3: Tentang proyek
# --------------------------------------------------------------------------
else:
    st.title("ℹ️ Tentang Proyek")
    st.markdown(
        """
### Implementasi Clustering untuk Menemukan Pola pada Data
**Metodologi:** CRISP-DM (Cross-Industry Standard Process for Data Mining)

**Dataset:** *Credit Card Dataset for Clustering* — [Kaggle](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
8.950 baris, 18 kolom (17 fitur numerik perilaku transaksi kartu kredit selama 6 bulan).

**Algoritma:** K-Means Clustering (k = 4), dengan tahapan:
- Data Preparation: imputasi *missing value*, penanganan *outlier* (capping), standardisasi fitur.
- Modeling: penentuan jumlah cluster optimal via *Elbow Method* & *Silhouette Score*.
- Evaluation: *Silhouette Score* dan *Davies-Bouldin Index*.
- Deployment: aplikasi web ini (Streamlit).

**Segmen yang dihasilkan:**
| Cluster | Segmen | Karakteristik |
|---|---|---|
| 0 | Low Activity / Nasabah Pasif | Aktivitas transaksi rendah |
| 1 | Revolver / Pengguna Cash Advance | Sering ambil cash advance, jarang full payment |
| 2 | Heavy Spender / Nasabah Premium | Pembelian & limit tinggi |
| 3 | Transactor / Pembayar Disiplin | Rasio full payment tinggi |

**Dibuat oleh:** Sekar — Sistem Informasi, Universitas Gunadarma
"""
    )
