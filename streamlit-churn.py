import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

st.set_page_config(page_title="Churn Prediction", layout="wide")

# Styling CSS
st.markdown("""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #1a1a1a;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* Sidebar button styling */
    [data-testid="stSidebar"] .stButton > button {
        background: #333;
        color: white !important;
        border: 1px solid #555;
        width: 100%;
        margin-bottom: 0.5rem;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #555;
        border-color: #777;
    }
    
    /* Main content */
    .main-header {
        background: #1a1a1a;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        border: 1px solid #333;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        color: white;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: #1a1a1a;
        color: white;
        border: 1px solid #333;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #333;
        border-color: #555;
    }
</style>
""", unsafe_allow_html=True)

ARTIFACT_PATH = "churn_artifacts.joblib"

# limit number of top-k features to use/display
TOP_K = 10


# ---------------------------
# Load model artifacts
# ---------------------------
@st.cache_resource
def load_artifacts(path: str):
    obj = joblib.load(path)

    required = ["model", "feature_cols_full", "scaler_full"]
    missing = [k for k in required if k not in obj]
    if missing:
        raise KeyError(f"Artifact kurang key wajib: {missing}")

    if "topk_idx" not in obj and "selected_features" not in obj:
        raise KeyError("Artifact harus punya salah satu: 'topk_idx' atau 'selected_features'.")

    return obj


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def get_selected_features(art: dict, limit: int | None = None) -> tuple[list[str], np.ndarray]:
    feature_cols_full = list(art["feature_cols_full"])

    if "topk_idx" in art and art["topk_idx"] is not None:
        topk_idx = np.asarray(art["topk_idx"])
        selected_features = [feature_cols_full[i] for i in topk_idx]
        # apply limit if requested
        if limit is not None and limit >= 0:
            selected_features = selected_features[:limit]
            topk_idx = topk_idx[:limit]
        return selected_features, topk_idx

    selected_features = list(art["selected_features"])
    topk_idx = np.array([feature_cols_full.index(f) for f in selected_features])
    if limit is not None and limit >= 0:
        selected_features = selected_features[:limit]
        topk_idx = topk_idx[:limit]
    return selected_features, topk_idx


def preprocess_input(df_in: pd.DataFrame, art: dict) -> tuple[np.ndarray, pd.DataFrame]:
    feature_cols_full = list(art["feature_cols_full"])
    scaler_full = art["scaler_full"]
    medians_full = art.get("medians_full", {})

    # use global limit if defined
    selected_features, topk_idx = get_selected_features(art, limit=TOP_K)

    df = df_in.copy()

    # input 21 fitur
    has_full = set(feature_cols_full).issubset(df.columns)

    # input 5 fitur
    has_topk_only = set(selected_features).issubset(df.columns)

    if not has_full and not has_topk_only:
        missing_full = sorted(list(set(feature_cols_full) - set(df.columns)))
        missing_topk = sorted(list(set(selected_features) - set(df.columns)))
        raise ValueError(
            "Kolom input tidak cocok.\n"
            f"- Jika pakai 21 fitur lengkap, kolom yang kurang: {missing_full}\n"
            f"- Jika pakai top-K saja, kolom yang kurang: {missing_topk}"
        )

    if has_full:
        df_used = df[feature_cols_full].copy()
        df_used = ensure_numeric(df_used, feature_cols_full)

        for c in feature_cols_full:
            if c in medians_full:
                df_used[c] = df_used[c].fillna(medians_full[c])

        X_full = df_used.to_numpy()
        X_full_scaled = scaler_full.transform(X_full)
        X_final = X_full_scaled[:, topk_idx]
        return X_final, df_used

    df_used = df[selected_features].copy()
    df_used = ensure_numeric(df_used, selected_features)

    for c in selected_features:
        if c in medians_full:
            df_used[c] = df_used[c].fillna(medians_full[c])

    mean_sel = scaler_full.mean_[topk_idx]
    scale_sel = scaler_full.scale_[topk_idx]
    X_sel = df_used.to_numpy()

    X_final = (X_sel - mean_sel) / scale_sel
    return X_final, df_used

def infer_label_candidates(df: pd.DataFrame) -> list[str]:
    candidates = []

    name_priority = {"churn", "target", "label", "y"}
    for c in df.columns:
        if c.lower() in name_priority:
            candidates.append(c)

    for c in df.columns:
        if c in candidates:
            continue
        s = df[c]
        s_num = pd.to_numeric(s, errors="coerce").dropna()
        if len(s_num) > 0:
            uniq = set(s_num.unique().tolist())
            if uniq.issubset({0, 1}):
                candidates.append(c)
                continue
        if s.dtype == bool:
            candidates.append(c)

    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

def predict_proba(df_in: pd.DataFrame, art: dict) -> tuple[np.ndarray, np.ndarray]:
    X_final, _ = preprocess_input(df_in, art)
    model = art["model"]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_final)[:, 1]
    else:
        scores = model.decision_function(X_final)
        proba = 1 / (1 + np.exp(-scores))

    return proba, X_final


def predict(df_in: pd.DataFrame, art: dict, threshold: float) -> pd.DataFrame:
    proba, _ = predict_proba(df_in, art)
    pred = (proba >= threshold).astype(int)

    out = df_in.copy()
    out["churn_proba"] = proba
    out["churn_pred"] = pred
    return out


# ---------------------------
# UI helpers (session data)
# ---------------------------
def read_csv_file(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(io.BytesIO(uploaded_file.read()))


def show_corr_heatmap(df: pd.DataFrame, cols: list[str]):
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        st.error("Kolom tidak cukup untuk membuat heatmap (minimal 2).")
        return

    df_num = ensure_numeric(df, cols)
    corr = df_num[cols].corr(numeric_only=True).fillna(0)

    n = len(cols)
    fig_w = max(8, min(20, 0.55 * n))
    fig_h = max(6, min(18, 0.55 * n))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticklabels(cols)

    ax.set_title("Correlation Matrix - Features")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=8,
                color="white" if abs(val) > 0.5 else "black"
            )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------
# Main App
# ---------------------------
st.markdown("""
<div class="main-header">
    <h1>📈 Customer Churn Dashboard</h1>
    <p>Prediksi pelanggan yang berpotensi churn menggunakan Machine Learning</p>
</div>
""", unsafe_allow_html=True)

try:
    artifacts = load_artifacts(ARTIFACT_PATH)
except Exception as e:
    st.error(f"Gagal load artifact: {e}")
    st.stop()

feature_cols_full = list(artifacts["feature_cols_full"])
# trim selected features to TOP_K if necessary
selected_features, topk_idx = get_selected_features(artifacts, limit=TOP_K)

threshold = 0.5

# Sidebar navigation
with st.sidebar:
    if "menu" not in st.session_state:
        st.session_state.menu = "Home"
    
    # Button navigation
    if st.button("Home", use_container_width=True):
        st.session_state.menu = "Home"
    if st.button("Upload Data", use_container_width=True):
        st.session_state.menu = "Upload Data"
    if st.button("Data Wrangling", use_container_width=True):
        st.session_state.menu = "Data Wrangling"
    if st.button("EDA", use_container_width=True):
        st.session_state.menu = "EDA"
    if st.button("Preprocessing", use_container_width=True):
        st.session_state.menu = "Data Preprocessing"
    if st.button("Modelling", use_container_width=True):
        st.session_state.menu = "Modelling"
    if st.button("Prediction", use_container_width=True):
        st.session_state.menu = "Prediction"
    
    menu = st.session_state.menu
    
    st.markdown("---")
    
    with st.expander("Model Info", expanded=False):
        st.write(f"Fitur full: **{len(feature_cols_full)}**")
        st.write(f"Top-K fitur: **{len(selected_features)}**")
        st.code(", ".join(selected_features))


# ---------------------------
# Pages
# ---------------------------
if menu == "Home":
    st.subheader("Home")
    st.write(
        "Aplikasi ini digunakan untuk analisis dan prediksi churn pelanggan. "
        "Navigasi di sidebar membagi alur menjadi EDA, preprocessing (termasuk korelasi), evaluasi model, dan prediksi."
    )
    st.info(
        "Catatan: Input prediksi bisa berupa CSV berisi **21 fitur lengkap** atau minimal **top-K fitur** sesuai model."
    )

elif menu == "Upload Data":
    st.subheader("Upload Data")
    up = st.file_uploader("Upload dataset (CSV)", type=["csv"])

    if up is not None:
        df = read_csv_file(up)
        st.session_state["df_uploaded"] = df
        st.success("Dataset tersimpan di session dan bisa dipakai di halaman EDA/Preprocessing/Evaluation.")
        st.write("Preview:")
        st.dataframe(df.head(20), use_container_width=True)
        st.caption("Jika dataset punya label, pastikan kolom target bernama `Churn` (atau bisa dipilih di halaman evaluasi).")

elif menu == "Data Wrangling":
    st.subheader("Data Wrangling")
    st.caption("Menampilkan proses pembersihan dan transformasi data sebelum digunakan untuk pemodelan.")

    if "df_uploaded" not in st.session_state:
        st.warning("Belum ada dataset. Upload dulu di menu **Upload Data**.")
    else:
        df_raw = st.session_state["df_uploaded"].copy()

        st.markdown("#### 1. Data Asli (Raw)")
        st.write(f"Ukuran: **{df_raw.shape[0]} baris × {df_raw.shape[1]} kolom**")
        st.dataframe(df_raw.head(10), use_container_width=True)

        st.markdown("#### 2. Tipe Data")
        dtype_df = pd.DataFrame({
            "Kolom": df_raw.dtypes.index,
            "Tipe Data": df_raw.dtypes.values.astype(str),
        }).reset_index(drop=True)
        st.dataframe(dtype_df, use_container_width=True)

        st.markdown("#### 3. Penanganan Missing Value")
        miss = df_raw.isna().sum()
        miss_df = pd.DataFrame({
            "Kolom": miss.index,
            "Jumlah Missing": miss.values,
            "Persentase (%)": (miss.values / len(df_raw) * 100).round(2),
        }).reset_index(drop=True)
        miss_df = miss_df[miss_df["Jumlah Missing"] > 0]

        if len(miss_df) == 0:
            st.success("Tidak ada missing value pada dataset.")
        else:
            st.dataframe(miss_df, use_container_width=True)

            medians_full = artifacts.get("medians_full", {})
            if medians_full:
                st.write("Imputasi missing value menggunakan **median** dari data training:")
                imp_df = pd.DataFrame([
                    {"Kolom": k, "Nilai Median (imputed)": v}
                    for k, v in medians_full.items()
                    if k in miss_df["Kolom"].values
                ])
                if len(imp_df) > 0:
                    st.dataframe(imp_df, use_container_width=True)
                else:
                    st.info("Tidak ada kolom missing yang cocok dengan median artifact.")
            else:
                st.info("Median artifact tidak tersedia, imputasi dilakukan dengan nilai kolom.")

        st.markdown("#### 4. Seleksi Fitur")
        st.write(f"Total fitur yang digunakan model: **{len(feature_cols_full)} fitur**")
        st.write(f"Top-**{len(selected_features)}** fitur terpilih berdasarkan feature importance:")
        feat_df = pd.DataFrame({"Fitur": selected_features, "Rank": range(1, len(selected_features) + 1)})
        st.dataframe(feat_df.set_index("Rank"), use_container_width=True)

        st.markdown("#### 5. Normalisasi (StandardScaler)")
        scaler = artifacts.get("scaler_full")
        if scaler is not None:
            scale_df = pd.DataFrame({
                "Fitur": feature_cols_full,
                "Mean": scaler.mean_,
                "Std (Scale)": scaler.scale_,
            })
            st.write("Parameter StandardScaler (mean & std dari data training):")
            st.dataframe(scale_df.style.format({"Mean": "{:.4f}", "Std (Scale)": "{:.4f}"}), use_container_width=True)
        else:
            st.info("Scaler tidak ditemukan di artifact.")

elif menu == "EDA":
    st.subheader("Exploratory Data Analysis (EDA)")

    if "df_uploaded" not in st.session_state:
        st.warning("Belum ada dataset. Upload dulu di menu **Upload Data**.")
    else:
        df = st.session_state["df_uploaded"]
        st.write(f"Ukuran data: **{df.shape[0]} baris × {df.shape[1]} kolom**")
        st.write("Preview:")
        st.dataframe(df.head(20), use_container_width=True)

        miss = df.isna().sum().sort_values(ascending=False)
        miss = miss[miss > 0]
        if len(miss) > 0:
            st.write("Ringkasan missing value:")
            st.dataframe(miss.rename("missing_count"), use_container_width=True)
        else:
            st.write("Tidak ada missing value yang terdeteksi.")

        if "Churn" in df.columns:
            st.write("Distribusi label Churn:")
            vc = df["Churn"].value_counts(dropna=False)
            st.dataframe(vc.rename("count"), use_container_width=True)

elif menu == "Data Preprocessing":
    st.subheader("Data Preprocessing")
    st.caption("Halaman ini menampilkan korelasi antar fitur (correlation heatmap).")

    if "df_uploaded" not in st.session_state:
        st.warning("Belum ada dataset. Upload dulu di menu **Upload Data**.")
    else:
        df = st.session_state["df_uploaded"].copy()

        if set(feature_cols_full).issubset(df.columns):
            usable_cols = feature_cols_full
            default_cols = feature_cols_full[:min(21, len(feature_cols_full))]
        elif set(selected_features).issubset(df.columns):
            usable_cols = selected_features
            default_cols = selected_features
        else:
            usable_cols = df.select_dtypes(include=["number"]).columns.tolist()
            default_cols = usable_cols[:min(10, len(usable_cols))]

        if len(usable_cols) < 2:
            st.error("Kolom numerik/fitur tidak cukup untuk membuat heatmap.")
        else:
            st.write(f"Kolom kandidat untuk korelasi: **{len(usable_cols)}**")

            cols_for_plot = st.multiselect(
                "Pilih kolom yang ingin ditampilkan di heatmap",
                options=usable_cols,
                default=default_cols
            )

            if len(cols_for_plot) < 2:
                st.warning("Pilih minimal 2 kolom.")
            else:
                show_corr_heatmap(df, cols_for_plot)

elif menu == "Modelling":
    st.subheader("Modelling")
    metrics_fixed = pd.DataFrame([
        {"Model":"XGBoost",            "Accuracy":0.9186,"Precision":0.9946,"Recall":0.9035,"F1-Score":0.9469,"ROC-AUC":0.9825},
        {"Model":"Random Forest",      "Accuracy":0.9092,"Precision":0.9943,"Recall":0.8920,"F1-Score":0.9404,"ROC-AUC":0.9808},
        {"Model":"Logistic Regression","Accuracy":0.6547,"Precision":0.9155,"Recall":0.6275,"F1-Score":0.7446,"ROC-AUC":0.7329},
        {"Model":"Stacking Ensemble",  "Accuracy":0.9240,"Precision":0.9922,"Recall":0.9124,"F1-Score":0.9506,"ROC-AUC":0.9825},
    ]).set_index("Model")

    st.dataframe(metrics_fixed.style.format("{:.4f}"), use_container_width=True)

elif menu == "Prediction":
    st.subheader("Prediction")
    st.caption(
        "Prediksi bisa dilakukan dari CSV (batch) atau input manual (1 baris). "
        "CSV boleh berisi **21 fitur lengkap** atau minimal **top-K fitur**."
    )

    tab1, tab2 = st.tabs(["Prediksi dari CSV", "Input Manual (1 baris)"])

    with tab1:
        df_session = st.session_state.get("df_uploaded", None)

        if df_session is not None:
            st.info("Menggunakan dataset dari **Upload Data** (tersimpan di session).")
            st.write("Preview dataset (session):")
            st.dataframe(df_session.head(20), use_container_width=True)

            if st.button("Prediksi dari dataset (session)", key="btn_pred_session"):
                try:
                    out = predict(df_session, artifacts, threshold)
                    st.session_state["df_pred_out"] = out  # simpan hasil prediksi
                    st.success("Prediksi selesai (dari dataset session).")
                except Exception as e:
                    st.error(f"Gagal prediksi: {e}")

        st.markdown("---")
        st.caption("Opsional: upload CSV lain khusus untuk prediksi (jika ingin mengganti dataset).")

        up_pred = st.file_uploader(
            "Upload file CSV untuk prediksi (opsional)",
            type=["csv"],
            key="pred_csv"
        )
        if up_pred is not None:
            df_pred = read_csv_file(up_pred)
            st.session_state["df_pred_input"] = df_pred
            st.write("Preview CSV prediksi:")
            st.dataframe(df_pred.head(20), use_container_width=True)

            if st.button("Prediksi dari CSV upload", key="btn_pred_upload"):
                try:
                    out = predict(df_pred, artifacts, threshold)
                    st.session_state["df_pred_out"] = out
                    st.success("Prediksi selesai (dari CSV upload).")
                except Exception as e:
                    st.error(f"Gagal prediksi: {e}")

        if "df_pred_out" in st.session_state:
            st.write("Hasil prediksi:")
            out = st.session_state["df_pred_out"]
            st.dataframe(out.head(50), use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download hasil (CSV)",
                data=csv_bytes,
                file_name="prediksi_churn.csv",
                mime="text/csv",
                key="dl_pred"
            )
        else:
            st.warning("Belum ada hasil prediksi. Klik tombol prediksi di atas.")

    with tab2:
        st.caption("Input manual default meminta **top-K fitur** sesuai model.")
        with st.form("manual_form"):
            cols = st.columns(2)
            values = {}
            for i, f in enumerate(selected_features):
                col = cols[i % 2]
                values[f] = col.number_input(f, value=0.0, step=1.0)
            submitted = st.form_submit_button("Prediksi 1 baris")

        if submitted:
            df_one = pd.DataFrame([values])
            try:
                out = predict(df_one, artifacts, threshold)
                st.write("Hasil:")
                st.dataframe(out, use_container_width=True)
            except Exception as e:
                st.error(f"Gagal prediksi: {e}")
