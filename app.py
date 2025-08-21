import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import zscore, iqr
from io import BytesIO
import base64

st.set_page_config(layout="wide", page_title="COVID-19 Dashboard", page_icon="🦠")

# -------------------------
# 1. CARGA DE DATOS
# -------------------------
@st.cache_data
def load_data():
    url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    df = pd.read_csv(url, parse_dates=["date"])
    return df

df = load_data()

# -------------------------
# 2. SIDEBAR FILTROS
# -------------------------
st.sidebar.header("Filtros")
paises = st.sidebar.multiselect("Selecciona países", df["location"].unique(), ["Peru", "Brazil"])
fecha_inicio, fecha_fin = st.sidebar.date_input(
    "Rango de fechas", [df["date"].min(), df["date"].max()]
)

df = df[(df["location"].isin(paises)) & (df["date"].between(fecha_inicio, fecha_fin))]

# -------------------------
# 3. KPIs PRINCIPALES
# -------------------------
st.title("🦠 Dashboard COVID-19")
st.subheader("Indicadores clave")

col1, col2, col3, col4 = st.columns(4)

for pais in paises:
    datos = df[df["location"] == pais]
    confirmados = datos["total_cases"].max()
    fallecidos = datos["total_deaths"].max()
    cfr = (fallecidos / confirmados * 100) if confirmados > 0 else 0
    tasa100k = (confirmados / datos["population"].max()) * 100000

    col1.metric(f"Confirmados - {pais}", f"{confirmados:,.0f}")
    col2.metric(f"Fallecidos - {pais}", f"{fallecidos:,.0f}")
    col3.metric(f"CFR - {pais}", f"{cfr:.2f}%")
    col4.metric(f"Tasa x100k - {pais}", f"{tasa100k:.2f}")

# -------------------------
# 4. TABS PRINCIPALES
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Visión General", "📈 Estadística Avanzada", 
    "⏳ Modelado Temporal", "🧩 Clustering & PCA", 
    "⚠️ Calidad de Datos"
])

# -------------------------
# TAB 1: VISIÓN GENERAL
# -------------------------
with tab1:
    st.subheader("Top-N Países Confirmados")
    top_confirmados = df.groupby("location")["total_cases"].max().sort_values(ascending=False).head(10)
    st.bar_chart(top_confirmados)

# -------------------------
# TAB 2: ESTADÍSTICA AVANZADA
# -------------------------
with tab2:
    st.subheader("Intervalos de confianza del CFR")
    for pais in paises:
        datos = df[df["location"] == pais]
        x = datos["total_deaths"].max()
        n = datos["total_cases"].max()
        if n > 0:
            p = x/n
            se = np.sqrt(p*(1-p)/n)
            ci_low, ci_high = p - 1.96*se, p + 1.96*se
            st.write(f"**{pais}**: CFR={p:.3f} ({ci_low:.3f}-{ci_high:.3f})")

    st.subheader("Test de hipótesis entre dos países (CFR)")
    if len(paises) == 2:
        x1 = df[df["location"] == paises[0]]["total_deaths"].max()
        n1 = df[df["location"] == paises[0]]["total_cases"].max()
        x2 = df[df["location"] == paises[1]]["total_deaths"].max()
        n2 = df[df["location"] == paises[1]]["total_cases"].max()
        stat, pval = proportions_ztest([x1, x2], [n1, n2])
        st.write(f"p-value={pval:.4f} → {'Diferencia significativa' if pval<0.05 else 'No significativa'}")

    st.subheader("Detección de outliers (Z-score)")
    casos = df.groupby("location")["total_cases"].max()
    zscores = zscore(casos)
    outliers = casos[np.abs(zscores) > 3]
    st.write("Países atípicos:", outliers.index.tolist())

    st.subheader("Gráfico de control (3σ) de muertes diarias")
    daily_deaths = df.groupby("date")["new_deaths"].sum()
    mean, std = daily_deaths.mean(), daily_deaths.std()
    fig, ax = plt.subplots()
    ax.plot(daily_deaths, label="Muertes diarias")
    ax.axhline(mean, color="green", linestyle="--")
    ax.axhline(mean + 3*std, color="red", linestyle="--")
    ax.axhline(mean - 3*std, color="red", linestyle="--")
    st.pyplot(fig)

# -------------------------
# TAB 3: MODELO TEMPORAL
# -------------------------
with tab3:
    st.subheader("Series temporales con suavizado 7d")
    for pais in paises:
        serie = df[df["location"] == pais].set_index("date")["new_cases"].rolling(7).mean()
        st.line_chart(serie, height=300)

    st.subheader("Pronóstico SARIMA (14 días)")
    pais = paises[0]  # solo primer país
    serie = df[df["location"] == pais].set_index("date")["new_cases"].dropna()
    modelo = SARIMAX(serie, order=(1,1,1), seasonal_order=(0,1,1,7))
    result = modelo.fit(disp=False)
    forecast = result.get_forecast(steps=14)
    pred_ci = forecast.conf_int()
    fig, ax = plt.subplots()
    serie.plot(ax=ax, label="Observado")
    forecast.predicted_mean.plot(ax=ax, label="Pronóstico")
    ax.fill_between(pred_ci.index, pred_ci.iloc[:,0], pred_ci.iloc[:,1], color='gray', alpha=0.3)
    ax.legend()
    st.pyplot(fig)

# -------------------------
# TAB 4: CLUSTERING Y PCA
# -------------------------
with tab4:
    st.subheader("K-means clustering")
    resumen = df.groupby("location").agg({
        "total_cases":"max", "total_deaths":"max", "population":"max"
    }).dropna()
    resumen["CFR"] = resumen["total_deaths"]/resumen["total_cases"]
    resumen["rate100k"] = resumen["total_cases"]/resumen["population"]*100000

    X = resumen[["CFR","rate100k"]].fillna(0)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
    resumen["cluster"] = kmeans.labels_

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    resumen["PC1"], resumen["PC2"] = pcs[:,0], pcs[:,1]

    fig, ax = plt.subplots()
    sns.scatterplot(data=resumen, x="PC1", y="PC2", hue="cluster", ax=ax, palette="Set2")
    st.pyplot(fig)

    st.write("Interpretación: cada clúster agrupa países con perfiles similares de CFR y tasas de contagio.")

# -------------------------
# TAB 5: CALIDAD DE DATOS
# -------------------------
with tab5:
    st.subheader("Valores nulos por columna")
    st.write(df.isna().sum())
