import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from scipy import stats

st.set_page_config(page_title="COVID-19 Viz – Pregunta 2", layout="wide")

GITHUB_BASE = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports"

@st.cache_data(show_spinner=False)
def load_daily_report(yyyy_mm_dd: str):
    yyyy, mm, dd = yyyy_mm_dd.split("-")
    url = f"{GITHUB_BASE}/{mm}-{dd}-{yyyy}.csv"
    df = pd.read_csv(url)
    # normalizar nombres por si varían
    lower = {c.lower(): c for c in df.columns}
    cols = {
        "country": lower.get("country_region", "Country_Region"),
        "province": lower.get("province_state", "Province_State"),
        "confirmed": lower.get("confirmed", "Confirmed"),
        "deaths": lower.get("deaths", "Deaths"),
        "recovered": lower.get("recovered", "Recovered") if "recovered" in lower else None,
        "active": lower.get("active", "Active") if "active" in lower else None,
        "population": lower.get("population", None),  # no siempre disponible
    }
    return df, url, cols

st.sidebar.title("Opciones")
fecha = st.sidebar.date_input("Fecha del reporte (JHU CSSE)", value=pd.to_datetime("2022-09-09"))
fecha_str = pd.to_datetime(fecha).strftime("%Y-%m-%d")
df, source_url, cols = load_daily_report(fecha_str)
st.sidebar.caption(f"Fuente: {source_url}")

st.title("Exploración COVID-19 – Versión Streamlit (Preg2)")
st.caption("Adaptación fiel del script original: mostrar/ocultar filas/columnas y varios gráficos (líneas, barras, sectores, histograma y boxplot).")

# ———————————————————————————————————————————————
# a) Mostrar filas
# ———————————————————————————————————————————————
st.header("a) Mostrar filas")
mostrar_todas = st.checkbox("Mostrar todas las filas", value=False)
if mostrar_todas:
    st.dataframe(df, use_container_width=True)
else:
    st.dataframe(df.head(25), use_container_width=True)

# ———————————————————————————————————————————————
# b) Mostrar columnas
# ———————————————————————————————————————————————
st.header("b) Mostrar columnas")
with st.expander("Vista de columnas"):
    st.write(list(df.columns))

st.caption("Usa el scroll horizontal de la tabla para ver todas las columnas en pantalla.")

# ———————————————————————————————————————————————
# c) Línea de fallecidos vs Confirmed/Recovered/Active por país
# ———————————————————————————————————————————————
st.header("c) Gráfica de líneas por país (muertes > 2500)")
C, D = cols["confirmed"], cols["deaths"]
R, A = cols["recovered"], cols["active"]

metrics = [m for m in [C, D, R, A] if m and m in df.columns]
base = df[[cols["country"]] + metrics].copy()
base = base.rename(columns={cols["country"]: "Country_Region"})

filtrado = base.loc[base[D] > 2500]
agr = filtrado.groupby("Country_Region").sum(numeric_only=True)
orden = agr.sort_values(D)

if not orden.empty:
    st.line_chart(orden[[c for c in [C, R, A] if c in orden.columns]])

# ———————————————————————————————————————————————
# d) Barras de fallecidos de estados de EE.UU.
# ———————————————————————————————————————————————
st.header("d) Barras: fallecidos por estado de EE.UU.")
country_col = cols["country"]
prov_col = cols["province"]

dfu = df[df[country_col] == "US"]
if len(dfu) == 0:
    st.info("Para esta fecha no hay registros con Country_Region='US'.")
else:
    agg_us = dfu.groupby(prov_col)[D].sum(numeric_only=True).sort_values(ascending=False)
    top_n = st.slider("Top estados por fallecidos", 5, 50, 20)
    st.bar_chart(agg_us.head(top_n))

# ———————————————————————————————————————————————
# e) Gráfica de sectores (simulada con barra)
# ———————————————————————————————————————————————
st.header("e) Gráfica de sectores (simulada)")
lista_paises = ["Colombia", "Chile", "Peru", "Argentina", "Mexico"]
sel = st.multiselect("Países", sorted(df[country_col].unique().tolist()), default=lista_paises)
agg_latam = df[df[country_col].isin(sel)].groupby(country_col)[D].sum(numeric_only=True)
if agg_latam.sum() > 0:
    st.write("Participación de fallecidos")
    st.dataframe(agg_latam)
    normalized = agg_latam / agg_latam.sum()
    st.bar_chart(normalized)
else:
    st.warning("Sin datos para los países seleccionados")

# ———————————————————————————————————————————————
# f) Histograma del total de fallecidos por país
# ———————————————————————————————————————————————
st.header("f) Histograma de fallecidos por país")
muertes_pais = df.groupby(country_col)[D].sum(numeric_only=True)
st.bar_chart(muertes_pais)

# ———————————————————————————————————————————————
# g) Boxplot simulado
# ———————————————————————————————————————————————
st.header("g) Boxplot (simulado)")
cols_box = [c for c in [C, D, R, A] if c and c in df.columns]
subset = df[cols_box].fillna(0)
subset_plot = subset.head(25)
st.write("Resumen estadístico (simulación de boxplot):")
st.dataframe(subset_plot.describe().T)

# ———————————————————————————————————————————————
# PARTE 2: Estadística descriptiva y avanzada
# ———————————————————————————————————————————————
st.header("PARTE 2 – Estadística descriptiva y avanzada")

# 1. Métricas clave
st.subheader("1. Métricas clave por país")
agg = df.groupby(country_col).sum(numeric_only=True)[[C, D]]
agg["CFR"] = agg[D] / agg[C]
agg["Rate_per_100k"] = np.where("Population" in df.columns, (agg[D] / df.groupby(country_col)["Population"].sum())*100000, np.nan)
st.dataframe(agg.head(20))

# 2. Intervalos de confianza para CFR
st.subheader("2. Intervalos de confianza para CFR")
pais_ic = st.selectbox("Seleccionar país", agg.index.tolist())
conf = 0.95
n = agg.loc[pais_ic, C]
x = agg.loc[pais_ic, D]
if n > 0:
    p_hat = x/n
    se = np.sqrt(p_hat*(1-p_hat)/n)
    z = stats.norm.ppf(1-(1-conf)/2)
    ic_low, ic_high = p_hat - z*se, p_hat + z*se
    st.write(f"CFR de {pais_ic}: {p_hat:.4f} (IC {conf*100:.0f}%: {ic_low:.4f} – {ic_high:.4f})")
else:
    st.warning("No hay suficientes casos para calcular IC.")

# 3. Test de hipótesis de proporciones
st.subheader("3. Test de hipótesis: comparación de CFR entre dos países")
pais1 = st.selectbox("País 1", agg.index.tolist(), index=0)
pais2 = st.selectbox("País 2", agg.index.tolist(), index=1)
n1, x1 = agg.loc[pais1, C], agg.loc[pais1, D]
n2, x2 = agg.loc[pais2, C], agg.loc[pais2, D]
if n1>0 and n2>0:
    p1, p2 = x1/n1, x2/n2
    p_pool = (x1+x2)/(n1+n2)
    se = np.sqrt(p_pool*(1-p_pool)*(1/n1+1/n2))
    z_stat = (p1-p2)/se
    p_val = 2*(1-stats.norm.cdf(abs(z_stat)))
    st.write(f"CFR {pais1}: {p1:.4f}, CFR {pais2}: {p2:.4f}")
    st.write(f"Z = {z_stat:.3f}, p-value = {p_val:.4g}")
else:
    st.warning("No hay suficientes casos para comparar.")

# 4. Detección de outliers
st.subheader("4. Detección de outliers")
serie = agg[D]
z_scores = np.abs(stats.zscore(serie))
outliers = serie[z_scores > 3]
if not outliers.empty:
    st.write("Outliers detectados (Z-score > 3):")
    st.dataframe(outliers)
else:
    st.info("No se detectaron outliers.")

# 5. Gráfico de control de muertes diarias (3σ)
st.subheader("5. Gráfico de control (3σ) de muertes diarias")
muertes = df.groupby(country_col)[D].sum(numeric_only=True)
media, sigma = muertes.mean(), muertes.std()
lim_sup, lim_inf = media + 3*sigma, max(media - 3*sigma, 0)
st.line_chart(muertes)
st.write(f"Media: {media:.2f}, Límite inferior: {lim_inf:.2f}, Límite superior: {lim_sup:.2f}")
