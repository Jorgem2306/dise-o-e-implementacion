import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="COVID-19 Viz – Pregunta 2", layout="wide")

GITHUB_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/09-09-2022.csv"

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(GITHUB_URL)
    df["Last_Update"] = pd.to_datetime(df["Last_Update"])
    return df

df = load_data()

# Parte 1 y 2: tu código original
st.sidebar.title("Opciones")
st.sidebar.caption(f"Fuente: {GITHUB_URL}")

st.title("Exploración COVID-19 – Versión Streamlit (Preg2)")
st.caption("Adaptación fiel del script original: mostrar/ocultar filas/columnas y varios gráficos (líneas, barras, sectores, histograma y boxplot).")

# a) Mostrar filas
st.header("a) Mostrar filas")
mostrar_todas = st.checkbox("Mostrar todas las filas", value=False)
if mostrar_todas:
    st.dataframe(df, use_container_width=True)
else:
    st.dataframe(df.head(25), use_container_width=True)

# b) Mostrar columnas
st.header("b) Mostrar columnas")
with st.expander("Vista de columnas"):
    st.write(list(df.columns))
st.caption("Usa el scroll horizontal de la tabla para ver todas las columnas en pantalla.")

# c) Gráfica de líneas (muertes > 2500)
st.header("c) Gráfica de líneas por país (muertes > 2500)")
lower = {c.lower(): c for c in df.columns}
C = lower.get("confirmed", "Confirmed")
D = lower.get("deaths", "Deaths")
R = lower.get("recovered", "Recovered")
A = lower.get("active", "Active")

metrics = [m for m in [C, D, R, A] if m in df.columns]
base = df[[lower.get("country_region", "Country_Region")] + metrics].copy()
base = base.rename(columns={lower.get("country_region", "Country_Region"): "Country_Region"})
filtrado = base.loc[base[D] > 2500]
agr = filtrado.groupby("Country_Region").sum(numeric_only=True)
orden = agr.sort_values(D)
if not orden.empty:
    st.line_chart(orden[[c for c in [C, R, A] if c in orden.columns]])

# d) Barras: fallecidos por estado de EE.UU.
st.header("d) Barras: fallecidos por estado de EE.UU.")
country_col = lower.get("country_region", "Country_Region")
prov_col = lower.get("province_state", "Province_State")
dfu = df[df[country_col] == "US"]
if len(dfu) == 0:
    st.info("Para esta fecha no hay registros con Country_Region='US'.")
else:
    agg_us = dfu.groupby(prov_col)[D].sum(numeric_only=True).sort_values(ascending=False)
    top_n = st.slider("Top estados por fallecidos", 5, 50, 20)
    st.bar_chart(agg_us.head(top_n))

# e) Simulación de gráfico de sectores (barra)
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

# f) Histograma de fallecidos por país
st.header("f) Histograma de fallecidos por país")
muertes_pais = df.groupby(country_col)[D].sum(numeric_only=True)
st.bar_chart(muertes_pais)

# g) Boxplot simulado
st.header("g) Boxplot (simulado)")
cols_box = [c for c in [C, D, R, A] if c in df.columns]
subset = df[cols_box].fillna(0)
subset_plot = subset.head(25)
st.write("Resumen estadístico (simulación de boxplot):")
st.dataframe(subset_plot.describe().T)

# PARTE 2 – Estadística descriptiva y avanzada
st.header("PARTE 2 – Estadística descriptiva y avanzada")

# 1. Métricas clave por país
st.subheader("1. Métricas clave por país")
agg = df.groupby(country_col).sum(numeric_only=True)[[C, D]]
agg["CFR"] = agg[D] / agg[C]
if "Incident_Rate" in df.columns:
    incident = df.groupby(country_col)["Incident_Rate"].mean(numeric_only=True)
    poblacion_est = (agg[C] * 100000 / incident).replace([np.inf, -np.inf], np.nan)
    agg["Deaths_per_100k"] = (agg[D] / poblacion_est) * 100000
else:
    agg["Deaths_per_100k"] = np.nan
    st.info("⚠️ No hay columna 'Incident_Rate' en este reporte.")

st.dataframe(agg.head(50))

# 2. IC para CFR
st.subheader("2. Intervalos de confianza para CFR")
pais_ic = st.selectbox("Seleccionar país", agg.index.tolist())
conf = 0.95
n = agg.loc[pais_ic, C]
x = agg.loc[pais_ic, D]
if n > 0:
    p_hat = x / n
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    ic_low, ic_high = p_hat - z * se, p_hat + z * se
    st.write(f"CFR de {pais_ic}: {p_hat:.4f} (IC {conf*100:.0f}%: {ic_low:.4f} – {ic_high:.4f})")
else:
    st.warning("No hay suficientes casos para calcular IC.")

# 3. Test de hipótesis para CFR
st.subheader("3. Test de hipótesis: comparación de CFR entre dos países")
pais1 = st.selectbox("País 1", agg.index.tolist(), index=0)
pais2 = st.selectbox("País 2", agg.index.tolist(), index=1)
n1, x1 = agg.loc[pais1, C], agg.loc[pais1, D]
n2, x2 = agg.loc[pais2, C], agg.loc[pais2, D]
if n1 > 0 and n2 > 0:
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z_stat = (p1 - p2) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
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

# 5. Gráfico de control 3σ
st.subheader("5. Gráfico de control (3σ) de muertes diarias")
muertes = df.groupby(country_col)[D].sum(numeric_only=True)
media, sigma = muertes.mean(), muertes.std()
lim_sup, lim_inf = media + 3 * sigma, max(media - 3 * sigma, 0)
st.line_chart(muertes)
st.write(f"Media: {media:.2f}, Límite inferior: {lim_inf:.2f}, Límite superior: {lim_sup:.2f}")

# ——————————————————————
# PARTE 3.1 – Series de tiempo con suavizado 7 días
# ——————————————————————
st.header("PARTE 3 – Series de tiempo")
st.subheader("3.1 Series de tiempo por país (suavizado 7 días)")

# URLs de series temporales globales
url_confirmed = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
url_deaths = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

df_conf = pd.read_csv(url_confirmed)
df_deaths = pd.read_csv(url_deaths)

# Reorganizar: columnas de fechas → filas
df_conf = df_conf.drop(columns=["Province/State", "Lat", "Long"]).groupby("Country/Region").sum().T
df_deaths = df_deaths.drop(columns=["Province/State", "Lat", "Long"]).groupby("Country/Region").sum().T

# Convertir índices a fecha
df_conf.index = pd.to_datetime(df_conf.index)
df_deaths.index = pd.to_datetime(df_deaths.index)

# Selector país
paises_ts = sorted(df_conf.columns.tolist())
pais_ts = st.selectbox("Selecciona país", paises_ts, index=(paises_ts.index("Peru") if "Peru" in paises_ts else 0))

# Calcular casos/muertes diarios
df_pais = pd.DataFrame({
    "Confirmed": df_conf[pais_ts],
    "Deaths": df_deaths[pais_ts]
})
df_pais["NewConfirmed"] = df_pais["Confirmed"].diff().clip(lower=0).fillna(0)
df_pais["NewDeaths"] = df_pais["Deaths"].diff().clip(lower=0).fillna(0)

# Suavizado 7 días
df_pais["NewConfirmed_7d"] = df_pais["NewConfirmed"].rolling(7, min_periods=1).mean()
df_pais["NewDeaths_7d"] = df_pais["NewDeaths"].rolling(7, min_periods=1).mean()

# Gráficas
c1, c2 = st.columns(2)
with c1:
    st.write(f"{pais_ts} – Nuevos confirmados (diario vs 7d)")
    st.line_chart(df_pais[["NewConfirmed", "NewConfirmed_7d"]])
with c2:
    st.write(f"{pais_ts} – Nuevas muertes (diario vs 7d)")
    st.line_chart(df_pais[["NewDeaths", "NewDeaths_7d"]])

# ——————————————————————
# PARTE 3.2 – Pronóstico con SARIMA o ETS
# ——————————————————————
st.subheader("3.2 Pronóstico de casos y muertes a 14 días")

# Selector de modelo
modelo_opcion = st.radio("Selecciona el modelo de pronóstico:", ["SARIMA", "ETS"])

# Datos del país seleccionado en 3.1
serie_confirmados = df_pais["NewConfirmed"].fillna(0)
serie_muertes = df_pais["NewDeaths"].fillna(0)

horizonte = 14  # días a predecir

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def pronosticar(serie, modelo="SARIMA", pasos=14):
    if modelo == "SARIMA":
        # Modelo SARIMA simple (puedes ajustar los parámetros)
        try:
            mod = SARIMAX(serie, order=(1,1,1), seasonal_order=(1,1,1,7))
            res = mod.fit(disp=False)
            pred = res.forecast(steps=pasos)
        except:
            pred = pd.Series([None]*pasos, index=pd.date_range(serie.index[-1]+pd.Timedelta(days=1), periods=pasos))
    else:  # ETS
        try:
            mod = ExponentialSmoothing(serie, trend="add", seasonal=None)
            res = mod.fit()
            pred = res.forecast(pasos)
        except:
            pred = pd.Series([None]*pasos, index=pd.date_range(serie.index[-1]+pd.Timedelta(days=1), periods=pasos))
    return pred

# Pronósticos
pred_conf = pronosticar(serie_confirmados, modelo_opcion, horizonte)
pred_muertes = pronosticar(serie_muertes, modelo_opcion, horizonte)

# Combinar series reales + forecast
df_forecast = pd.DataFrame({
    "Real Confirmados": serie_confirmados,
    "Real Muertes": serie_muertes
})
df_forecast_pred = pd.DataFrame({
    "Pred Confirmados": pred_conf,
    "Pred Muertes": pred_muertes
})

# Gráficas
c1, c2 = st.columns(2)
with c1:
    st.write(f"{pais_ts} – Pronóstico de nuevos confirmados ({modelo_opcion})")
    st.line_chart(pd.concat([df_forecast["Real Confirmados"], df_forecast_pred["Pred Confirmados"]]))
with c2:
    st.write(f"{pais_ts} – Pronóstico de nuevas muertes ({modelo_opcion})")
    st.line_chart(pd.concat([df_forecast["Real Muertes"], df_forecast_pred["Pred Muertes"]]))
