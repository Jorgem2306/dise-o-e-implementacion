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
st.header("PARTE 2 – Estadística descriptiva y avanzada")   # Título principal de esta sección en la app

# 1. Métricas clave
st.subheader("1. Métricas clave por país")                  # Subtítulo para las métricas básicas
agg = df.groupby(country_col).sum(numeric_only=True)[[C, D]]  # Agrupar por país y sumar confirmados y muertes
agg["CFR"] = agg[D] / agg[C]                               # Calcular CFR (muertes/confirmados)

# Si existe la columna Incident_Rate en el dataset
if "Incident_Rate" in df.columns:
    # Calcular la tasa de incidencia promedio por país (casos por 100k)
    incident = df.groupby(country_col)["Incident_Rate"].mean(numeric_only=True)
    # Estimar población a partir de confirmed e incident_rate
    poblacion_est = (agg[C] * 100000 / incident).replace([np.inf, -np.inf], np.nan)
    # Calcular muertes por 100k habitantes
    agg["Deaths_per_100k"] = (agg[D] / poblacion_est) * 100000
else:
    agg["Deaths_per_100k"] = np.nan                        # Si no hay incident_rate, asignar NaN
    st.info("⚠️ No hay columna 'Incident_Rate' en este reporte, no se puede estimar la tasa por 100k.")

st.dataframe(agg.head(50))                                 # Mostrar primeras 20 filas con métricas

# 2. Intervalos de confianza para CFR
st.subheader("2. Intervalos de confianza para CFR")         # Subtítulo
pais_ic = st.selectbox("Seleccionar país", agg.index.tolist())  # Dropdown para elegir país
conf = 0.95                                                # Nivel de confianza 95%
n = agg.loc[pais_ic, C]                                    # Confirmados del país elegido
x = agg.loc[pais_ic, D]                                    # Fallecidos del país elegido
if n > 0:                                                  # Solo calcular si hay casos
    p_hat = x/n                                            # Proporción de muertes (CFR)
    se = np.sqrt(p_hat*(1-p_hat)/n)                        # Error estándar de la proporción
    z = stats.norm.ppf(1-(1-conf)/2)                       # Valor z para el nivel de confianza
    ic_low, ic_high = p_hat - z*se, p_hat + z*se           # Intervalo de confianza
    st.write(f"CFR de {pais_ic}: {p_hat:.4f} (IC {conf*100:.0f}%: {ic_low:.4f} – {ic_high:.4f})")
else:
    st.warning("No hay suficientes casos para calcular IC.")

# 3. Test de hipótesis de proporciones
st.subheader("3. Test de hipótesis: comparación de CFR entre dos países")
pais1 = st.selectbox("País 1", agg.index.tolist(), index=0)  # Selección de país 1
pais2 = st.selectbox("País 2", agg.index.tolist(), index=1)  # Selección de país 2
n1, x1 = agg.loc[pais1, C], agg.loc[pais1, D]                # Confirmados y muertes país 1
n2, x2 = agg.loc[pais2, C], agg.loc[pais2, D]                # Confirmados y muertes país 2
if n1>0 and n2>0:
    p1, p2 = x1/n1, x2/n2                                  # CFR de cada país
    p_pool = (x1+x2)/(n1+n2)                               # Proporción combinada
    se = np.sqrt(p_pool*(1-p_pool)*(1/n1+1/n2))            # Error estándar combinado
    z_stat = (p1-p2)/se                                    # Estadístico Z
    p_val = 2*(1-stats.norm.cdf(abs(z_stat)))              # p-value bilateral
    st.write(f"CFR {pais1}: {p1:.4f}, CFR {pais2}: {p2:.4f}")
    st.write(f"Z = {z_stat:.3f}, p-value = {p_val:.4g}")
else:
    st.warning("No hay suficientes casos para comparar.")

# 4. Detección de outliers
st.subheader("4. Detección de outliers")                    # Subtítulo
serie = agg[D]                                             # Serie de muertes totales por país
z_scores = np.abs(stats.zscore(serie))                     # Calcular Z-score de cada país
outliers = serie[z_scores > 3]                             # Detectar valores fuera de 3σ
if not outliers.empty:
    st.write("Outliers detectados (Z-score > 3):")          # Mostrar si hay outliers
    st.dataframe(outliers)
else:
    st.info("No se detectaron outliers.")

# 5. Gráfico de control de muertes diarias (3σ)
st.subheader("5. Gráfico de control (3σ) de muertes diarias")
muertes = df.groupby(country_col)[D].sum(numeric_only=True)  # Muertes totales por país
media, sigma = muertes.mean(), muertes.std()                 # Media y desviación estándar
lim_sup, lim_inf = media + 3*sigma, max(media - 3*sigma, 0)  # Límites superior e inferior (3σ)
st.line_chart(muertes)                                       # Gráfico de línea de muertes
st.write(f"Media: {media:.2f}, Límite inferior: {lim_inf:.2f}, Límite superior: {lim_sup:.2f}")

# ———————————————————————————————————————————————
# PARTE 3: Modelado y proyecciones
# ———————————————————————————————————————————————
st.header("PARTE 3 – Modelado y proyecciones")

# 3.1. Series de tiempo por país con suavizado
st.subheader("3.1. Serie de tiempo con suavizado de 7 días")
pais_ts = st.selectbox("Seleccionar país para series de tiempo", df[country_col].unique())
df_pais = df[df[country_col] == pais_ts].groupby("Date")[[C, D]].sum().reset_index()
df_pais["C_suavizado"] = df_pais[C].rolling(7, min_periods=1).mean()
df_pais["D_suavizado"] = df_pais[D].rolling(7, min_periods=1).mean()

st.line_chart(df_pais.set_index("date")[["C_suavizado", "D_suavizado"]])

# 3.2. Modelo de pronóstico SARIMA
st.subheader("3.2. Modelo de pronóstico (SARIMA)")
from statsmodels.tsa.statespace.sarimax import SARIMAX

serie_casos = df_pais.set_index("Date")[C].asfreq("D").fillna(0)

try:
    # Ajustar modelo SARIMA simple (puedes tunear p,d,q)
    modelo = SARIMAX(serie_casos, order=(1,1,1), seasonal_order=(0,1,1,7))
    resultado = modelo.fit(disp=False)

    # Proyectar 14 días
    forecast = resultado.get_forecast(steps=14)
    pred_mean = forecast.predicted_mean
    pred_ci = forecast.conf_int()

    # Mostrar proyección
    st.line_chart(pd.concat([serie_casos, pred_mean], axis=1).rename(
        columns={C: "Histórico", "predicted_mean": "Pronóstico"}
    ))

    st.write("Intervalos de confianza del forecast:")
    st.dataframe(pred_ci)

except Exception as e:
    st.error(f"No se pudo ajustar el modelo SARIMA: {e}")

# 3.3. Validación con backtesting
st.subheader("3.3. Validación (Backtesting con MAE/MAPE)")
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Usamos últimos 14 días como validación
train = serie_casos.iloc[:-14]
test = serie_casos.iloc[-14:]

try:
    modelo_bt = SARIMAX(train, order=(1,1,1), seasonal_order=(0,1,1,7))
    resultado_bt = modelo_bt.fit(disp=False)
    pred_bt = resultado_bt.get_forecast(steps=14).predicted_mean

    mae = mean_absolute_error(test, pred_bt)
    mape = mean_absolute_percentage_error(test, pred_bt)

    st.write(f"MAE: {mae:.2f}, MAPE: {mape:.2%}")

except Exception as e:
    st.error(f"No se pudo validar el modelo: {e}")

# 3.4. Bandas de confianza en forecast
st.subheader("3.4. Forecast con bandas de confianza")
try:
    forecast = resultado.get_forecast(steps=14)
    pred_mean = forecast.predicted_mean
    pred_ci = forecast.conf_int()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,4))
    serie_casos.plot(ax=ax, label="Histórico")
    pred_mean.plot(ax=ax, label="Pronóstico", color="red")
    ax.fill_between(pred_ci.index, pred_ci.iloc[:,0], pred_ci.iloc[:,1], 
                    color="pink", alpha=0.3, label="IC 95%")
    ax.legend()
    st.pyplot(fig)
except:
    st.warning("No se pudieron graficar bandas de confianza.")
