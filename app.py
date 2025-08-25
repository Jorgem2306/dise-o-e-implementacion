import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# ============================
# CONFIGURACIÓN
# ============================
date_col = "Last_Update"
country_col = "Country_Region"
C, D = "Confirmed", "Deaths"

# Cargar datos desde URL
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/09-09-2022.csv"
df = pd.read_csv(url)

# Convertir fecha
df[date_col] = pd.to_datetime(df[date_col])

# ============================
# PARTE 3.1 – Series de tiempo
# ============================
st.header("PARTE 3 – Series de tiempo y pronósticos")
st.subheader("3.1 Series de tiempo por país (suavizado 7 días)")

# Selector país
paises_ts = sorted(df[country_col].unique().tolist())
pais_ts = st.selectbox("Selecciona país", paises_ts, index=(paises_ts.index("Peru") if "Peru" in paises_ts else 0))

df_pais = df[df[country_col] == pais_ts].copy().sort_values(date_col)
df_pais["NewConfirmed"] = df_pais[C].diff().clip(lower=0).fillna(0)
df_pais["NewDeaths"] = df_pais[D].diff().clip(lower=0).fillna(0)
df_pais["NewConfirmed_7d"] = df_pais["NewConfirmed"].rolling(7, min_periods=1).mean()
df_pais["NewDeaths_7d"] = df_pais["NewDeaths"].rolling(7, min_periods=1).mean()

# Gráficas
c1, c2 = st.columns(2)
with c1:
    st.write(f"{pais_ts} – Nuevos confirmados (diario vs 7d)")
    st.line_chart(df_pais.set_index(date_col)[["NewConfirmed", "NewConfirmed_7d"]])
with c2:
    st.write(f"{pais_ts} – Nuevas muertes (diario vs 7d)")
    st.line_chart(df_pais.set_index(date_col)[["NewDeaths", "NewDeaths_7d"]])

# ============================
# PARTE 3.2 – Pronóstico
# ============================
st.subheader("3.2 Pronóstico de casos y muertes a 14 días")

modelo = st.radio("Selecciona modelo", ["SARIMA", "ETS"])

def forecast(serie, modelo, pasos=14):
    try:
        if modelo == "SARIMA":
            mod = SARIMAX(serie, order=(1,1,1), seasonal_order=(1,1,1,7))
            res = mod.fit(disp=False)
            return res.get_forecast(steps=pasos).predicted_mean
        else:  # ETS
            mod = ExponentialSmoothing(serie, trend="add", seasonal="add", seasonal_periods=7)
            res = mod.fit()
            return res.forecast(pasos)
    except Exception as e:
        st.warning(f"⚠️ Error al generar pronóstico: {e}")
        return pd.Series()

# Series
serie_confirmados = df_pais["NewConfirmed_7d"].dropna()
serie_muertes = df_pais["NewDeaths_7d"].dropna()

forecast_conf = forecast(serie_confirmados, modelo)
forecast_muertes = forecast(serie_muertes, modelo)

# Mostrar gráficos de pronóstico
fig1, ax1 = plt.subplots()
serie_confirmados.plot(ax=ax1, label="Histórico")
forecast_conf.plot(ax=ax1, label="Pronóstico", style="--")
ax1.set_title(f"{pais_ts} – Pronóstico Confirmados ({modelo})")
ax1.legend()
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
serie_muertes.plot(ax=ax2, label="Histórico")
forecast_muertes.plot(ax=ax2, label="Pronóstico", style="--")
ax2.set_title(f"{pais_ts} – Pronóstico Muertes ({modelo})")
ax2.legend()
st.pyplot(fig2)

# ——————————————————————
# PARTE 3.4 – Forecast con intervalos de confianza
# ——————————————————————
st.subheader("3.4 Pronóstico con bandas de confianza (14 días)")

modelo_forecast = st.radio("Modelo para forecast:", ["SARIMA", "ETS"], index=0)

serie_casos = df_pais["NewConfirmed"]
serie_muertes = df_pais["NewDeaths"]
pasos = 14

def forecast_conf(serie, modelo, pasos=14):
    if modelo == "SARIMA":
        modelo_fit = SARIMAX(serie, order=(1,1,1), seasonal_order=(1,1,1,7)).fit(disp=False)
    else:
        modelo_fit = ExponentialSmoothing(serie, trend="add", seasonal=None).fit()

    pred = modelo_fit.get_forecast(steps=pasos)
    media = pred.predicted_mean
    ci = pred.conf_int(alpha=0.05)
    return media, ci

# Casos
media_casos, ci_casos = forecast_conf(serie_casos, modelo_forecast, pasos)
fig, ax = plt.subplots()
ax.plot(serie_casos.index, serie_casos, label="Histórico")
ax.plot(media_casos.index, media_casos, color="red", label="Forecast")
ax.fill_between(media_casos.index, ci_casos.iloc[:,0], ci_casos.iloc[:,1], 
                color="pink", alpha=0.3, label="IC 95%")
ax.legend()
ax.set_title(f"{pais_ts} – Nuevos casos (Forecast con IC)")
st.pyplot(fig)

# Muertes
media_muertes, ci_muertes = forecast_conf(serie_muertes, modelo_forecast, pasos)
fig, ax = plt.subplots()
ax.plot(serie_muertes.index, serie_muertes, label="Histórico")
ax.plot(media_muertes.index, media_muertes, color="red", label="Forecast")
ax.fill_between(media_muertes.index, ci_muertes.iloc[:,0], ci_muertes.iloc[:,1], 
                color="pink", alpha=0.3, label="IC 95%")
ax.legend()
ax.set_title(f"{pais_ts} – Nuevas muertes (Forecast con IC)")
st.pyplot(fig)
