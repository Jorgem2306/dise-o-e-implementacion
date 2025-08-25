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

# ============================
# PARTE 3.3 – Validación Backtesting
# ============================
st.subheader("3.3 Validación con Backtesting (MAE / MAPE)")

def backtest(serie, modelo, pasos=14, ventana=14):
    errores_mae = []
    errores_mape = []
    for i in range(ventana, len(serie) - pasos):
        train = serie[:i]
        test = serie[i:i+pasos]

        pred = forecast(train, modelo, pasos)

        if len(pred) == pasos:
            try:
                # Evitar problemas de MAPE con ceros
                test_safe = test.replace(0, 1e-6)
                errores_mae.append(mean_absolute_error(test, pred))
                errores_mape.append(mean_absolute_percentage_error(test_safe, pred))
            except Exception as e:
                continue

    if len(errores_mae) > 0:
        return np.mean(errores_mae), np.mean(errores_mape)
    else:
        return np.nan, np.nan

mae_conf, mape_conf = backtest(serie_confirmados, modelo)
mae_muertes, mape_muertes = backtest(serie_muertes, modelo)

st.write(f"{pais_ts} – Validación {modelo}")
st.write(f"Nuevos confirmados → MAE: {mae_conf:.2f}, MAPE: {mape_conf:.2%}" if not np.isnan(mae_conf) else "⚠️ No hay suficientes datos para confirmados")
st.write(f"Nuevas muertes → MAE: {mae_muertes:.2f}, MAPE: {mape_muertes:.2%}" if not np.isnan(mae_muertes) else "⚠️ No hay suficientes datos para muertes")

    st.warning("⚠️ No hay suficientes datos confiables para validar muertes.")
else:
    st.write(f"- Nuevas muertes → MAE: {mae_muertes:.2f}, MAPE: {mape_muertes:.2%}")
