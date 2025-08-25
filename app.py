import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from scipy import stats
from datetime import datetime, date, timedelta
import statsmodels.api as sm
import matplotlib.pyplot as plt

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
        "incident_rate": lower.get("incident_rate", "Incident_Rate") if "incident_rate" in lower else None,
    }
    return df, url, cols

# === NUEVO ===
@st.cache_data(show_spinner=True)
def load_reports_range(start_date: str, end_date: str):
    """Descarga y concatena reportes diarios entre dos fechas (ambas incluidas).
    Devuelve df con columna 'Report_Date' (datetime). Ignora días faltantes.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    frames = []
    last_cols = None

    d = start
    while d <= end:
        yyyy_mm_dd = d.strftime("%Y-%m-%d")
        try:
            df_d, _, cols_d = load_daily_report(yyyy_mm_dd)
            df_d = df_d.copy()
            df_d["Report_Date"] = pd.to_datetime(yyyy_mm_dd)
            frames.append((df_d, cols_d))
            last_cols = cols_d
        except Exception:
            # día no disponible
            pass
        d += pd.Timedelta(days=1)

    if not frames:
        return pd.DataFrame(), None

    # usar las columnas del último archivo cargado
    cols = frames[-1][1]
    df_all = pd.concat([f[0] for f in frames], ignore_index=True)

    # asegurar tipos y columnas clave
    C = cols["confirmed"]; D = cols["deaths"]
    country_col = cols["country"]

    # filtrar columnas mínimas necesarias
    keep = [c for c in [country_col, C, D, "Report_Date"] if c in df_all.columns]
    df_all = df_all[keep].copy()
    return df_all, cols

st.sidebar.title("Opciones")
fecha = st.sidebar.date_input("Fecha del reporte (JHU CSSE)", value=pd.to_datetime("2022-09-09"))
fecha_str = pd.to_datetime(fecha).strftime("%Y-%m-%d")
df, source_url, cols = load_daily_report(fecha_str)
st.sidebar.caption(f"Fuente (día): {source_url}")

st.title("Exploración COVID-19 – Versión Streamlit (Preg2)")
st.caption("Adaptación fiel del script original: mostrar/ocultar filas/columnas y varios gráficos (líneas, barras, sectores, histograma y boxplot).")

# ———————————————————————————————————————————————
# a) Mostrar todas las filas del dataset, luego volver al estado inicial
# ———————————————————————————————————————————————
st.header("a) Mostrar filas")
mostrar_todas = st.checkbox("Mostrar todas las filas", value=False)
if mostrar_todas:
    st.dataframe(df, use_container_width=True)
else:
    st.dataframe(df.head(25), use_container_width=True)

# ———————————————————————————————————————————————
# b) Mostrar todas las columnas del dataset
# ———————————————————————————————————————————————
st.header("b) Mostrar columnas")
with st.expander("Vista de columnas"):
    st.write(list(df.columns))

st.caption("Usa el scroll horizontal de la tabla para ver todas las columnas en pantalla.")

# ———————————————————————————————————————————————
# c) Línea del total de fallecidos (>2500) vs Confirmed/Recovered/Active por país
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
# d) Barras de fallecidos de estados de Estados Unidos
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
# e) Gráfica de sectores (simulada con barra si no hay pie nativo)
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
# f) Histograma del total de fallecidos por país (simulado con bar_chart)
# ———————————————————————————————————————————————
st.header("f) Histograma de fallecidos por país")
muertes_pais = df.groupby(country_col)[D].sum(numeric_only=True)
st.bar_chart(muertes_pais)

# ———————————————————————————————————————————————
# g) Boxplot de Confirmed, Deaths, Recovered, Active (simulado con box_chart)
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

if "Incident_Rate" in df.columns:
    incident = df.groupby(country_col)["Incident_Rate"].mean(numeric_only=True)
    poblacion_est = (agg[C] * 100000 / incident).replace([np.inf, -np.inf], np.nan)
    agg["Deaths_per_100k"] = (agg[D] / poblacion_est) * 100000
else:
    agg["Deaths_per_100k"] = np.nan
    st.info("⚠️ No hay columna 'Incident_Rate' en este reporte, no se puede estimar la tasa por 100k.")

st.dataframe(agg.head(50))

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

# ———————————————————————————————————————————————
# PARTE 3: Series de tiempo y pronósticos (NUEVO)
# ———————————————————————————————————————————————
st.header("PARTE 3 – Series de tiempo y pronósticos")

st.markdown("**Descarga histórica** para construir series diarias por país (confirmados y muertes).")
colA, colB = st.columns(2)
with colA:
    start_hist = st.date_input("Fecha inicial (histórico)", value=pd.to_datetime("2020-03-01"))
with colB:
    end_hist = st.date_input("Fecha final (histórico)", value=pd.to_datetime(fecha_str))

df_all, cols_hist = load_reports_range(pd.to_datetime(start_hist).strftime("%Y-%m-%d"),
                                       pd.to_datetime(end_hist).strftime("%Y-%m-%d"))

if df_all.empty:
    st.warning("No se pudo construir el histórico para el rango dado.")
else:
    ccol = cols_hist["country"]; Cc = cols_hist["confirmed"]; Dd = cols_hist["deaths"]

    # Agregación por día y país (valores acumulados)
    daily = (
        df_all.groupby(["Report_Date", ccol])[[Cc, Dd]]
        .sum(numeric_only=True)
        .sort_index()
        .rename(columns={Cc: "ConfirmedCum", Dd: "DeathsCum"})
        .reset_index()
    )

    # Selector de país (usar los disponibles en el rango)
    paises_ts = sorted(daily[ccol].unique().tolist())
    pais_ts = st.selectbox("País para series y pronóstico", paises_ts, index=(paises_ts.index("Peru") if "Peru" in paises_ts else 0))

    df_pais = daily[daily[ccol] == pais_ts].copy()
    df_pais = df_pais.set_index("Report_Date").sort_index()

    # Derivar series diarias (diferencia de acumulados)
    for col in ["ConfirmedCum", "DeathsCum"]:
        df_pais[col] = df_pais[col].clip(lower=0).fillna(0)

    df_pais["NewConfirmed"] = df_pais["ConfirmedCum"].diff().clip(lower=0).fillna(0)
    df_pais["NewDeaths"] = df_pais["DeathsCum"].diff().clip(lower=0).fillna(0)

    # 3.1 Suavizado 7 días
    df_pais["NewConfirmed_7d"] = df_pais["NewConfirmed"].rolling(7, min_periods=1).mean()
    df_pais["NewDeaths_7d"] = df_pais["NewDeaths"].rolling(7, min_periods=1).mean()

    st.subheader("3.1 Series de tiempo (diarias y suavizadas 7d)")
    st.markdown(f"**{pais_ts}** – Nuevos confirmados y muertes (promedio móvil 7 días).")
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(df_pais[["NewConfirmed", "NewConfirmed_7d"]])
    with col2:
        st.line_chart(df_pais[["NewDeaths", "NewDeaths_7d"]])

    # ——— utilidades de modelado ———
    def _clean_series(s: pd.Series):
        s = s.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # algunas observaciones pueden ser 0 por cortes de fin de semana; mantener, pero evitar todo cero
        if s.sum() == 0:
            return None
        return s

    def fit_sarimax_and_forecast(y: pd.Series, steps: int = 14):
        """SARIMAX con estacionalidad semanal; retorna pred, conf_int y modelo."""
        # orden básico razonable para conteos diarios
        mod = sm.tsa.statespace.SARIMAX(
            y,
            order=(1,1,1),
            seasonal_order=(1,1,1,7),
            enforce_stationarity=False,
            enforce_invertibility=False,
            trend="n"
        )
        res = mod.fit(disp=False)
        fc = res.get_forecast(steps=steps)
        mean = fc.predicted_mean
        conf = fc.conf_int(alpha=0.05)  # 95%
        return res, mean, conf

    def backtest_mae_mape(y: pd.Series, window_days: int = 21):
        """Backtesting 1-step-ahead con refit diario en los últimos `window_days`."""
        y = y.copy()
        if len(y) <= window_days + 7:
            return None, None, None  # muy corta

        y_train_full = y.iloc[:-window_days]
        preds = []
        actuals = []

        # caminata sobre los últimos window_days puntos
        for i in range(window_days):
            # datos hasta el día t-1
            y_train = y.iloc[:-(window_days - i)]
            try:
                _, mean1, _ = fit_sarimax_and_forecast(y_train, steps=1)
                preds.append(float(mean1.iloc[0]))
                actuals.append(float(y.iloc[len(y_train)]))
            except Exception:
                # si algo falla, usar pred naive (último valor)
                preds.append(float(y_train.iloc[-1]))
                actuals.append(float(y.iloc[len(y_train)]))

        preds = np.array(preds)
        actuals = np.array(actuals)

        mae = np.mean(np.abs(actuals - preds))
        mask = actuals > 0
        mape = (np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask])) * 100) if mask.any() else np.nan
        df_bt = pd.DataFrame({
            "Fecha": y.index[-window_days:],
            "Real": actuals,
            "Predicción": preds,
            "Error Absoluto": np.abs(actuals - preds)
        }).set_index("Fecha")
        return mae, mape, df_bt

    # Parámetros del modelo
    st.subheader("3.2–3.4 Pronóstico y validación")
    target = st.selectbox("Variable objetivo", ["Nuevos confirmados", "Nuevas muertes"])
    horizon = st.slider("Horizonte de pronóstico (días)", min_value=7, max_value=28, value=14, step=1)
    bt_window = st.slider("Backtesting - ventana (días)", min_value=7, max_value=35, value=21, step=1)

    if target == "Nuevos confirmados":
        y_raw = df_pais["NewConfirmed"]
        y_smooth = df_pais["NewConfirmed_7d"]
        titulo = f"{pais_ts} – Pronóstico de nuevos confirmados"
    else:
        y_raw = df_pais["NewDeaths"]
        y_smooth = df_pais["NewDeaths_7d"]
        titulo = f"{pais_ts} – Pronóstico de nuevas muertes"

    # elegimos modelar la serie suavizada (ruido menor) pero puedes alternar a y_raw
    y = _clean_series(y_smooth)
    if y is None or y.sum() == 0 or len(y) < 21:
        st.warning("Serie insuficiente para modelar (muy corta o todo ceros). Prueba otro país o rango.")
    else:
        # 3.3 Backtesting
        with st.spinner("Ejecutando backtesting..."):
            mae, mape, df_bt = backtest_mae_mape(y, window_days=bt_window)

        if mae is not None:
            met1, met2 = st.columns(2)
            with met1:
                st.metric("MAE (últimos días)", f"{mae:,.2f}")
            with met2:
                st.metric("MAPE (últimos días)", f"{mape:.2f}%" if pd.notna(mape) else "N/A")
            with st.expander("Ver detalle de backtesting"):
                st.dataframe(df_bt)

        # 3.2 & 3.4 Entrenar en toda la serie y pronosticar con bandas
        with st.spinner("Ajustando modelo y generando pronóstico..."):
            res, mean_fc, conf_fc = fit_sarimax_and_forecast(y, steps=horizon)

        # preparar dataframe de forecast
        fc_index = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        df_fc = pd.DataFrame({
            "mean": mean_fc.values,
            "lower": conf_fc.iloc[:,0].values,
            "upper": conf_fc.iloc[:,1].values
        }, index=fc_index)

        # 3.4 Gráfica con bandas de confianza
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y.index, y.values, label="Serie (7d)")
        ax.plot(df_fc.index, df_fc["mean"], label="Forecast")
        ax.fill_between(df_fc.index, df_fc["lower"], df_fc["upper"], alpha=0.3, label="IC 95%")
        ax.set_title(titulo)
        ax.set_xlabel("Fecha"); ax.set_ylabel("Casos diarios (suavizados)")
        ax.legend()
        st.pyplot(fig)

        with st.expander("Tabla de pronóstico (valores y bandas)"):
            st.dataframe(df_fc.round(2))
