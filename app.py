import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import io

# ======================
#  Cargar dataset
# ======================
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-18-2022.csv"
df = pd.read_csv(url)

# Agrupación por país
cases_by_country = df.groupby("Country_Region")[["Confirmed", "Deaths", "Recovered", "Active"]].sum()

st.title("📊 Análisis COVID-19 - 18 Abril 2022 (Johns Hopkins)")

# ======================
# 1.a) Primeras filas + info general
# ======================
st.header("1.a) Vista inicial del dataset")
st.write("Primeras 10 filas del dataset:")
st.dataframe(df.head(10))

st.subheader("Información general del dataset:")
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)

st.subheader("Valores faltantes en el dataset:")
st.write(df.isnull().sum())

# ======================
# 1.b) Casos por país
# ======================
st.header("1.b) Casos confirmados, fallecidos, recuperados y activos por país")
st.dataframe(cases_by_country)

# ======================
# 2.a) Mostrar todas las filas
# ======================
st.header("2.a) Mostrar todas las filas")
st.dataframe(df)

# ======================
# 2.b) Mostrar todas las columnas
# ======================
st.header("2.b) Mostrar todas las columnas")
st.dataframe(df.head())  # Streamlit ya muestra todas las columnas

# ======================
# 2.c) Gráfica de líneas (países con fallecidos > 2500)
# ======================
st.header("2.c) Gráfica de líneas - Países con fallecidos > 2500")
filtered = cases_by_country[cases_by_country["Deaths"] > 2500]
fig, ax = plt.subplots(figsize=(12,6))
filtered[["Confirmed", "Deaths", "Recovered", "Active"]].plot(kind="line", marker="o", ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)

# ======================
# 2.d) Gráfica de barras - Estados de EE.UU.
# ======================
st.header("2.d) Fallecidos por estado en EE.UU.")
us_states = df[df["Country_Region"] == "US"].groupby("Province_State")["Deaths"].sum()
fig, ax = plt.subplots(figsize=(12,6))
us_states.plot(kind="bar", ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)

# ======================
# 2.e) Pie chart - Países seleccionados
# ======================
st.header("2.e) Fallecidos en países seleccionados (Colombia, Chile, Perú, Argentina, México)")
selected_countries = cases_by_country.loc[["Colombia", "Chile", "Peru", "Argentina", "Mexico"], "Deaths"]
fig, ax = plt.subplots()
selected_countries.plot(kind="pie", autopct="%1.1f%%", ax=ax)
ax.set_ylabel("")
st.pyplot(fig)

# ======================
# 2.f) Histograma de fallecidos por país
# ======================
st.header("2.f) Histograma de fallecidos por país")
fig, ax = plt.subplots(figsize=(8,6))
cases_by_country["Deaths"].plot(kind="hist", bins=30, ax=ax)
plt.xlabel("Número de muertes")
st.pyplot(fig)

# ======================
# 2.g) Boxplot
# ======================
st.header("2.g) Boxplot de Confirmed, Deaths, Recovered y Active")
fig, ax = plt.subplots(figsize=(8,6))
cases_by_country[["Confirmed", "Deaths", "Recovered", "Active"]].plot(kind="box", ax=ax)
st.pyplot(fig)
