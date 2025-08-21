import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import io

# ======================
#  Cargar dataset
# ======================
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-18-2022.csv"
df = pd.read_csv(url)

# Agrupación por país (lo usamos varias veces)
cases_by_country = df.groupby("Country_Region")[["Confirmed", "Deaths", "Recovered", "Active"]].sum()

# ======================
#  Configuración de la App
# ======================
st.title("📊 Análisis COVID-19 - 18 Abril 2022 (Johns Hopkins)")

menu = st.sidebar.radio(
    "Selecciona el inciso que quieres ver:",
    (
        "1.a - Vista inicial",
        "1.b - Casos por país",
        "2.a - Todas las filas",
        "2.b - Todas las columnas",
        "2.c - Gráfica de líneas",
        "2.d - Barras EE.UU.",
        "2.e - Pie chart LATAM",
        "2.f - Histograma",
        "2.g - Boxplot"
    )
)

# ======================
# 1.a) Primeras filas + info general
# ======================
if menu == "1.a - Vista inicial":
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
elif menu == "1.b - Casos por país":
    st.header("1.b) Casos confirmados, fallecidos, recuperados y activos por país")
    st.dataframe(cases_by_country)

# ======================
# 2.a) Mostrar todas las filas
# ======================
elif menu == "2.a - Todas las filas":
    st.header("2.a) Mostrar todas las filas")
    st.dataframe(df)

# ======================
# 2.b) Mostrar todas las columnas
# ======================
elif menu == "2.b - Todas las columnas":
    st.header("2.b) Mostrar todas las columnas")
    st.dataframe(df.head())  # Streamlit ya muestra todas las columnas

# ======================
# 2.c) Gráfica de líneas (países con fallecidos > 2500)
# ======================
elif menu == "2.c - Gráfica de líneas":
    st.header("2.c) Gráfica de líneas - Países con fallecidos > 2500")
    filtered = cases_by_country[cases_by_country["Deaths"] > 2500]
    fig, ax = plt.subplots(figsize=(12,6))
    filtered[["Confirmed", "Deaths", "Recovered", "Active"]].plot(kind="line", marker="o", ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

# ======================
# 2.d) Gráfica de barras - Estados de EE.UU.
# ======================
elif menu == "2.d - Barras EE.UU.":
    st.header("2.d) Fallecidos por estado en EE.UU.")
    us_states = df[df["Country_Region"] == "US"].groupby("Province_State")["Deaths"].sum()
    fig, ax = plt.subplots(figsize=(12,6))
    us_states.plot(kind="bar", ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

# ======================
# 2.e) Pie chart - Países seleccionados
# ======================
elif menu == "2.e - Pie chart LATAM":
    st.header("2.e) Fallecidos en países seleccionados (Colombia, Chile, Perú, Argentina, México)")
    selected_countries = cases_by_country.loc[["Colombia", "Chile", "Peru", "Argentina", "Mexico"], "Deaths"]
    fig, ax = plt.subplots()
    selected_countries.plot(kind="pie", autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

# ======================
# 2.f) Histograma de fallecidos por país
# ======================
elif menu == "2.f - Histograma":
    st.header("2.f) Histograma de fallecidos por país")
    fig, ax = plt.subplots(figsize=(8,6))
    cases_by_country["Deaths"].plot(kind="hist", bins=30, ax=ax)
    plt.xlabel("Número de muertes")
    st.pyplot(fig)

# ======================
# 2.g) Boxplot
# ======================
elif menu == "2.g - Boxplot":
    st.header("2.g) Boxplot de Confirmed, Deaths, Recovered y Active")
    fig, ax = plt.subplots(figsize=(8,6))
    cases_by_country[["Confirmed", "Deaths", "Recovered", "Active"]].plot(kind="box", ax=ax)
    st.pyplot(fig)
