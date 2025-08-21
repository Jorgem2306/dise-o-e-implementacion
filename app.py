import pandas as pd
import matplotlib.pyplot as plt

# 1. Leer dataset desde el URL
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-18-2022.csv"
df = pd.read_csv(url)

# ==========================================================
# 1.a) Primeras 10 líneas, info general y valores faltantes
# ==========================================================
print("Primeras 10 filas:")
print(df.head(10))

print("\nInformación general del dataset:")
print(df.info())

print("\nValores faltantes en cada columna:")
print(df.isnull().sum())

# ==========================================================
# 1.b) Casos confirmados, fallecidos, recuperados y activos por país
# ==========================================================
cases_by_country = df.groupby("Country_Region")[["Confirmed", "Deaths", "Recovered", "Active"]].sum()
print("\nCasos por país:")
print(cases_by_country)

# ==========================================================
# 2.a) Mostrar todas las filas, luego volver al estado inicial
# ==========================================================
pd.set_option("display.max_rows", None)  # Mostrar todas las filas
print("\nTodas las filas:")
print(df)

pd.reset_option("display.max_rows")  # Volver al estado inicial

# ==========================================================
# 2.b) Mostrar todas las columnas, luego volver al estado inicial
# ==========================================================
pd.set_option("display.max_columns", None)  # Mostrar todas las columnas
print("\nTodas las columnas:")
print(df.head())

pd.reset_option("display.max_columns")  # Volver al estado inicial

# ==========================================================
# 2.c) Gráfica de líneas: países con más de 2500 fallecidos
# ==========================================================
filtered = cases_by_country[cases_by_country["Deaths"] > 2500]
filtered[["Confirmed", "Deaths", "Recovered", "Active"]].plot(kind="line", figsize=(12,6), marker="o")
plt.title("COVID-19 por país (fallecidos > 2500)")
plt.ylabel("Número de casos")
plt.xticks(rotation=90)
plt.show()

# ==========================================================
# 2.d) Gráfica de barras: fallecidos en estados de Estados Unidos
# ==========================================================
us_states = df[df["Country_Region"] == "US"].groupby("Province_State")["Deaths"].sum()
us_states.plot(kind="bar", figsize=(12,6))
plt.title("Fallecidos por estado en EE.UU.")
plt.ylabel("Número de muertes")
plt.xticks(rotation=90)
plt.show()

# ==========================================================
# 2.e) Fallecidos en Colombia, Chile, Perú, Argentina y México (gráfico de sectores)
# ==========================================================
selected_countries = cases_by_country.loc[["Colombia", "Chile", "Peru", "Argentina", "Mexico"], "Deaths"]
selected_countries.plot(kind="pie", autopct="%1.1f%%", figsize=(6,6))
plt.title("Fallecidos en países seleccionados")
plt.ylabel("")
plt.show()

# ==========================================================
# 2.f) Histograma del total de fallecidos por país
# ==========================================================
cases_by_country["Deaths"].plot(kind="hist", bins=30, figsize=(8,6))
plt.title("Histograma de fallecidos por país")
plt.xlabel("Número de muertes")
plt.show()

# ==========================================================
# 2.g) Boxplot de Confirmed, Deaths, Recovered y Active
# ==========================================================
cases_by_country[["Confirmed", "Deaths", "Recovered", "Active"]].plot(kind="box", figsize=(8,6))
plt.title("Distribución de casos por país")
plt.show()

