import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

st.set_page_config(page_title='Peramalan Harga Beras')
st.write("""
# Hasil Peramalan Harga Beras Januari 2022 - Juni 2022 Menggunakan Model GRU
""")
tab1, tab2, tab3 = st.tabs(["Data", "Forecasting a few days", "Forecasting one day"])
my_model_params = [
    [32, 7, 100],
    [32, 30, 100],
    [64, 7, 100],
    [64, 30, 100],
    [32, 7, 63],
    [32, 30, 62],
    [64, 7, 100],
    [64, 30, 100]
]

with tab1:
    st.info("Model pada aplikasi ini dilatih menggunakan data harga beras IR-64 I harian antara 1 Januari 2016 - 31 Desember 2021 yang diambil dari website PT FoodStation")
    dataset = pd.read_csv("Data Beras IR-64 I.csv")
    dataset.insert(loc = 0,
            column = 'No',
            value = range(1, len(dataset) + 1))
    gb = GridOptionsBuilder.from_dataframe(dataset)
    gb.configure_pagination(
        paginationAutoPageSize=False, 
        paginationPageSize=10
    )
    gridOptions = gb.build()
    AgGrid(dataset, gridOptions=gridOptions)

    fig, ax = plt.subplots()
    ax.plot(dataset['IR-64 I'])
    ax.set_title("Visualisasi Harga Beras IR-64 I 2016 - 2021")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    st.pyplot(fig)

with tab2:
    model = st.radio(
        "Select model",
        ('Model 1', 'Model 2', 'Model 3', 'Model 4',
         'Model 5', 'Model 6', 'Model 7', 'Model 8')
    )
    
    filename = 'Data Prediksi Beras 2022 Skenario {}.csv'.format(model.split()[1])
    df = pd.read_csv(filename)
    df.insert(loc = 0,
            column = 'No',
            value = range(1, 365))
    tanggal = df['Tanggal'].values

    params = my_model_params[int(model.split()[1]) - 1]
    st.info("""
        GRU Neuron/Units: **{}**,  Window Size: **{}**, Max Epoch: **{}**"""
            .format(params[0], params[1], params[2])
    )

    number = st.slider(
        "Pick number of days to forecast", 
        min_value = 1, max_value = 364 , 
        value=(1, 364)
    )
    start_date = datetime.datetime.strptime(
        tanggal[number[0] - 1], 
        '%Y-%m-%d'
    ).strftime("%d %B %Y")
    end_date = datetime.datetime.strptime(
        tanggal[number[1] - 1], 
        '%Y-%m-%d'
    ).strftime("%d %B %Y")
    st.info("""
    Start Date: **{}**,  End Date: **{}**""".format(start_date, end_date))
    new_df = df[(df['No'] >= number[0]) & (df['No'] <= number[1])]
    gb = GridOptionsBuilder.from_dataframe(new_df)
    gb.configure_pagination(
        paginationAutoPageSize=False, 
        paginationPageSize=10
    )
    gridOptions = gb.build()
    AgGrid(new_df, gridOptions=gridOptions)
    mse = mean_squared_error(
        new_df['IR-64 I Actual Price'].values, 
        new_df['IR-64 I Predictions Price'].values
    )
    mse_without_squared = mean_squared_error(
        new_df['IR-64 I Actual Price'].values, 
        new_df['IR-64 I Predictions Price'].values,
        squared=False
    )
    mape = mean_absolute_percentage_error(
        new_df['IR-64 I Actual Price'].values, 
        new_df['IR-64 I Predictions Price'].values,
    )
    if (mape < 0.1):
        justifikasi = "sangat bagus"
    elif (mape >= 0.1 and mape < 0.2):
        justifikasi = "bagus"
    elif (mape >= 0.2 and mape < 0.5):
        justifikasi = "cukup bagus"
    else:
        justifikasi = "kurang bagus"

    st.info("""
    Mean Squared Error : **{}**\n
    Root Mean Squared Error : **{}**\n
    Mean Absolute Precentage Error : **{}**\n
    \n
    Berdasarkan nilai metrik diatas, maka hasil dari peramalan model dapat dikatakan **{}**.
    """.format(mse, mse_without_squared, mape, justifikasi))

    arr = new_df[['IR-64 I Actual Price', 'IR-64 I Predictions Price']]
    fig, ax = plt.subplots()
    ax.plot(arr)
    ax.legend(['Actual Price', 'Predictions Price'])
    ax.set_title("Visualisasi Harga Beras Actual dan Prediksi")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    st.pyplot(fig)

with tab3:
    d = st.date_input(
        "Select date to forecast",
        datetime.date(2022, 1, 1),
        min_value = datetime.date(2022, 1, 1),
        max_value = datetime.date(2022, 6, 30)
    )
    actual_price = df[df['Tanggal'] == d.strftime(
        '%Y-%m-%d'
    )]['IR-64 I Actual Price'].values[0]
    predicted_price = df[df['Tanggal'] == d.strftime(
        '%Y-%m-%d'
    )]['IR-64 I Predictions Price'].values[0]
    st.write("""
    Predicted Price : {}\n
    Actual Price : {} 
    """.format(predicted_price, actual_price))