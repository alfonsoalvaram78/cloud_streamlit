#streamlit run streamlit_cloud.py --server.port 8501
#python.exe -m pip install pipreqs
#pipreqs --encoding=utf8
import streamlit as st
import streamlit_funciones as s_fun
import datetime
import pandas as pd
import plotly.express as px  # interactive charts
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh


st.set_page_config(
    page_title="Criptomonedas",
    page_icon=":dollar:",
    layout="wide",
)

#se inicializan las funciones
#se carga la información diaria
s_fun.update_crypto_values_history()
#se carga la información intradia
s_fun.update_crypto_values_day()

st.title("Criptomonedas: Evoluación diaria")

cat_monedas = s_fun.get_data_table('cat_criptomonedas')
rend_hist, fmin, fmax = s_fun.rendimiento_log()
mean_hist = rend_hist['Mean'].dot(rend_hist['Volume'])/rend_hist['Volume'].sum()
std_hist = rend_hist['Std Dev'].dot(rend_hist['Volume'])/rend_hist['Volume'].sum()

f_6meses = datetime.datetime.strptime(fmax, '%Y-%m-%d' ).date() - datetime.timedelta(182)
f_6meses = f_6meses.strftime('%Y-%m-%d')

rend_6m, fmin_6m, fmax_6m = s_fun.rendimiento_log(f_6meses)
mean_6m = rend_6m['Mean'].dot(rend_6m['Volume'])/rend_6m['Volume'].sum()
std_6m = rend_6m['Std Dev'].dot(rend_6m['Volume'])/rend_6m['Volume'].sum()

f_30d = datetime.datetime.strptime(fmax, '%Y-%m-%d' ).date() - datetime.timedelta(30)
f_30d = f_30d.strftime('%Y-%m-%d')

rend_30d, fmin_30d, fmax_30d = s_fun.rendimiento_log(f_30d)
mean_30d = rend_30d['Mean'].dot(rend_30d['Volume'])/rend_30d['Volume'].sum()
std_30d = rend_30d['Std Dev'].dot(rend_30d['Volume'])/rend_30d['Volume'].sum()

placeholder = st.empty()

with placeholder.container():
    _rend, _desv, _coef_var = st.columns(3)

    _rend.metric(
        label="Rendimiento del {} al {} ".format(fmin, fmax),
        value=round(mean_hist,2),
    )

    _desv.metric(
        label="Desviación Estándar del {} al {} ".format(fmin, fmax),
        value=round(std_hist,2),
    )

    _coef_var.metric(
        label="Coef. de Variación {} al {} ".format(fmin, fmax),
        value=round(mean_hist/std_hist,2),
    )

placeholder_6m = st.empty()

with placeholder_6m.container():
    _rend_6m, _desv_6m, _coef_var_6m = st.columns(3)

    _rend_6m.metric(
        label="Rendimiento del {} al {} ".format(fmin_6m, fmax_6m),
        value=round(mean_6m,2),
    )

    _desv_6m.metric(
        label="Desviación Estándar del {} al {} ".format(fmin_6m, fmax_6m),
        value=round(std_6m,2),
    )

    _coef_var_6m.metric(
        label="Coef. de Variación {} al {} ".format(fmin_6m, fmax_6m),
        value=round(mean_6m/std_6m,2),
    )

placeholder_30d = st.empty()

with placeholder_30d.container():
    _rend_30d, _desv_30d, _coef_var_30d = st.columns(3)

    _rend_30d.metric(
        label="Rendimiento del {} al {} ".format(fmin_30d, fmax_30d),
        value=round(mean_30d,2),
    )

    _desv_30d.metric(
        label="Desviación Estándar del {} al {} ".format(fmin_30d, fmax_30d),
        value=round(std_30d,2),
    )

    _coef_var_30d.metric(
        label="Coef. de Variación {} al {} ".format(fmin_30d, fmax_30d),
        value=round(mean_30d/std_30d,2),
    )

####botón para bajar la información
rend_hist['Periodo']  = 'Desde 2022-01-02'
rend_6m['Periodo']  = '6 meses'
rend_30d['Periodo']  = '30 días'
rendimiento = pd.concat([rend_hist, rend_6m, rend_30d], axis = 0, ignore_index = True)

rendimiento = s_fun.put_cripto_names(rendimiento)
download_rendimiento = st.download_button( 'Download Rendimiento', rendimiento.to_csv(index=False, encoding = 'latin1'),file_name = 'rendimiento.csv' )

####Gráficas de desempeño por
genre = st.radio(
    "Escoge el preiodo",
    ('Desde 2022-01-02', 'Últimos 6 meses', 'Últimos 30 días'))

if genre == 'Desde 2022-01-02':
    periodo = 'Desde 2022-01-02'
    _rend = mean_hist
elif genre == 'Últimos 6 meses':
    periodo = '6 meses'
    _rend = mean_6m
else:
    periodo = '30 días'
    _rend = mean_30d

st.markdown("Gráfico de Rendimientos: {}".format(genre))
df = rendimiento[rendimiento['Periodo'] == periodo]
df['Promedio'] = _rend
df.sort_values(by = 'Mean', ascending=True, inplace = True)
fig1 = px.bar(data_frame=df, y="Mean", x="nombre")
fig2 = px.line(data_frame=df, y="Promedio", x="nombre")
fig2['data'][0]['line']['color']='rgb(139,0,0)'
fig = go.Figure(data = fig1.data + fig2.data)
st.write(fig)
comentario = '<p style="font-family:sans-serif; color:gray; font-size: 10px;">La linea corresponde al promedio ponderado por volumen de las Criptomonedas en análisis</p>'
st.markdown(comentario, unsafe_allow_html=True)

st.write('Escoja dos periodos de análisis')

option = st.multiselect('Escoja dos opciones',
                         ['Desde 2022-01-02', 'Últimos 6 meses' ,'Últimos 30 días'],
                         ['Últimos 6 meses', 'Últimos 30 días'])

zoom = st.checkbox('Escoja si desea hacer zoom a la gráfica',
                   value=False)

if len(option)!=2:
    st.write('Debe escoger dos periodos')
else:
    if option[0] == 'Últimos 6 meses' or option[0] == 'Últimos 30 días':
        option[0] = option[0][8:]
    if option[1] == 'Últimos 6 meses' or option[1] == 'Últimos 30 días':
        option[1] = option[1][8:]

    df0 = rendimiento[rendimiento['Periodo'] == option[0]]
    df0 = df0[['nombre', 'Mean']]
    df0.columns = ['nombre', option[0]]
    df1 = rendimiento[rendimiento['Periodo'] == option[1]]
    df1 = df1[['nombre', 'Mean', 'Volume']]
    df1.columns = ['nombre', option[1], 'Volume']
    df = pd.merge(df0, df1, on = ['nombre'])
    df = df[df['Volume']>100]
    v_max = df[[option[0], option[1]]].max().max()
    v_max = int(v_max)
    v_min = df[[option[0], option[1]]].min().min()
    v_min = int(v_min)
    x2 = np.linspace(v_min, v_max, num=v_max-v_min)
    y2 = x2
    fig_2p_1 = px.scatter(df, x=option[0], y=option[1], color='nombre')
    fig_2p_2 = px.line(y=y2, x=x2)
    fig_2p_2['data'][0]['line']['width']=0.5
    fig_2p_2['data'][0]['line']['color']='rgb(139,0,0)'
    fig_2p = go.Figure(data = fig_2p_1.data + fig_2p_2.data)
    if zoom == True:
        config = dict({'scrollZoom': True})
        fig_2p.show(config=config)
        st.plotly_chart(fig_2p)

    st.plotly_chart(fig_2p)
comentario2 = '<p style="font-family:sans-serif; color:gray; font-size: 10px;">Sólo se presentan las criptomonedas con volumen de operación promedio diario mayor a 100 unidades</p>'
st.markdown(comentario2, unsafe_allow_html=True)

comentario3 = '<p style="font-family:sans-serif; color:firebrick; font-size: 25px;">Información de Precios Intradía</p>'
st.markdown(comentario3, unsafe_allow_html=True)

cripto_intradia = s_fun.get_data_table('criptomonedas_day')
cripto_intradia = s_fun.put_cripto_names(cripto_intradia)
crito_names = cripto_intradia['nombre'].unique()
cripto_intradia_gb = cripto_intradia[['nombre', 'Volume']].groupby(by=['nombre'])
cripto_intradia_gb = cripto_intradia_gb['Volume'].mean()
cripto_intradia_gb.sort_values(ascending=False, inplace = True)

fig_daily = make_subplots(rows=1, cols=3, start_cell="top-left")

nom_cripto = cripto_intradia_gb.index

for i in range(3):
    cripto_intradia_sub = cripto_intradia[cripto_intradia['nombre'] == nom_cripto[i]]
    cripto_intradia_sub.sort_values(by = ['Time'], inplace = True)
    cripto_intradia_sub = cripto_intradia_sub[['Time', 'Close']]
    fig_daily.add_trace(go.Scatter(x=cripto_intradia_sub['Time'],
                                   y=cripto_intradia_sub['Close'],
                                   name=nom_cripto[i]),
                                   row=1, col=i+1)

st.plotly_chart(fig_daily, use_container_width=True)


cripto_intradia_gb = cripto_intradia_gb.reset_index()
cripto_intradia_gb['Volume'] = cripto_intradia_gb['Volume'].apply(lambda x: '{:,.0f}'.format(x) )

comentario4 = '<p style="font-family:sans-serif; color:navy; font-size: 15px;">Volumen promedio operado intradía</p>'
st.markdown(comentario4, unsafe_allow_html=True)
st.dataframe(cripto_intradia_gb)

#autorefresh
counter = st_autorefresh(interval=15 * 60 * 1000, key="dataframerefresh")
hh = datetime.datetime.now()
hh = hh.strftime('%Y-%m-%d %H:%M:%S')
st.write('Última Actualización {}'.format(hh))
