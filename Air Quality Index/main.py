import pandas as pd 
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.templates.default = "plotly_white"

data = pd.read_csv("delhiaqi.csv")
print(data.head())

data['date'] = pd.to_datetime(data['date'])

fig = go.Figure()
for pollutant in ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']:
    fig.add_trace(go.Scatter(x=data['date'], y=data[pollutant], mode='lines', name=pollutant))

fig.update_layout(title='Time Series Analysis of Air Pollutants in Delhi',xaxis_title='Date',yaxis_title='Concentration (µg/m³)')
fig.show()

