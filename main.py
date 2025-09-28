import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import numpy as np

# Fix for yfinance rate limiting issues in Python 3.13
try:
    from curl_cffi import requests
    session = requests.Session(impersonate="chrome")
except ImportError:
    # Fallback to standard requests if curl_cffi is not available
    import requests
    session = requests.Session()

START = "2017-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Symphony 🚀")

stocks = ("AAPL", "GOOGL", "MSFT", "TSLA", "GME")
selected_stocks = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    """Load stock data with session handling for rate limit fix and MultiIndex column handling"""
    try:
        # Use session to avoid rate limiting and disable MultiIndex columns
        data = yf.download(ticker, start=START, end=TODAY, session=session, multi_level_index=False)
        
        # If MultiIndex still exists (fallback for older yfinance versions)
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten the MultiIndex columns by taking the first level
            data.columns = data.columns.get_level_values(0)
        
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Try selecting a different stock or check your internet connection.")
        return None

data_load_state = st.text("Loading data...")
data = load_data(selected_stocks)

if data is not None and not data.empty:
    data_load_state.text("Loading data... done!")
    
    # Debug: Show column names to verify they're clean
    st.write(f"Column names: {list(data.columns)}")
    
    st.subheader('Raw Data')
    st.write(data.tail())

    def plot_raw_data():
        """Plot raw stock data"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
        fig.update_layout(title_text="Time Series Data", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # Forecasting with proper data type handling
    try:
        # Create a clean copy of the data for Prophet
        df_train = data[['Date', 'Close']].copy()
        
        # Ensure Date column is datetime
        df_train['Date'] = pd.to_datetime(df_train['Date'])
        
        # Ensure Close column is numeric and handle any non-numeric values
        df_train['Close'] = pd.to_numeric(df_train['Close'], errors='coerce')
        
        # Drop any rows with NaN values
        df_train = df_train.dropna()
        
        # Rename columns for Prophet (ds must be datetime, y must be numeric)
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
        
        # Additional data validation
        if len(df_train) < 10:
            st.error("Not enough valid data points for forecasting. Please try a different stock.")
        elif df_train['y'].isna().sum() > 0:
            st.error("Data contains invalid values. Please try a different stock or time period.")
        else:
            # Verify data types before fitting
            st.write(f"Prophet data shape: {df_train.shape}")
            st.write(f"Date range: {df_train['ds'].min()} to {df_train['ds'].max()}")
            st.write(f"Price range: ${df_train['y'].min():.2f} to ${df_train['y'].max():.2f}")
            st.write(f"Prophet DataFrame dtypes: {df_train.dtypes}")
            
            # Initialize and fit Prophet model
            m = Prophet(
                daily_seasonality=False,
                yearly_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.05,  # Reduce overfitting
                seasonality_prior_scale=10.0,
                interval_width=0.95
            )
            
            # Fit the model
            m.fit(df_train)
            
            # Create future dataframe
            future = m.make_future_dataframe(periods=period)
            
            # Make predictions
            forecast = m.predict(future)

            st.subheader('Forecast data')
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

            st.write('Forecast visualization')
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)

            st.write('Forecast components')
            fig2 = m.plot_components(forecast)
            st.pyplot(fig2)
            
    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")
        st.info("Debug information:")
        if 'df_train' in locals():
            st.write(f"DataFrame columns: {df_train.columns.tolist()}")
            st.write(f"DataFrame dtypes: {df_train.dtypes}")
            st.write(f"DataFrame shape: {df_train.shape}")
            st.write("Sample data:")
            st.write(df_train.head())
        st.info("Try a different stock or check if the data contains valid numeric values.")
else:
    data_load_state.text("Failed to load data!")
    st.error("Could not load stock data. Please try again or select a different stock.")

st.caption('Made with ❤️ by Ritesh')
