
# In[5]:

import math
import numpy as np
import streamlit as st
import plotly.graph_objs as go



# In[5]:


def option_price(S0, K, T, r, sigma, steps,option_style ="european" ,option_type="call") : 
    dt = T / steps  # Time step
    u = math.exp(sigma * dt**0.5)  # Up factor
    d = 1 / u  # Down factor
    p = (math.exp(r * dt) - d) / (u - d)  # Risk-neutral probability
    q = 1-p
    Stock_Prices = []
    for i in range(steps+1) :
        node_prices = []
        for j in range(i+1) :
            node_prices.append(S0*u**(i-j)*d**j)
        Stock_Prices.append(node_prices)

    Payoffs = []
    for j in range(steps+1) :
        if option_type=="put" :
            Payoffs.append(np.max([K-Stock_Prices[steps][j],0]))
        elif option_type=="call" :
            Payoffs.append(np.max([Stock_Prices[steps][j]-K,0]))

    for i in range(steps-1,-1,-1) :
        interm = []
        for j in range (i+1) :
            if option_style=="european" :
                interm.append(math.exp(-r*dt)*(p*Payoffs[j]+q*Payoffs[j+1]))
            elif option_style=="american" :
                if option_type=="call" :
                    interm.append(np.max([Stock_Prices[i][j] - K, math.exp(-r*dt)*(p*Payoffs[j]+q*Payoffs[j+1])]))
                elif option_type=="put" :
                    interm.append(np.max([K - Stock_Prices[i][j], math.exp(-r * dt) * (p * Payoffs[j] + q * Payoffs[j + 1])]))
        Payoffs = interm
        if i==1 :
            deltas = Payoffs
        if i==2 :
            gammas = Payoffs
    delta = (deltas[1]-deltas[0])/(Stock_Prices[1][1]-Stock_Prices[1][0])
    h = 0.5*(Stock_Prices[2][0]-Stock_Prices[2][2])
    h1 = Stock_Prices[2][0]-Stock_Prices[2][1]
    h2 = Stock_Prices[2][1]-Stock_Prices[2][2]
    f1 = gammas[2]-gammas[1]
    f2 = gammas[1]-gammas[0]
    gamma  = (f1/h2-f2/h1)/h
    theta = (gammas[1]-Payoffs[0])/(2*dt)

    return({"price" : Payoffs[0] , "delta" : delta , "gamma" : gamma , "theta" : theta })


# Streamlit App
st.title("American Option Pricing Dashboard")

# User Inputs
st.sidebar.header("Input Parameters")
S0 = st.sidebar.number_input("Initial Stock Price (S0)", value=100.00, format="%.2f")
K = st.sidebar.number_input("Strike Price (K)", value=90.0, format="%.2f")
T = st.sidebar.number_input("Time to Maturity (T, years)", value=1.5000, format="%.4f")
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, format="%.5f")
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, format="%.4f")
steps = st.sidebar.number_input("Number of Steps in Binomial Tree", value=4, min_value=1, step=1)

option_style = st.sidebar.selectbox("Option Style", ["american", "european"])

# Function for Option Pricing (Already Defined Above)
# ...

# Get call and put prices and Greeks
put = option_price(S0, K, T, r, sigma, steps, option_style=option_style, option_type="put")
call = option_price(S0, K, T, r, sigma, steps, option_style=option_style, option_type="call")

# Display Metrics in Separate Containers
with st.container(border = True):
    st.markdown(f"### {option_style.capitalize()} Call Option")
    col1, col2,col3,col4 = st.columns(4)
    with col1:
        st.metric(label="Call Price", value=f"${call['price']:.4f}")
    with col2:
        st.metric(label="Call Delta", value=f"{call['delta']:.4f}")
    with col3:
        st.metric(label="Call Gamma", value=f"{call['gamma']:.4f}")
    with col4:
        st.metric(label="Call Theta", value=f"{call['theta']:.4f}")

with st.container(border = True):
    st.markdown(f"### {option_style.capitalize()} Put Option")
    col1, col2,col3,col4 = st.columns(4)
    with col1:
        st.metric(label="Put Price", value=f"${put['price']:.4f}")
    with col2:
        st.metric(label="Put Delta", value=f"{put['delta']:.4f}")
    with col3:
        st.metric(label="Put Gamma", value=f"{put['gamma']:.4f}")
    with col4:
        st.metric(label="Put Theta", value=f"{put['theta']:.4f}")


def delta(S0, K, T, r, sigma, steps, option_style, option_type):
    return option_price(S0, K, T, r, sigma, steps, option_style, option_type)["delta"]

def gamma(S0, K, T, r, sigma, steps, option_style, option_type):
    return option_price(S0, K, T, r, sigma, steps, option_style, option_type)["gamma"]

def theta(S0, K, T, r, sigma, steps, option_style, option_type):
    return option_price(S0, K, T, r, sigma, steps, option_style, option_type)["theta"]



###################################
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Assuming the option pricing functions (delta, gamma, theta) are defined elsewhere.

# Create a container for the visualization and controls
st.markdown("### Graph Options")

# Radio button for choosing Call, Put, or Both (above the graph)
option_type = st.radio(
    "Option Type to Plot:",
    ["Call", "Put", "Both"],
    horizontal=True  # Display radio buttons horizontally
)

# Create columns for the controls and the plot
col_controls, col_plot = st.columns([1, 3])  # 1:3 ratio for controls and plot

with col_controls:
    # Controls for the plot
    visualization = st.selectbox("Metric to Plot:", ["Delta", "Gamma", "Theta"])
    
    # Control for selecting the x-axis variable
    x_axis_variable = st.selectbox(
        "Choose the Variable for X-Axis:",
        ["Spot Price (S0)", "Strike Price (K)", "Interest Rate (r)", "Time to Maturity (T)", "Volatility (sigma)"]
    )
    
    # Define common min and max values for x-axis
    X_min = st.number_input("Minimum X Value", value=50.0, format="%.2f")
    X_max = st.number_input("Maximum X Value", value=150.0, format="%.2f")
    
    # Define values for K, r, T, sigma which will not change unless specified
    K = st.number_input("Strike Price (K)", value=100.0, format="%.2f")
    r = st.number_input("Interest Rate (r)", value=0.05, format="%.4f")
    T = st.number_input("Time to Maturity (T)", value=1.0, format="%.2f")
    sigma = st.number_input("Volatility (sigma)", value=0.2, format="%.2f")
    
    num_points = st.number_input("Number of Points:", value=50, min_value=10, step=1)

with col_plot:
    # Generate data for the plot based on the selected x-axis variable
    if x_axis_variable == "Spot Price (S0)":
        x_values = np.linspace(X_min, X_max, num_points)
        call_values, put_values = [], []

        if visualization == "Delta":
            call_values = np.array([delta(s, K, T, r, sigma, num_points, option_type="call") for s in x_values])
            put_values = np.array([delta(s, K, T, r, sigma, num_points, option_type="put") for s in x_values])
        elif visualization == "Gamma":
            call_values = np.array([gamma(s, K, T, r, sigma, num_points, option_type="call") for s in x_values])
            put_values = np.array([gamma(s, K, T, r, sigma, num_points, option_type="put") for s in x_values])
        elif visualization == "Theta":
            call_values = np.array([theta(s, K, T, r, sigma, num_points, option_type="call") for s in x_values])
            put_values = np.array([theta(s, K, T, r, sigma, num_points, option_type="put") for s in x_values])

    elif x_axis_variable == "Strike Price (K)":
        x_values = np.linspace(X_min, X_max, num_points)
        call_values, put_values = [], []

        if visualization == "Delta":
            call_values = np.array([delta(S0, k, T, r, sigma, num_points, option_type="call") for k in x_values])
            put_values = np.array([delta(S0, k, T, r, sigma, num_points, option_type="put") for k in x_values])
        elif visualization == "Gamma":
            call_values = np.array([gamma(S0, k, T, r, sigma, num_points, option_type="call") for k in x_values])
            put_values = np.array([gamma(S0, k, T, r, sigma, num_points, option_type="put") for k in x_values])
        elif visualization == "Theta":
            call_values = np.array([theta(S0, k, T, r, sigma, num_points, option_type="call") for k in x_values])
            put_values = np.array([theta(S0, k, T, r, sigma, num_points, option_type="put") for k in x_values])

    elif x_axis_variable == "Interest Rate (r)":
        x_values = np.linspace(X_min, X_max, num_points)
        call_values, put_values = [], []

        if visualization == "Delta":
            call_values = np.array([delta(S0, K, T, rate, sigma, num_points, option_type="call") for rate in x_values])
            put_values = np.array([delta(S0, K, T, rate, sigma, num_points, option_type="put") for rate in x_values])
        elif visualization == "Gamma":
            call_values = np.array([gamma(S0, K, T, rate, sigma, num_points, option_type="call") for rate in x_values])
            put_values = np.array([gamma(S0, K, T, rate, sigma, num_points, option_type="put") for rate in x_values])
        elif visualization == "Theta":
            call_values = np.array([theta(S0, K, T, rate, sigma, num_points, option_type="call") for rate in x_values])
            put_values = np.array([theta(S0, K, T, rate, sigma, num_points, option_type="put") for rate in x_values])

    elif x_axis_variable == "Time to Maturity (T)":
        x_values = np.linspace(X_min, X_max, num_points)
        call_values, put_values = [], []

        if visualization == "Delta":
            call_values = np.array([delta(S0, K, time, r, sigma, num_points, option_type="call") for time in x_values])
            put_values = np.array([delta(S0, K, time, r, sigma, num_points, option_type="put") for time in x_values])
        elif visualization == "Gamma":
            call_values = np.array([gamma(S0, K, time, r, sigma, num_points, option_type="call") for time in x_values])
            put_values = np.array([gamma(S0, K, time, r, sigma, num_points, option_type="put") for time in x_values])
        elif visualization == "Theta":
            call_values = np.array([theta(S0, K, time, r, sigma, num_points, option_type="call") for time in x_values])
            put_values = np.array([theta(S0, K, time, r, sigma, num_points, option_type="put") for time in x_values])

    elif x_axis_variable == "Volatility (sigma)":
        x_values = np.linspace(X_min, X_max, num_points)
        call_values, put_values = [], []

        if visualization == "Delta":
            call_values = np.array([delta(S0, K, T, r, vol, num_points, option_type="call") for vol in x_values])
            put_values = np.array([delta(S0, K, T, r, vol, num_points, option_type="put") for vol in x_values])
        elif visualization == "Gamma":
            call_values = np.array([gamma(S0, K, T, r, vol, num_points, option_type="call") for vol in x_values])
            put_values = np.array([gamma(S0, K, T, r, vol, num_points, option_type="put") for vol in x_values])
        elif visualization == "Theta":
            call_values = np.array([theta(S0, K, T, r, vol, num_points, option_type="call") for vol in x_values])
            put_values = np.array([theta(S0, K, T, r, vol, num_points, option_type="put") for vol in x_values])

    # Create the plot using Plotly
    fig = go.Figure()

    # Add traces based on the selected option type
    if option_type in ["Call", "Both"]:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=call_values,
                mode="lines",
                name="Call Option"
            )
        )
    if option_type in ["Put", "Both"]:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=put_values,
                mode="lines",
                name="Put Option"
            )
        )

    # Update layout of the plot
    fig.update_layout(
        title=f"{visualization} as a Function of {x_axis_variable}",
        xaxis_title=x_axis_variable,
        yaxis_title=f"{visualization}",
        template="plotly_white"
    )

    # Display the plot
    st.plotly_chart(fig)
