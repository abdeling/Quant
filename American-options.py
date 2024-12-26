
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
###################################

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
    visualization = st.selectbox("Metric to Plot:", ["Delta", "Gamma","Theta"])
    S0_min = st.number_input("Minimum Spot Price", value=50.0, format="%.2f")
    S0_max = st.number_input("Maximum Spot Price", value=150.0, format="%.2f")
    num_points = st.number_input("Number of Points:", value=50, min_value=10, step=1)

with col_plot:
    # Generate data for the plot
    spot_prices = np.linspace(S0_min, S0_max, num_points)
    call_values, put_values = [], []

    # Calculate values for Call and Put options
    if visualization == "Delta":
        call_values = np.array([delta(s, K, T, r, sigma, steps, option_style=option_style, option_type="call") for s in spot_prices])
        put_values = np.array([delta(s, K, T, r, sigma, steps, option_style=option_style, option_type="put") for s in spot_prices])
    elif visualization == "Gamma":
        call_values = np.array([gamma(s, K, T, r, sigma, steps, option_style=option_style, option_type="call") for s in spot_prices])
        put_values = np.array([gamma(s, K, T, r, sigma, steps, option_style=option_style, option_type="put") for s in spot_prices])
    elif visualization == "Theta":
        call_values = np.array([gamma(s, K, T, r, sigma, steps, option_style=option_style, option_type="call") for s in spot_prices])
        put_values = np.array([gamma(s, K, T, r, sigma, steps, option_style=option_style, option_type="put") for s in spot_prices])

    # Create the plot using Plotly
    fig = go.Figure()

    # Add traces based on the selected option type
    if option_type in ["Call", "Both"]:
        fig.add_trace(
            go.Scatter(
                x=spot_prices,
                y=call_values,
                mode="lines",
                name="Call Option"
            )
        )
    if option_type in ["Put", "Both"]:
        fig.add_trace(
            go.Scatter(
                x=spot_prices,
                y=put_values,
                mode="lines",
                name="Put Option"
            )
        )

    # Update layout of the plot
    fig.update_layout(
        title=f"{visualization} as a Function of Spot Price",
        xaxis_title="Spot Price",
        yaxis_title=f"{visualization}",
        template="plotly_white"
    )

    # Display the plot
    st.plotly_chart(fig)
