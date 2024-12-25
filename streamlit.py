#!/usr/bin/env python
# coding: utf-8

# ## streamlit
# 
# New notebook

# In[8]:


#!pip install streamlit
import math
import numpy as np
import streamlit as st


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

    return(Payoffs[0])










# In[6]:


S0 = 57.48  # Initial stock price
K = 50  # Strike price
T = 1  # Time to maturity (1 year)
r = 0.04313  # Risk-free rate (5%)
sigma = 0.1365  # Volatility (20%)
steps = 4  # Number of steps in the binomial tree

option_price(S0, K, T, r, sigma, steps,option_style ="american" ,option_type="put")


# In[9]:


# Streamlit App
st.title("American Option Pricing Dashboard")

# User Inputs
st.sidebar.header("Input Parameters")
S0 = st.sidebar.number_input("Initial Stock Price (S0)", value=57.48, format="%.2f")
K = st.sidebar.number_input("Strike Price (K)", value=50.0, format="%.2f")
T = st.sidebar.number_input("Time to Maturity (T, years)", value=1.0, format="%.2f")
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.04313, format="%.5f")
sigma = st.sidebar.number_input("Volatility (σ)", value=0.1365, format="%.4f")
steps = st.sidebar.number_input("Number of Steps in Binomial Tree", value=4, min_value=1, step=1)

option_style = st.sidebar.selectbox("Option Style", ["american", "european"])
option_type = st.sidebar.selectbox("Option Type", ["put", "call"])

# Calculate Option Price
if st.button("Calculate Option Price"):
    price = option_price(S0, K, T, r, sigma, steps, option_style=option_style, option_type=option_type)
    st.metric(label=f"{option_style.capitalize()} {option_type.capitalize()} Option Price", value=f"${price:.2f}")

