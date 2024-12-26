import math
import numpy as np
import streamlit as st
import plotly.graph_objs as go

# Fonction pour calculer les options
def option_price(S0, K, T, r, sigma, steps, option_style="european", option_type="call"):
    dt = T / steps  # Pas de temps
    u = math.exp(sigma * dt**0.5)  # Facteur de hausse
    d = 1 / u  # Facteur de baisse
    p = (math.exp(r * dt) - d) / (u - d)  # Probabilité neutre au risque
    q = 1 - p

    # Construction des prix des actions
    Stock_Prices = []
    for i in range(steps + 1):
        node_prices = [S0 * u**(i - j) * d**j for j in range(i + 1)]
        Stock_Prices.append(node_prices)

    # Calcul des payoffs finaux
    Payoffs = [
        max(0, K - Stock_Prices[steps][j]) if option_type == "put" else max(0, Stock_Prices[steps][j] - K)
        for j in range(steps + 1)
    ]

    # Rétropropagation pour calculer les prix et les grecs
    for i in range(steps - 1, -1, -1):
        interm = []
        for j in range(i + 1):
            continuation = math.exp(-r * dt) * (p * Payoffs[j] + q * Payoffs[j + 1])
            if option_style == "american":
                if option_type == "put":
                    interm.append(max(K - Stock_Prices[i][j], continuation))
                elif option_type == "call":
                    interm.append(max(Stock_Prices[i][j] - K, continuation))
            else:
                interm.append(continuation)
        Payoffs = interm

    return {"price": Payoffs[0]}

# Application Streamlit
st.title("American Option Pricing Dashboard")

# Paramètres utilisateur
st.sidebar.header("Input Parameters")
S0 = st.sidebar.number_input("Initial Stock Price (S0)", value=100.00, format="%.2f")
K = st.sidebar.number_input("Strike Price (K)", value=90.0, format="%.2f")
T = st.sidebar.number_input("Time to Maturity (T, years)", value=1.5000, format="%.4f")
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, format="%.5f")
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, format="%.4f")
steps = st.sidebar.number_input("Number of Steps in Binomial Tree", value=4, min_value=1, step=1)

option_style = st.sidebar.selectbox("Option Style", ["american", "european"])

# Options Call et Put
put = option_price(S0, K, T, r, sigma, steps, option_style=option_style, option_type="put")
call = option_price(S0, K, T, r, sigma, steps, option_style=option_style, option_type="call")

# Section principale avec les graphiques et contrôles côte à côte
col1, col2 = st.columns([1, 3])

with col1:
    # Contrôles pour les graphiques
    st.markdown("### Graph Options")
    option_type = st.radio("Option Type", ["Call", "Put", "Both"], horizontal=False)
    visualization = st.selectbox("Metric to Plot", ["Delta", "Gamma"])
    S0_min = st.number_input("Min Spot Price", value=50.0, format="%.2f")
    S0_max = st.number_input("Max Spot Price", value=150.0, format="%.2f")
    num_points = st.number_input("Number of Points", value=50, min_value=10, step=1)

# Génération des données pour le graphique
spot_prices = np.linspace(S0_min, S0_max, num_points)
call_values = np.array([spot_price * 0.01 for spot_price in spot_prices])  # Exemple
put_values = np.array([spot_price * 0.02 for spot_price in spot_prices])  # Exemple

with col2:
    # Création du graphique
    fig = go.Figure()

    if option_type in ["Call", "Both"]:
        fig.add_trace(go.Scatter(x=spot_prices, y=call_values, mode="lines", name="Call Option"))

    if option_type in ["Put", "Both"]:
        fig.add_trace(go.Scatter(x=spot_prices, y=put_values, mode="lines", name="Put Option"))

    fig.update_layout(
        title=f"{visualization} as a Function of Spot Price",
        xaxis_title="Spot Price",
        yaxis_title=visualization,
        template="plotly_white"
    )

    st.plotly_chart(fig)
