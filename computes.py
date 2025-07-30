import streamlit as st
st.set_page_config(layout="wide")
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

import seaborn as sns


def delete_columns_with_zeros(matrix):
    """
    Deletes columns from a NumPy matrix that contain at least one zero.

    Args:
        matrix (np.ndarray): The input NumPy matrix.

    Returns:
        np.ndarray: A new NumPy matrix with columns containing zeros deleted.
                    Returns an empty array if all columns contain zeros.
    """
    if not isinstance(matrix, np.ndarray):
        print("Error: Input must be a NumPy array.")
        return np.array([])

    if matrix.size == 0:
        print("Input matrix is empty.")
        return np.array([])

    # Find columns that contain at least one zero
    # np.any(matrix == 0, axis=0) returns a boolean array
    # where True indicates a column has at least one zero.
    columns_to_keep = np.all(matrix != 0, axis=0)

    # Use boolean indexing to select only the columns that do not contain zeros
    new_matrix = matrix[:, columns_to_keep]

    return new_matrix

### App functions ######################################################
def BlackScholes(r, S, K, T, sigma, tipo = 'C') : 
    ''' 
    r : Interest Rate
    S : Spot Price
    K : Strike Price
    T : Days due expiration / 365
    sigma : Annualized Volatility 
    '''
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma* np.sqrt(T)) 
    d2 = d1 - sigma * np.sqrt(T)
    try : 
        if tipo == 'C' : 
            precio = S * norm.cdf(d1, 0,1) -  K * np.exp(-r * T)*norm.cdf(d2, 0,1)
        elif tipo == 'P' : 
            precio = K * np.exp(-r * T)*norm.cdf(-d2, 0,1) - S * norm.cdf(-d1, 0,1)
    except : 
        print('Error')
    return precio

def HeatMapMatrix(Spot_Prices, Volatilities, Strike, Interest_Rate, Days_to_Exp, type = 'C') :
    M = np.zeros(shape=(len(Spot_Prices), len(Volatilities)))
    T = Days_to_Exp / 365
    for i in range(len(Spot_Prices)) : 
        for j in range(len(Volatilities)) : 
            BS_result =  BlackScholes(Interest_Rate,  Spot_Prices[i], Strike, T, Volatilities[j], type  )
            M[i,j] = round(BS_result,2)
    return M
###############################################################################################################
#### Sidebar parameters ###############################################
st.sidebar.header('Option Parameters')
Underlying_price = st.sidebar.number_input('Spot Price', value = 100)
trade_type = st.sidebar.segmented_control("Contract type", ['Call', 'Put'], default= 'Call')
SelectedStrike = st.sidebar.number_input('Strike/Exercise Price', value = 80.0)
days_to_maturity = st.sidebar.number_input('Time to Maturity (days)', value = 365)
relative_maturity_time = days_to_maturity / 365
Risk_Free_Rate = st.sidebar.number_input('Risk-Free Interest Rate ', value = 0.1)
volatility = st.sidebar.number_input('Annualized Volatility', value = 0.2)
st.sidebar.subheader('P&L Parameters')
option_purchase_price = st.sidebar.number_input("Option's Price") 
transaction_cost = st.sidebar.number_input("Opening/Closing Cost") 

st.sidebar.subheader('Heatmap Parameters')
min_spot_price = st.sidebar.number_input('Min Spot price',value= 50)
max_spot_price = st.sidebar.number_input('Max Spot price', value = 110)

min_vol = st.sidebar.slider('Min Volatility', 0.01, 1.00)
max_vol = st.sidebar.slider('Max Volatility', 0.01, 1.00, 1.00)
grid_size = st.sidebar.slider('Grid size (nxn)', 5, 20, 10)
#### Variables ########################################################
SpotPrices_space = np.linspace(min_spot_price, max_spot_price, grid_size)
Volatilities_space = np.linspace(min_vol,max_vol,grid_size)
########################################################################

st.header('Black Scholes options heatmap')
st.write("Calculates an option's arbitrage-free premium using the Black Scholes option pricing model.")



call_price = BlackScholes(Risk_Free_Rate,  Underlying_price, SelectedStrike, days_to_maturity / 365, volatility)
put_price = BlackScholes(Risk_Free_Rate,  Underlying_price, SelectedStrike, days_to_maturity / 365, volatility, 'P')

cal_contract_prices = [call_price, put_price]
t1_col1, t1_col2 = st.columns(2)
with t1_col1 : 
    st.markdown(f"Call value: **{round(call_price,3)}**")
with t1_col2 : 
    st.markdown(f"Put value: **{round(put_price,3)}**")


tab1, tab2, tab3 = st.tabs(["Option's fair value heatmap", "Option's P&L heatmap", "Expected underlying distribution"])

###### Operations

output_matrix_C = HeatMapMatrix(SpotPrices_space, Volatilities_space, SelectedStrike, Risk_Free_Rate, days_to_maturity)
output_matrix_P = HeatMapMatrix(SpotPrices_space, Volatilities_space, SelectedStrike, Risk_Free_Rate, days_to_maturity, type= 'P')
##### General Info ######

##### Heatmaps configuration    #################################################################

with tab1 : 
    st.write("Explore different contract's values given variations in Spot Prices and Annualized Volatilities")
    fig, axs = plt.subplots(2, 1, figsize=(25, 25))

    sns.heatmap(output_matrix_C.T, annot=True, fmt='.1f' ,
                                xticklabels=[str(round(i, 2)) for i in SpotPrices_space], 
                                yticklabels= [str(round(i, 2)) for i in Volatilities_space], ax=axs[0], 
                                cbar_kws={'label': 'Call Value',})
    axs[0].set_title('Call heatmap', fontsize=20)
    axs[0].set_xlabel('Spot Price', fontsize=15)
    axs[0].set_ylabel('Annualized Volatility', fontsize=15)

    sns.heatmap(output_matrix_P.T, annot=True, fmt='.1f' ,
                                xticklabels=[str(round(i, 2)) for i in SpotPrices_space], 
                                yticklabels= [str(round(i, 2)) for i in Volatilities_space], ax=axs[1], 
                                cbar_kws={'label': 'Put Value',})

    axs[1].set_title('Put heatmap', fontsize=20)
    axs[1].set_xlabel('Spot Price', fontsize=15)
    axs[1].set_ylabel('Annualized Volatility', fontsize=15)

    st.pyplot(fig)
with tab2 : 
    st.write("Explore different expected P&L's from a specific contract trade given variations in the Spot Price and Annualized Volatility ")
        
         

    fig, axs = plt.subplots(1, 1, figsize=(25, 15))

    call_PL = output_matrix_C.T - option_purchase_price - 2 * transaction_cost
    put_PL = output_matrix_P.T- option_purchase_price - 2 * transaction_cost
    PL_options =  [call_PL, put_PL]
    selection = 0
    if trade_type == 'Call' : 
        selection = 0
    else :
        selection = 1

    specific_contrac_pl = cal_contract_prices[selection] - option_purchase_price - 2 * transaction_cost

    relative_specific_contrac_pl =  specific_contrac_pl / (option_purchase_price + 2 * transaction_cost)
    
    t2_col1, t2_col2 = st.columns(2)
    t2_col1.metric(f"**Expected P&L:**", round(specific_contrac_pl,2), border = True)
    t2_col2.metric(f"**Expected relative P&L:**", f'{round(relative_specific_contrac_pl *100 ,2)}%', border = True)

    maping_color = sns.diverging_palette(15, 145, s=60, as_cmap=True)

    sns.heatmap(PL_options[selection], annot=True, fmt='.1f' ,
                                xticklabels=[str(round(i, 2)) for i in SpotPrices_space], 
                                yticklabels= [str(round(i, 2)) for i in Volatilities_space], ax=axs, 
                                cmap =maping_color, center = 0)
    axs.set_title(f'{trade_type} Expected P&L', fontsize=20)
    axs.set_xlabel('Spot Price', fontsize=15)
    axs.set_ylabel('Annualized Volatility', fontsize=15)


    st.pyplot(fig)

with tab3 : 
    st.write('Calculate the expected distribution of the underlying asset price, the option premium and the p&l from trading the option')
    with st.expander("See methodology"):
        st.write('The distribution is obtained by simulating $N$ times the underlying asset price as a geometric brownian process during a specified time period.' \
        ' The function $S : [0, \infty) \mapsto [0, \infty) $ will describe the stochastic process as: ')
        st.latex('S(t) = S(0) e^{(\mu - \sigma^2 / 2)t + \sigma W(t)} ')
        st.write('Where $\mu$ is the risk free rate, $\sigma$ the annualized volatility of the asset you want to simulate and $S(0)$ the asset price at the beginning (spot price)')
    t3_col1, t3_col2, t3_col3 = st.columns(3)
    with t3_col1 : 
        NS  = st.slider('Number of simulations ($N$)', 100, 10000, 1000, 10)
    with t3_col2 :
        s_selection = st.radio('Select time interval', ['Days', 'Hours', 'Minutes'], horizontal= True, help= 'The time inerval each price point will represent. This option is merely for visual purposes.')
    with t3_col3 : 
        timeshot = st.slider("Select chart's timestamp (days/year)", 0.0, days_to_maturity / 365, days_to_maturity / 365) 

    if s_selection == 'Days' : 
        step = days_to_maturity 
    elif s_selection == 'Hours' : 
        step = days_to_maturity * 24 
    elif s_selection == 'Minutes' :
        step = days_to_maturity * 24 * 60 
    
    #### Creating the simulations
    
    @st.cache_data
    def simulate(NS, days_to_maturity, s, volatility, Risk_Free_Rate) : 
        dt = (days_to_maturity / 365) /s
        Z = np.random.normal(0, np.sqrt(dt), (s, NS) )
        paths =  np.vstack([np.ones(NS), np.exp((Risk_Free_Rate - 0.5 * volatility**2 ) * dt + volatility * Z)]).cumprod(axis = 0)
        
        return paths 
    
    simulation_paths = Underlying_price * simulate(NS, days_to_maturity, step, volatility, Risk_Free_Rate)

    def get_Option_Price(K,St, type = 'Call'):
        dynamic_index = -int(step - timeshot * 365 * (step/days_to_maturity) + 1)
        expiration_price = 0
        try: 
            if type == 'Call' : 
                expiration_price =np.max(np.vstack([ St[dynamic_index, :] - K, np.zeros(St.shape[1])]), axis = 0)
            elif type == 'Put' : 
                expiration_price =np.max( np.vstack([K - St[dynamic_index, :], np.zeros(St.shape[1])]), axis = 0)
        except : 
            print('Error')
        return expiration_price

    option_prices = get_Option_Price(SelectedStrike, simulation_paths, trade_type )
    pl_results = option_prices - option_purchase_price + 2 * transaction_cost
    relative_pl_results = pl_results / (option_purchase_price + 2 *transaction_cost)

    otm_probability = round(sum(option_prices == 0) / len(option_prices), 2)
    itm_probability = round(1 - otm_probability, 2)

    positive_pl_proba = round(sum(pl_results > 0 ) / len(pl_results), 2)

    st.subheader('Results')

    t32_col1, t32_col2, t32_col3 = st.columns(3)
    t32_col1.metric("In-the money probability", itm_probability, border = True)
    t32_col2.metric("Out-the money probability", otm_probability,border = True )
    t32_col3.metric("Positive P&L probability", positive_pl_proba,border = True )
    #### Plots

    t33_col1, t33_col2 = st.columns(2)
    with t33_col1 : 
        price_distribution =simulation_paths[ - int(step - timeshot * step + 1), :] 
        ## transformación logarimica para checar distribuciones (ocpiona)
        ###log_transform = np.log(price_distribution)
        t3_fig1 = plt.figure(figsize=(8, 8))
        sns.histplot(price_distribution, kde = True, stat= 'probability')
        plt.xlabel('Price')
        plt.axvline(SelectedStrike, 0,1, color = 'r', label = 'Strike price')
        plt.title(f'Expected underlying asset price distribution at day {int(timeshot * 365)}')
        plt.legend()
        st.pyplot(t3_fig1)

    with t33_col2 : 
    
        t3_fig2 = plt.figure(figsize=(8, 3))
        sns.histplot(option_prices, kde = True, stat= 'probability')
        plt.xlabel('Price')

        plt.title(f'Expected {trade_type} premium at day {int(timeshot * 365)}')
        plt.legend()
        st.pyplot(t3_fig2)

        t3_fig3 = plt.figure(figsize=(8, 3))
        sns.histplot(pl_results, kde = True, stat= 'probability')
        plt.xlabel('Price')

        plt.title(f'Expected relative P&L distribution at day {int(timeshot * 365)}')
        plt.legend()
        st.pyplot(t3_fig3)
        
    expiration_price =np.max(np.vstack([simulation_paths - SelectedStrike, np.zeros(simulation_paths.shape[1])]), axis = 0)
    
    



    ### Matriz de los resultados de cada simulación
    call_prices = np.maximum(simulation_paths - SelectedStrike, np.zeros(simulation_paths.shape))
    p_and_l = call_prices - option_purchase_price
    relative_pl = p_and_l / option_purchase_price
    
    d1 = (np.log(Underlying_price / SelectedStrike) + (Risk_Free_Rate + volatility**2 / 2) * relative_maturity_time) / (volatility * np.sqrt(relative_maturity_time))

    st.write(d1)

    st.write(volatility)
    st.write(round(norm.cdf(d1, 0, 1),2))
    st.write(round(norm.cdf(d1 - (volatility * np.sqrt(relative_maturity_time)), 0, 1),2))
    ####


    ##Esperanza del retorno relativo
    E_r = np.log(Underlying_price) + (Risk_Free_Rate + volatility**2 / 2) * relative_maturity_time - np.log(SelectedStrike + option_purchase_price + 2 * transaction_cost)
    
    d1_r =  (-E_r)  / (volatility * np.sqrt(relative_maturity_time))
    st.write(round(norm.cdf(d1_r, 0, 1),2))