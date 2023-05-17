import streamlit as st
import numpy as np
import pandas as pd
import jax
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# Fonts
plt.style.use('seaborn-v0_8-darkgrid')
plt.rc('xtick', labelsize=13)
plt.rc('axes', labelsize=13)
plt.rc('axes', titlesize=13)
plt.rc('legend', fontsize=13)

# Header
st.title('Compare different Euribor rates, using historical data!')
st.markdown(f"""Specify your loan parameters on the left and press the run button.
            The page is reloaded automatically after any changes on the parameters,
            which may take some time. This application is part of an article that you may find here: https://medium.com/@rasmushaa""")

# Sidebar
st.sidebar.subheader('Please, provide your own loan parameters')
LOAN_AMOUNT = st.sidebar.number_input('Specify the loan amount [€]', 100000, 500000, 200000, 10000, key="1")
MATURITY = st.sidebar.number_input('Specify term of the loan [Years]', 5, 24, 20, 1, key="2")
EURIBOR_1 = st.sidebar.selectbox("Choose the Euribor rate of the first loan [Months]", (12, 6, 3), 0, key="3")
MARGIN_1 = st.sidebar.number_input('Specify the margin of first loan [%]', 0.0, 2.0, 0.4, 0.01, key="4")
EURIBOR_2 = st.sidebar.selectbox("Choose the Euribor rate of the second loan [Months]", (12, 6, 3), 2, key="5")
MARGIN_2 = st.sidebar.number_input('Specify the margin of second loan [%]', 0.0, 2.0, 0.46, 0.01, key="6")
RUN = st.sidebar.button("Run analysis!")


@st.cache_data
def load_data():
    df = pd.read_csv('euribor_rates.csv', index_col='date')
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
    return df

@st.cache_data
def load_frames(df, maturity):
    LW = 1.5
    trace12 = go.Scatter(x=df.index,
                         y=df['euribor_12_month'],
                         mode='lines',
                         name='Euribor(Mo12)',
                         line=dict(width=LW, color='red')
                         )
    trace6 = go.Scatter(x=df.index,
                        y=df['euribor_6_month'],
                        mode='lines',
                        name='Euribor(Mo6)',
                        line=dict(width=LW, color='blue')
                        )
    trace3 = go.Scatter(x=df.index,
                        y=df['euribor_3_month'],
                        mode='lines',
                        name='Euribor(Mo3)',
                        line=dict(width=LW, color='purple')
                        )
    trace_animation = go.Scatter(x=[df.index[0]],
                                 y=[0],
                                 mode='lines',
                                 name='Loan term',
                                 line=dict(width=0),
                                 line_color='grey'
                        )
    traces = [trace12, trace6, trace3, trace_animation]
    
    # Animation frames
    data_min = df['euribor_12_month'].min()
    data_max = df['euribor_12_month'].max()
    frames = [dict(data=[dict(type='scatter',
                            x=df.index,
                            y=df['euribor_12_month']),
                        dict(type='scatter',
                            x=df.index,
                            y=df['euribor_6_month']),
                        dict(type='scatter',
                            x=df.index,
                            y=df['euribor_3_month']),
                        dict(type='scatter',
                            x=[df.index[k], df.index[k], df.index[k+maturity*365], df.index[k+maturity*365], df.index[k]],
                            y=[data_min, data_max, data_max, data_min,data_min],
                            fill="toself")
                        ]
                )for k  in  range(1, len(df.index)-maturity*365-1, int(20/maturity*30))]
    
    # Last frame
    frames.append(dict(data=[dict(type='scatter',
                            x=df.index,
                            y=df['euribor_12_month']),
                        dict(type='scatter',
                            x=df.index,
                            y=df['euribor_6_month']),
                        dict(type='scatter',
                            x=df.index,
                            y=df['euribor_3_month']),
                        dict(type='scatter',
                            x=[df.index[0]],
                            y=[0])
                        ]
                    )
                )
    return traces, frames


def plot_historical_rates(traces, frames):    
    layout = go.Layout(showlegend=False,
                    hovermode='x unified',
                    updatemenus=[
                                dict(
                                type='buttons', showactive=False,
                                y=1.05,
                                x=1.15,
                                xanchor='right',
                                yanchor='top',
                                pad=dict(t=0, r=10),
                                buttons=[
                                    dict(label='Play',
                                    method='animate',
                                    args=[None, 
                                        dict(frame=dict(duration=200, 
                                                    redraw=False),
                                                    transition=dict(duration=0),
                                                    fromcurrent=True,
                                                    mode='immediate')]
                                )]
                            )
                        ]              
                    )
    layout.update(xaxis=dict(autorange=True),
                  yaxis=dict(autorange=True),
                  hovermode='x unified',
                  yaxis_title="Interest rate [%]")
    fig = go.Figure(data=traces, frames=frames, layout=layout)

    st.markdown(f"""## Historical Euribor rates""")
    st.markdown(f"""The graph presented below illustrates the daily euribor rates spanning from 
                {traces[0].x.min():%Y-%m-%d} to {traces[0].x.max():%Y-%m-%d}.
                You can analyse rates by selecting a specific area of interest 
                and utilizing the tools located in the upper corner of the figure.
                By utilizing the play button, it is possible to demonstrates how your {MATURITY}-year loan 
                will be evaluated by sliding it across the dataset and 
                calculating potential payments for each loan origination date.""")
    st.plotly_chart(fig, use_container_width=True)


def monthly_payment_annuety(N, r, margin, T):
    '''
    N loan amount, r interest rate, maring of loan, T loan maturity in years
    '''
    r += margin
    if r < margin:
        r = margin + 0.01
    return ((1 + r/12 / 100)**(T*12) * (r/12 / 100)) / ((1 + r/12 / 100)**(T*12) - 1) * N

def backtesting(df, rate_type, N, T, margin):
    '''
    rate_type euribor [int], N loan amoun, T loan maturity in years
    '''
    if rate_type == 12:  period = 360;  rate_col = 0
    elif rate_type == 6: period = 180;  rate_col = 1
    elif rate_type == 3: period = 90;   rate_col = 2
    else:
        raise ValueError('Invalid rate type')

    payments_list = []
    total_payments_list = []
    origination_date_list = []
    dates_list = []
    maturity_days = T*360
    for i in range(len(df)):
        if i+maturity_days >= len(df):
            break

        indexes = np.arange(i, i+maturity_days-1, period)
        rates = df.iloc[indexes, rate_col]
        indexes = np.arange(i, i+maturity_days-1, 30)
        dates = df.iloc[indexes].index - pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1)
        dates = dates.values.astype('datetime64[D]').tolist()
        dates_list += dates
        origination_date_list.append(df.index[i])
        payments = 0

        for j, rate in enumerate(rates):
            payment = monthly_payment_annuety(N, rate, margin, T)
            payments += payment*rate_type
            payments_list += [payment]*rate_type
        total_payments_list.append(payments)

    return total_payments_list, origination_date_list, payments_list, dates_list


def plot_backtesting(df):
    fig = go.Figure()

    toal_payments, origination_dates, payments, dates = backtesting(df, EURIBOR_1, LOAN_AMOUNT, MATURITY, MARGIN_1)
    fig.add_trace(go.Scatter(x=origination_dates,
                             y=toal_payments,
                             mode='lines',
                             name=f'Loan 1',
                             line=dict(width=1.5),
                             line_color='blue'
                            ))
    fig.add_hline(y=np.mean(toal_payments), 
                  line_width=1.5, 
                  annotation_text=f'Mean {np.mean(toal_payments)/1000:.0f}k€',
                  line_dash='dash', 
                  line_color='blue',
                  opacity=0.5)
    
    toal_payments, origination_dates, payments, dates = backtesting(df, EURIBOR_2, LOAN_AMOUNT, MATURITY, MARGIN_2)
    fig.add_trace(go.Scatter(x=origination_dates,
                             y=toal_payments,
                             mode='lines',
                             name=f'Loan 2',
                             line=dict(width=1.5),
                             line_color='red'
                            ))
    fig.add_hline(y=np.mean(toal_payments), 
                  line_width=1.5, 
                  annotation_text=f'Mean {np.mean(toal_payments)/1000:.0f}k€',
                  line_dash='dash', 
                  line_color='red',
                  opacity=0.5)

    fig.update_layout(showlegend=False, 
                      hovermode='x unified',
                      yaxis_title="Total payments [€]")
    
    st.markdown(f"""## Total payments""")
    st.markdown(f"""The total payment amount is assessed through backtesting, wherein the loan is repaid in the usual manner, but the loan origination date is varied each time. 
                With a {MATURITY}-year loan, there are {len(df.index)-MATURITY*365-1} potential loan origination dates, and each date has an impact on the overall interest rate of the loan as it is readjusted based on the selected reference Euribor rate.
                For instance, you chose the Euribor(Mo{EURIBOR_2}) as the reference for the second loan, and if the loan is initiated on {df.index[0]:%Y-%m-%d}
                it will be readjusted on {df.index[EURIBOR_2*30*1]:%Y-%m-%d}, {df.index[EURIBOR_2*30*2]:%Y-%m-%d}, {df.index[EURIBOR_2*30*3]:%Y-%m-%d} and so forth. 
                Each position on the graph represents a unique possibility that would have occurred had the loan been initiated on the respective day.
                For the Euribor(Mo{EURIBOR_2}), the mean of all dates amounts to {np.mean(toal_payments)/1000:.0f}k€""")
    st.plotly_chart(fig, use_container_width=True)


def plot_var(df):
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.25)
    toal_payments, origination_dates, payments, dates = backtesting(df, EURIBOR_1, LOAN_AMOUNT, MATURITY, MARGIN_1)
    mean = np.mean(payments)
    q95 = np.percentile(payments, 95)
    payments = np.array(payments)
    es = np.mean(payments[payments >= q95])
    fig.add_trace(go.Histogram(x=payments,
                               name=f'Loan 1',
                               marker_color='blue'),
                               row=1, col=1)
    fig.add_vline(x=mean, 
                  line_width=1.5, 
                  annotation_text=f'Mean {mean:.0f}€',
                  line_dash='dash', 
                  line_color='blue',
                  opacity=0.5,
                  row=1, col=1)
    fig.add_vline(x=q95, 
                  line_width=1.5, 
                  annotation_text=f'VaR(95) {q95:.0f}€',
                  annotation_position='left',
                  line_dash='dash', 
                  line_color='grey',
                  opacity=0.5,
                  row=1, col=1)
    fig.add_vline(x=es, 
                  line_width=1.5, 
                  annotation_text=f'ES(95) {es:.0f}€',
                  line_dash='dashdot', 
                  line_color='grey',
                  opacity=0.5,
                  row=1, col=1)
    
    toal_payments, origination_dates, payments, dates = backtesting(df, EURIBOR_2, LOAN_AMOUNT, MATURITY, MARGIN_2)
    mean = np.mean(payments)
    q95 = np.percentile(payments, 95)
    payments = np.array(payments)
    es = np.mean(payments[payments >= q95])
    fig.add_trace(go.Histogram(x=payments,
                               name=f'Loan 2',
                               marker_color='red'),
                               row=2, col=1)
    fig.add_vline(x=mean, 
                  line_width=1.5, 
                  annotation_text=f'Mean {mean:.0f}€',
                  line_dash='dash', 
                  line_color='red',
                  opacity=0.5,
                  row=2, col=1)
    fig.add_vline(x=q95, 
                  line_width=1.5, 
                  annotation_text=f'VaR(95) {q95:.0f}€',
                  annotation_position='left',
                  line_dash='dash', 
                  line_color='grey',
                  opacity=0.5,
                  row=2, col=1)
    fig.add_vline(x=es, 
                  line_width=1.5, 
                  annotation_text=f'ES(95) {es:.0f}€',
                  line_dash='dashdot', 
                  line_color='grey',
                  opacity=0.5,
                  row=2, col=1)
    
    fig.update_layout(showlegend=True)
    fig.update_xaxes(title_text="Possible payments [€]", row=1, col=1)  
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text="Possible payments [€]", row=2, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)

    st.markdown(f"""## Monthly payments""")
    st.markdown(f"""The individual payments derived from the previous analysis are further examined using a histogram, which provides a visual representation of the payment distribution.
                Each bar in the histogram represents the probability of monthly repayment amounts, and the mean value signifies the expected payment.
                The Value At Risk (VaR) at a 95% confidence level indicates the worst-case scenario that would have occurred in 5% of the total payments, while the Expected Shortfall (ES) represents the average value of the VaR.
                In practical terms, for your Euribor(Mo{EURIBOR_2})
                it can be anticipated that the monthly payment amount would average around {mean:.0f}€.
                However, it is important to be prepared for potential fluctuations, 
                as there is a 5% chance that the monthly payment could reach up to {q95:.0f}€""")
    st.plotly_chart(fig, use_container_width=True)


def plot_payment_interval(df):
    fig = go.Figure()
    toal_payments, origination_dates, payments, dates = backtesting(df, EURIBOR_1, LOAN_AMOUNT, MATURITY, MARGIN_1)
    plot_df = pd.DataFrame({'date': dates, 'payments': payments})
    plot_df = plot_df.groupby(['date']).agg(median=('payments', 'median'), 
                                            lower_95=('payments', lambda x: np.quantile(x, q=0.025)), 
                                            upper_95=('payments', lambda x: np.quantile(x, q=0.975))).reset_index()
    
    fig.add_trace(go.Scatter(x=plot_df['date'],
                             y=plot_df['lower_95'],
                             mode='lines',
                             name=f'Loan 1 Lower',
                             line=dict(width=1.5),
                             line_color='blue',
                             opacity=0.5,
                             fill=None
                             ))
    fig.add_trace(go.Scatter(x=plot_df['date'],
                             y=plot_df['upper_95'],
                             mode='lines',
                             name=f'Loan 1 Upper',
                             line=dict(width=1.5),
                             line_color='blue',
                             opacity=0.5,
                             fill='tonexty'
                             ))
    
    toal_payments, origination_dates, payments, dates = backtesting(df, EURIBOR_2, LOAN_AMOUNT, MATURITY, MARGIN_2)
    plot_df = pd.DataFrame({'date': dates, 'payments': payments})
    plot_df = plot_df.groupby(['date']).agg(median=('payments', 'median'), 
                                            lower_95=('payments', lambda x: np.quantile(x, q=0.025)), 
                                            upper_95=('payments', lambda x: np.quantile(x, q=0.975))).reset_index()
    
    fig.add_trace(go.Scatter(x=plot_df['date'],
                             y=plot_df['lower_95'],
                             mode='lines',
                             name=f'Loan 2 Lower',
                             line=dict(width=1.5),
                             line_color='red',
                             opacity=0.5,
                             fill=None
                             ))
    fig.add_trace(go.Scatter(x=plot_df['date'],
                             y=plot_df['upper_95'],
                             mode='lines',
                             name=f'Loan 2 Upper',
                             line=dict(width=1.5),
                             line_color='red',
                             opacity=0.5,
                             fill='tonexty'
                             ))

    fig.update_layout(showlegend=False, hovermode='x unified')
    fig.update_yaxes(title_text="Monthy Payment [€]")

    st.markdown(f"""## Confidence Interval""")
    st.markdown(f"""The figure presented here analyzes monthly payments by plotting them along the actual date axes. 
                To create the 95% confidence interval, all possible values from each loan origination date are plotted. 
                In simple terms, the graph illustrates the range within which your monthly payment would have fallen on a given date, 
                considering the 95% confidence interval and varying loan origination dates. 
                It is worth noting that longer Euribor rates result in wider intervals, which can extend above or below those of shorter rates.""")
    st.plotly_chart(fig, use_container_width=True)


def plot_historical_difference(df):
    df1 = df[f'euribor_{EURIBOR_1}_month'] + MARGIN_1
    df2 = df[f'euribor_{EURIBOR_2}_month'] + MARGIN_2
    interest_diff = df1 - df2

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=interest_diff.values,
                               name=f'Historical Difference',
                               marker_color='purple',
                               opacity=0.5,
                               marker=dict(line=dict(width=0.8, color="white"))))
    
    mean = np.mean(interest_diff.values)
    fig.add_vline(x=mean, 
                  line_width=1.5, 
                  annotation_text=f'Mean {mean:.2f}%',
                  line_dash='dash', 
                  line_color='purple',
                  opacity=1.0)
    
    fig.update_layout(showlegend=False, hovermode='x unified')
    fig.update_xaxes(title_text="Interest rate difference [%]")
    fig.update_yaxes(title_text="Density")

    st.markdown(f"""## Historical difference""")
    st.markdown(f"""The distribution depicted in the graph showcases the historical disparity between 
                Euribor(Mo{EURIBOR_1}) and Euribor(Mo{EURIBOR_2}), considering the specified margins.
                 In general, shorter rates are expected to be lower than longer rates. 
                 However, there may be instances where data points are located on both sides of the y-axis, 
                 indicating deviations from this general trend""")
    st.plotly_chart(fig, use_container_width=True)


def model_normal(true_events, false_events):
    theta = numpyro.sample('theta', dist.Beta(1, 1))
    y = numpyro.sample('y', dist.Binomial(probs=theta, total_count=true_events+false_events), obs=true_events)

def posterior_sampler(model_function, data):
    algo = NUTS(model_function)
    sampler = MCMC(algo, num_warmup=1000, num_samples=10000, num_chains=1)
    seed = jax.random.PRNGKey(0)
    sampler.run(seed, **data) 
    sample = sampler.get_samples()
    sampler.print_summary()
    return sample

def plot_bayesian_analysis(df):
    df1 = df[f'euribor_{EURIBOR_1}_month'] + MARGIN_1
    df2 = df[f'euribor_{EURIBOR_2}_month'] + MARGIN_2
    interest_diff = df1 - df2
    data = {
    "true_events": interest_diff.loc[interest_diff > 0].count(),
    "false_events": interest_diff.loc[interest_diff <= 0].count()
    }
    sample = posterior_sampler(model_normal, data)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=sample['theta'],
                               name=f'Historical Difference',
                               marker_color='purple',
                               opacity=0.5,
                               marker=dict(line=dict(width=0.8, color="white"))))
    
    mean = np.mean(sample['theta'])
    fig.add_vline(x=mean, 
                  line_width=1.5, 
                  annotation_text=f'Mean {mean*100:.0f}%',
                  line_dash='dash', 
                  line_color='purple',
                  opacity=1.0)
    
    fig.update_layout(showlegend=False, hovermode='x unified')
    fig.update_xaxes(title_text="Probability")
    fig.update_yaxes(title_text="Density")

    st.markdown(f"""## Bayesian analysis""")
    st.markdown(f"""Bayesian analysis is a statistical approach that updates our beliefs about a hypothesis or parameter using prior knowledge and new data.
                It combines prior information with the likelihood of observing the data to obtain a posterior distribution, which represents the updated belief. 
                This method allows for the quantification of uncertainty and provides a flexible framework for decision-making and inference in various fields.
                In this particular test, Bayesian analysis was employed to estimate the probability, denoted as $\\theta$, that 
                Euribor(Mo{EURIBOR_2}) is less than Euribor(Mo{EURIBOR_1}). 
                A uniform Beta distribution was chosen as the prior distribution for theta, given that the historical difference between the two rates does not follow a specific common distribution.
                The posterior distribution, which results from this analysis, represents the updated belief regarding $\\theta$. 
                It indicates that Euribor(Mo{EURIBOR_2}) is, on average, expected to be lower than Euribor(Mo{EURIBOR_1}) with a mean propability of {mean*100:.0f}%, while taking into account the specified margins.""")
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    outcomes =[]
    for i in range(500):
        outcomes += np.random.binomial(n=EURIBOR_2*MATURITY, p=sample['theta'][i], size=100).tolist()
    fig.add_trace(go.Histogram(x=outcomes,
                               name=f'Historical Difference',
                               marker_color='purple',
                               opacity=0.5,
                               marker=dict(line=dict(width=0.8, color="white")),
                               histnorm='percent'))

    mean = np.mean(outcomes)
    fig.add_vline(x=mean, 
                  line_width=1.5, 
                  annotation_text=f'Mean {mean:.1f}',
                  line_dash='dash', 
                  line_color='purple',
                  opacity=1.0)

    q5 = np.percentile(outcomes, 5)
    fig.add_vline(x=q5, 
                  line_width=1.5, 
                  annotation_text=f'VaR(95) {q5:.1f}',
                  line_dash='dash', 
                  line_color='grey',
                  opacity=1.0)

    outcomes = np.array(outcomes)
    es = np.mean(outcomes[outcomes <= q5])
    fig.add_vline(x=es, 
                line_width=1.5, 
                annotation_text=f'ES(95)  {es:.1f}',
                annotation_position='left',
                line_dash='dashdot', 
                line_color='grey',
                opacity=1.0)
    
    fig.update_layout(showlegend=False, hovermode='x unified')
    fig.update_xaxes(title_text="Times")
    fig.update_yaxes(title_text="Probability [%]")

    st.markdown(f"""## Possible outcome""")
    st.markdown(f"""Your second loan utilizes Euribor(Mo{EURIBOR_2}) as a reference interest rate and spans a term {MATURITY} years.
                Consequently, it will undergo a total of {EURIBOR_2*MATURITY} readjustments throughout the loan's duration. 
                To model the likelihood of Euribor(Mo{EURIBOR_2}) being lower than Euribor(Mo{EURIBOR_1}), 
                the binomial distribution is employed since the outcome is a binary variable. 
                Random samples are drawn from the $\\theta$ distribution to account for data uncertainty. 
                The resulting distribution is presented below, indicating that based on historical data, an average of {mean:.1f}
                occurrences out of the {EURIBOR_2*MATURITY} can be expected where the Euribor(Mo{EURIBOR_2}) is the preferable choice. 
                However, in the worst-case scenario, there may be as few as {q5:.1f} occurrences.""")
    st.plotly_chart(fig, use_container_width=True)



# Load data
data = load_data()
traces, frames = load_frames(data, MATURITY)

if RUN == True:
    plot_historical_rates(traces, frames)
    plot_backtesting(data)
    plot_var(data)
    plot_payment_interval(data)
    plot_historical_difference(data)
    plot_bayesian_analysis(data)
