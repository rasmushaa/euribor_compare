import streamlit as st
import numpy as np
import pandas as pd
import jax
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# Fonts
plt.style.use('seaborn-v0_8-darkgrid')
plt.rc('xtick', labelsize=13)
plt.rc('axes', labelsize=13)
plt.rc('axes', titlesize=13)
plt.rc('legend', fontsize=13)

st.title('Compare different Euribor rates, using historical data!')

st.subheader('Please, provide your own loan parameters')
LOAN_AMOUNT = st.number_input('Specify the loan amount [€]', 100000, 500000, 200000, 10000, key="1")
MATURITY = st.number_input('Specify term of the loan [Years]', 5, 25, 20, 1, key="2")
MARGIN_12 = st.number_input('Specify the margin for Euribor(12Mo) rate [%]', 0.0, 2.0, 0.4, 0.01, key="3")
MARGIN_3 = st.number_input('Specify the margin forEuribor(3Mo) rate [%]', 0.0, 2.0, 0.5, 0.01, key="4")
STATE = st.button("Run analysis!")


def load_data():
    df = pd.read_csv('euribor_rates.csv', index_col='date')
    df.index = pd.to_datetime(df.index)
    return df


def plot_historical_rates(df):
    plt.figure('Historical Euribor rates', figsize=(10, 5))
    plt.clf()

    plt.plot(df['euribor_12_month'], color='red', label='Euribor(12Mo)')
    plt.plot(df['euribor_6_month'], color='blue', label='Euribor(3Mo)')
    plt.plot(df['euribor_3_month'], color='purple', label='Euribor(1Mo)')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y')) 
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=360))
    plt.gca().xaxis.set_tick_params(rotation = 45) 

    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Interest Rate [%]")
    plt.title(f'Historical daily Euribor rates with {len(df)} data points.')
    plt.tight_layout()
    st.pyplot(plt.gcf())


def monthly_payment_annuety(N, r, margin, T):
    '''
    N loan amoun, r interest rate, maring of loan, T loan maturity in years
    '''
    r += margin
    if r < margin:
        r = margin + 0.01
    return ((1 + r/12 / 100)**(T*12) * (r/12 / 100)) / ((1 + r/12 / 100)**(T*12) - 1) * N

def backtesting(df, rate_type, N, T):
    '''
    rate_type euribor [int], N loan amoun, T loan maturity in years
    '''
    if rate_type == 12:  period = 360; rate_col = 0; margin = MARGIN_12
    elif rate_type == 3: period = 90;  rate_col = 2; margin = MARGIN_3
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
        origination_date_list.append(df.index.values[i])
        payments = 0

        for j, rate in enumerate(rates):
            payment = monthly_payment_annuety(N, rate, margin, T)
            payments += payment*rate_type
            payments_list += [payment]*rate_type
        total_payments_list.append(payments)

    return total_payments_list, origination_date_list, payments_list, dates_list


def plot_backtesting(df):
    plt.figure('Backtesting', figsize=(10, 5))
    plt.clf()

    toal_payments, origination_dates, payments, dates = backtesting(df, 12, LOAN_AMOUNT, MATURITY)
    plt.plot(origination_dates, toal_payments, color='blue', label='Euribor(12Mo)')
    plt.axhline(np.mean(toal_payments), c='blue', linestyle='--', alpha=0.5, label=f'Mean {np.mean(toal_payments)/1000:.0f}k€')

    toal_payments, origination_dates, payments, dates = backtesting(df, 3, LOAN_AMOUNT, MATURITY)
    plt.plot(origination_dates, toal_payments, color='red', label='Euribor(3Mo)')
    plt.axhline(np.mean(toal_payments), c='red', linestyle='--', alpha=0.5, label=f'Mean {np.mean(toal_payments)/1000:.0f}k€')

    plt.gca().xaxis.set_tick_params(rotation = 40) 
    plt.legend()
    plt.xlabel("Loan origination date")
    plt.ylabel("Total Payments")
    plt.title(f'Backtesting a {MATURITY} year loan of {LOAN_AMOUNT}€, with your margins')
    plt.tight_layout()
    st.pyplot(plt.gcf())


def plot_var(df):
    plt.figure('VaR', figsize=(10, 6))
    plt.clf()

    plt.subplot(2, 1, 1)
    toal_payments, origination_dates, payments, dates = backtesting(df, 12, LOAN_AMOUNT, MATURITY)
    plt.hist(payments, 50, color='blue', edgecolor='white', alpha=0.5, label='Euribor(12Mo)')
    mean = np.mean(payments)
    plt.axvline(mean, c='blue', linestyle='--', label=f'Mean {mean:.0f}€')
    q95 = np.percentile(payments, 95)
    plt.axvline(q95, c='grey', linestyle='--', label=f'VaR(95) {q95:.0f}€')
    payments = np.array(payments)
    es = np.mean(payments[payments >= q95])
    plt.axvline(es, c='grey', linestyle='-.', label=f'ES(95)   {es:.0f}€')
    plt.legend()
    plt.xlabel("Monthly payment [€]")
    plt.ylabel("Density")

    plt.subplot(2, 1, 2)
    toal_payments, origination_dates, payments, dates = backtesting(df, 3, LOAN_AMOUNT, MATURITY)
    plt.hist(payments, 50, color='red', edgecolor='white', alpha=0.5, label='Euribor(3Mo)')
    mean = np.mean(payments)
    plt.axvline(mean, c='red', linestyle='--', label=f'Mean {mean:.0f}€')
    q95 = np.percentile(payments, 95)
    plt.axvline(q95, c='grey', linestyle='--', label=f'VaR(95) {q95:.0f}€')
    payments = np.array(payments)
    es = np.mean(payments[payments >= q95])
    plt.axvline(es, c='grey', linestyle='-.', label=f'ES(95)   {es:.0f}€')
    plt.legend()
    plt.xlabel("Monthly payment [€]")
    plt.ylabel("Density")

    plt.suptitle(f'Possible monthly payments and VaR(95%)')
    plt.tight_layout()
    st.pyplot(plt.gcf())


def plot_payment_interval(df):
    plt.figure('Possible payments', figsize=(10, 5))
    plt.clf()
    # Euribor 12
    toal_payments, origination_dates, payments, dates = backtesting(df, 12, LOAN_AMOUNT, MATURITY)
    plot_df = pd.DataFrame({'date': dates, 'payments': payments})
    plot_df = plot_df.groupby(['date']).agg(median=('payments', 'median'), 
                                            lower_95=('payments', lambda x: np.quantile(x, q=0.025)), 
                                            upper_95=('payments', lambda x: np.quantile(x, q=0.975))).reset_index()

    plt.fill_between(plot_df['date'], plot_df['lower_95'], y2=plot_df['upper_95'], color='blue', alpha=0.5, label='Euribor(12Mo)\n95% CI')
    # Euribor 3
    toal_payments, origination_dates, payments, dates = backtesting(df, 3, LOAN_AMOUNT, MATURITY)
    plot_df = pd.DataFrame({'date': dates, 'payments': payments})
    plot_df = plot_df.groupby(['date']).agg(median=('payments', 'median'), 
                                            lower_95=('payments', lambda x: np.quantile(x, q=0.025)), 
                                            upper_95=('payments', lambda x: np.quantile(x, q=0.975))).reset_index()

    plt.fill_between(plot_df['date'], plot_df['lower_95'], y2=plot_df['upper_95'], color='red', alpha=0.5, label='Euribor(3Mo)\n95% CI')

    plt.gca().xaxis.set_tick_params(rotation = 40) 
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Monthly Payment [€]")
    plt.title(f'95% Confidence Interval of possible payments')
    plt.tight_layout()
    st.pyplot(plt.gcf())


def plot_historical_difference(df):
    plt.figure('Historical difference', figsize=(10, 5))
    plt.clf()
    df['euribor_3_month'] = df['euribor_3_month'] + MARGIN_3
    df['euribor_12_month'] = df['euribor_12_month'] + MARGIN_12
    interest_diff = df['euribor_3_month'] - df['euribor_12_month']

    n, bins, patches = plt.hist(interest_diff.values , 50, color='purple', edgecolor='white', alpha=0.5, label='Data')
    z = np.linspace(0.0, 1.0, len(bins))
    for i in range(len(bins[bins<=0])-1, len(bins)-1):
        patches[i].set_alpha(0.3)
    mean = np.mean(interest_diff.values)
    plt.axvline(mean, c='purple', linestyle='--', label=f'Mean {mean:.2f}%')

    plt.legend()
    plt.xlabel("Interest rate difference")
    plt.ylabel("Density")
    plt.title(f'Historical difference between Euribor(3Mo) and Euribor(12Mo) rates, including margins')
    st.pyplot(plt.gcf())


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
    df['euribor_3_month'] = df['euribor_3_month'] + MARGIN_3
    df['euribor_12_month'] = df['euribor_12_month'] + MARGIN_12
    interest_diff = df['euribor_3_month'] - df['euribor_12_month']
    data = {
    "true_events": interest_diff.loc[interest_diff < 0].count(),
    "false_events": interest_diff.loc[interest_diff >= 0].count()
    }
    sample = posterior_sampler(model_normal, data)

    plt.figure(figsize=(10, 5))
    plt.cla()
    plt.clf()
    plt.hist(sample['theta'], 50, color='purple', edgecolor='white', alpha=0.5, label='Theta')
    mean = np.mean(sample['theta'])
    plt.axvline(mean, c='purple', linestyle='--', label=f'Mean {mean*100:.0f}%')
    plt.legend()
    plt.xlabel("Propability")
    plt.ylabel("Density")
    plt.title('Results of Bayesian prediction for propability $\\theta$\nthat Euribor(3Mo) is less than Euribor(12Mo) for given data.')
    st.pyplot(plt.gcf())

    plt.figure('binomial', figsize=(10, 5))
    plt.cla()
    plt.clf()
    outcomes =[]
    for i in range(500):
        outcomes += np.random.binomial(n=4*MATURITY, p=sample['theta'][i], size=100).tolist()
    weights = np.ones_like(outcomes)/float(len(outcomes))
    bins = max(outcomes) - min(outcomes)
    plt.hist(outcomes, bins=bins, weights=weights, color='purple', edgecolor='white',  alpha=0.5)
    mean = np.mean(outcomes)
    plt.axvline(mean, c='purple', linestyle='--', label=f'Mean {mean:.1f}')
    q5 = np.percentile(outcomes, 5)
    plt.axvline(q5, c='grey', linestyle='--', label=f'VaR(95) {q5:.1f}')
    outcomes = np.array(outcomes)
    es = np.mean(outcomes[outcomes <= q5])
    plt.axvline(es, c='grey', linestyle='-.', label=f'ES(95)  {es:.1f}')
    plt.legend()
    plt.xlabel("Euribor(3Mo) is less this many times")
    plt.ylabel("Propability")
    plt.title(f"Given the distribution of $\\theta$,\nhow many times Euribor(3Mo) may be less than Euribor(12Mo)\nduring the lifetime of the loan ({4*MATURITY} quarters).")
    st.pyplot(plt.gcf())


if STATE == True:
    data = load_data()
    plot_historical_rates(data)
    plot_backtesting(data)
    plot_var(data)
    plot_payment_interval(data)
    plot_historical_difference(data)
    plot_bayesian_analysis(data)
