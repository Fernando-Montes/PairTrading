import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde

plt.style.use('default')

def firstLook(series):
    plt.rcParams["figure.figsize"] = (10,8)
    fig, sub = plt.subplots(3, 1, sharex=True)
    sub[0].plot(series['ABC'], linewidth=1.0, label = 'ABC')
    sub[0].plot(series['XYZ'], linewidth=1.0, label = 'XYZ')
    sub[0].grid('on', linestyle='--')
    sub[0].legend()
    sub[1].plot(series['RATIO'], linewidth=1.0)
    sub[1].set_ylabel("Ratio")
    sub[1].grid('on', linestyle='--')
    sub[2].plot(series['DIFF'], linewidth=1.0)
    sub[2].set_xlabel("Days")
    sub[2].set_ylabel("Diff")
    sub[2].grid('on', linestyle='--')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.show()

def rollingEstimates(series, window):
    '''
    Plots rolling estimates of mean, standard deviations, Pearson's coefficients
    '''

    plt.rcParams["figure.figsize"] = (10,8)
    fig, sub = plt.subplots(3, 1, sharex=True)
    rolling_mean = series['RATIO'].rolling(window).mean()
    rolling_std = series['RATIO'].rolling(window).std()
    df = pd.DataFrame()
    df['RATIO'] = series['RATIO']
    df['RATIO_mean'] = rolling_mean
    df['RATIO_meanHI1'] = rolling_mean + (rolling_std)
    df['RATIO_meanLO1'] = rolling_mean - (rolling_std)
    df['RATIO_meanHI2'] = rolling_mean + 2*(rolling_std)
    df['RATIO_meanLO2'] = rolling_mean - 2*(rolling_std)
    sub[0].plot(series['RATIO'], linewidth=1.0, label = 'Ratio')
    sub[0].plot(df['RATIO_mean'], label='Mean')
    sub[0].plot(df['RATIO_meanHI1'], linestyle ='--', c='r', label='1-std')
    sub[0].plot(df['RATIO_meanLO1'], linestyle ='--', c='r', label='_nolegend_')
    sub[0].plot(df['RATIO_meanHI2'], linestyle ='--', c='g', label='2-std')
    sub[0].plot(df['RATIO_meanLO2'], linestyle ='--', c='g', label='_nolegend_')
    for fit_range in range(30, 121, 20):
        days = []
        pearson = []
        for i in range(fit_range,len(series)):
            days.append(i)
            pearson.append( pearsonr( series.loc[(i-fit_range):i,'ABC'],
                                      series.loc[(i-fit_range):i,'XYZ'] )[0] )
        sub[2].plot(days, pearson, linewidth=1.0, label=str(fit_range))
    sub[0].grid('on', linestyle='--')
    sub[0].legend()
    df['RATIO_stdUnits'] = (series['RATIO']-rolling_mean)/rolling_std
    sub[1].plot(df['RATIO_stdUnits'])
    sub[1].axhline(y=1, linestyle ='--', c='r')
    sub[1].axhline(y=-1, linestyle ='--', c='r')
    sub[1].axhline(y=2, linestyle ='--', c='g')
    sub[1].axhline(y=-2, linestyle ='--', c='g')
    sub[1].grid('on', linestyle='--')
    sub[1].set_ylabel("Ratio (XYZ/ABC)/std")
    sub[2].legend(title='Correlation window [days]')
    sub[2].set_xlabel("Days")
    sub[2].set_ylabel("Pearson's correlation coefficient")
    sub[2].set_ylim(0.85,1.01)
    sub[2].grid('on', linestyle='--')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.show()

def plotStrategy(df):
    '''
    Plots Entry and exit points based on given strategy
    '''
    try:
        plt.hold(False)
        plt.rcParams["figure.figsize"] = (10,8)
        fig, sub = plt.subplots(3, 1, sharex=True)
        sub[0].plot(df['RATIO_stdUnits'])
        sub[0].axhline(y=1, linestyle ='--', c='r')
        sub[0].axhline(y=-1, linestyle ='--', c='r')
        sub[0].axhline(y=2, linestyle ='--', c='g')
        sub[0].axhline(y=-2, linestyle ='--', c='g')
        sub[0].grid('on', linestyle='--')
        sub[0].set_ylabel("Ratio (XYZ/ABC)/std")
        sub[1].plot(df['Position'])
        sub[1].grid('on', linestyle='--')
        sub[1].set_ylabel("Positions")
        sub[1].axhline(y=1, linestyle ='--', c='r')
        sub[1].text(df.index[0]-5, 0.87, 'Entry long ABC. short XYZ', fontsize=9, color='r')
        sub[1].axhline(y=0, linestyle ='--', c='g')
        sub[1].text(df.index[0]-5, 0.05, 'Exit positions', fontsize=9, color='g')
        sub[1].axhline(y=-1, linestyle ='--', c='r')
        sub[1].text(df.index[0]-5, -0.95, 'Entry short ABC. long XYZ', fontsize=9, color='r')
        sub[2].plot(df['CumulativeCompoundedReturns'])
        sub[2].set_xlabel("Days")
        sub[2].set_ylabel("Cumulative Returns")
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        plt.show()
    except :
        plt.rcParams["figure.figsize"] = (5,4)
        plt.plot(df['CumulativeCompoundedReturns'])
        plt.xlabel("Days")
        plt.ylabel("Cumulative Returns")
        plt.show()

def plotAR(results):
    plt.rcParams["figure.figsize"] = (5,4)
    plt.scatter(results[:,3], results[:,4])
    plt.xlabel("AR train data [%]")
    plt.ylabel("AR test data [%]")
    plt.show()

def plotHisto(data, label, bins):
    plt.rcParams["figure.figsize"] = (5,4)
    plt.hist(data, bins = bins, histtype='step')
    plt.xlabel(label)
    plt.ylabel('Counts')

def plot_ratioFitting(results, xlim_lo=None, xlim_hi=None, ylim_lo=None, ylim_hi=None,
                               xres_lo=-0.2, xres_hi=0.2):
    plt.rcParams["figure.figsize"] = (10,8)
    fig, sub = plt.subplots(3, 1)
    results_comp = results.dropna()
    sub[0].plot(results['RATIO_stdUnits'], linewidth=1.0, label = 'Ratio/std')
    for i in range(6,len(results.columns)):
        sub[0].plot(results_comp[results.columns[i]], linewidth=1.0, label = results.columns[i])
        data = np.asarray(results_comp['RATIO_stdUnits']-results_comp[results_comp.columns[i]], dtype=np.float64)
        sub[1].plot(results_comp.index, data, label=results_comp.columns[i])
        density = gaussian_kde(data)
        density.covariance_factor = lambda : .1
        density._compute_covariance()
        xs = np.linspace(xres_lo, xres_hi, 100)
        sub[2].plot(xs, density(xs), label=results_comp.columns[i])
    sub[0].set_xlabel("Days")
    sub[0].set_ylabel("Ratio/std")
    sub[0].set_xlim(xlim_lo, xlim_hi)
    sub[0].set_ylim(ylim_lo, ylim_hi)
    sub[0].legend()
    sub[1].set_xlabel("Days")
    sub[1].set_ylabel("Residuals")
    sub[1].set_xlim(xlim_lo, xlim_hi)
    sub[1].legend()
    sub[2].legend()
    sub[2].set_xlabel("Residuals")
    fig.tight_layout()
    plt.show()

def summary(ensemble_str):
    x = ['Thr. entry and exit', 'Logistic Regression', 'Random Forest', 'RNN', 'LSTM', 'Ensemble all','Ens. RNN, LSTM', 'Ens. RNN, LSTM, Random Forest']
    y = []
    yerr = []
    ensemble_str.apply(model = 'BasicStrategy')
    mean, std = ensemble_str.ARdistribution(returnMetricOnly = True)
    y.append(mean)
    yerr.append(std)
    ensemble_str.apply(model = 'LogisticRegression')
    mean, std = ensemble_str.ARdistribution(returnMetricOnly = True)
    y.append(mean)
    yerr.append(std)
    ensemble_str.apply(model = 'RandomForest')
    mean, std = ensemble_str.ARdistribution(returnMetricOnly = True)
    y.append(mean)
    yerr.append(std)
    ensemble_str.apply(model = 'RNN')
    mean, std = ensemble_str.ARdistribution(returnMetricOnly = True)
    y.append(mean)
    yerr.append(std)
    ensemble_str.apply(model = 'LSTM')
    mean, std = ensemble_str.ARdistribution(returnMetricOnly = True)
    y.append(mean)
    yerr.append(std)
    ensemble_str.apply(model = 'Ensemble', includeRF = True, includeLR = True, includeBSC = True)
    mean, std = ensemble_str.ARdistribution(returnMetricOnly = True)
    y.append(mean)
    yerr.append(std)
    ensemble_str.apply(model = 'Ensemble')
    mean, std = ensemble_str.ARdistribution(returnMetricOnly = True)
    y.append(mean)
    yerr.append(std)
    ensemble_str.apply(model = 'Ensemble', includeRF = True)
    mean, std = ensemble_str.ARdistribution(returnMetricOnly = True)
    y.append(mean)
    yerr.append(std)
    #plt.rcParams['axes.labelsize'] = 6
    plt.rcParams['xtick.labelsize'] = 10
    plt.xticks(rotation=70)
    plt.errorbar(x, y, yerr, fmt='o')
    plt.ylabel("AR test data [%]")
    plt.show()
