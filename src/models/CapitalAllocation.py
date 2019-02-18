import numpy as np
from src.visualization.visualize import *
from src.models.Ensemble import *

def constantAllocation(returns, days):
    '''
    Returns mean and std of investment allocated over number of days
    '''
    dailyAlloc = 1/days
    ARdist = []
    for it in range(10000):
        sampled_returns = np.random.choice(returns, 256)
        investment = 0
        toAllocate = 1
        daysRemaining = days
        for t in range(256):
            if daysRemaining > 0: # Keep allocating
                investment = investment + dailyAlloc
                daysRemaining = daysRemaining - 1
            investment = investment * (1+sampled_returns[t])
            if investment <= 0:
                break
        investment = investment
        ARdist.append(investment)
    #plotHisto(ARdist, label = "AR", bins=30)
    return np.mean(ARdist), np.std(ARdist)

def constantAllocationPlot(seriesInput):
    plt.style.use('default')
    seriesFull = seriesInput.copy()
    ensemble_str = ensembleStrategy(seriesFull)
    ensemble_str.apply(model = 'Ensemble')
    returns = ensemble_str.print()
    returns = returns.loc[404:576, 'DailyReturn']
    returns = returns.dropna()
    x = [1, 2, 3, 6, 10, 20, 40, 80, 160]
    y = []
    yerr = []
    for i in x:
        mean, std = constantAllocation(returns, i)
        y.append(mean)
        yerr.append(std)
    plt.errorbar(x, y, yerr, fmt='o')
    plt.xlabel("Days")
    plt.ylabel("Capital/$10M")

    plt.show()
