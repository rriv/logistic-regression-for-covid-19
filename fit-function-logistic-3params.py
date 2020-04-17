# -*- coding: utf-8 -*-
import pymc
import numpy as np
import matplotlib.pyplot as plt
import sys

#  repris de https://stackoverflow.com/questions/24804298/fit-a-non-linear-function-to-data-observations-with-pymcmc-pymc
# et fonction cible réimplémentée par une logistique

# on passe en arg le nbre de jours à examiner
j=int(sys.argv[1])

# fr
flabel = "France (milliers)"
fcontamines = np.array([0, 0.002, 0.003, 0.003, 0.003, 0.004, 0.005, 0.005, 0.005, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.014, 0.018, 0.038, 0.057, 0.1, 0.13, 0.191, 0.204, 0.285, 0.377, 0.653, 0.949, 1.126, 1.209, 1.784, 2.281, 2.281, 3.661, 4.469, 4.499, 6.633, 7.652, 9.043, 10.871, 12.612, 14.459, 16.018, 19.856,22.302,25.233, 29.155, 32.964])
# décès https://fr.wikipedia.org/wiki/Pand%C3%A9mie_de_maladie_%C3%A0_coronavirus_de_2020_en_France 27/3
flabel = u"Décès France (centaines)"
f = np.array([1,1,1,1,1,2,2,4,4,7,9,16,21,25,33,48,61,79,91,127,148,175,244,372,450,562,674,860,1100,1331,1696, 1995]) / 100.

# décès Hubei https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv 26/3
fhlabel = "Hubei (centaines)"
fh = np.array([17, 17, 24, 40, 52, 76, 125, 125, 162, 204, 249, 350, 414, 479, 549, 618, 699, 780, 871, 974, 1068, 1068, 1310, 1457, 1596, 1696, 1789, 1921, 2029, 2144, 2144, 2346, 2346, 2495, 2563, 2615, 2641, 2682, 2727, 2761, 2803, 2835, 2871, 2902, 2931, 2959, 2986, 3008, 3024, 3046, 3056, 3062, 3075, 3085, 3099, 3111, 3122, 3130, 3133, 3139, 3153, 3153, 3160, 3163, 3169]) / 100.

# it
fi = np.array([0.003,
              0.02,
              0.079,
              0.152,
              0.229,
              0.322,
              0.4,
              0.65,
              0.888,
              1.128,
              1.694,
              2.036,
              2.502,
              3.089,
              3.858,
              4.636,
              5.883,
              7.375,
              9.172,
              10.149,
              12.464,
              15.113,
              17.66,
              21.157,
              24.747,
              27.98,
              31.506,
              35.713,
              41.035,
              47.021,
              53.578,
              59.158,
              63.927,
              69.176])

# corée
fc = np.array([0.03,
              0.031,
              0.051,
              0.104,
              0.204,
              0.433,
              0.602,
              0.833,
              0.977,
              1.261,
              1.766,
              2.337,
              3.15,
              4.212,
              4.812,
              5.328,
              5.766,
              6.284,
              6.593,
              7.041,
              7.134,
              7.382,
              7.513,
              7.755,
              7.869,
              7.979,
              8.086,
              8.162,
              8.236,
              8.32,
              8.413,
              8.565,
              8.652,
              8.799,
              8.897,
              8.961,
              9.037])

njours = f.shape[0]
print( "N jours ",njours)
if (j != 0):
    njours=j
    f=f[:njours]
    print( "  tronques a ",njours)

f_error = np.ones_like(f)*0.05*f.max()



# define the model/function to be fitted.
def model(x, f):
    # priors uniformes
    pente = pymc.Uniform('pente', -10., 3.)
    amp = pymc.Uniform('amp', 1., 200.)
    bias = pymc.Uniform('bias',1,150)

    # foncion logistique à approximer
    @pymc.deterministic(plot=False)
    def lgs(x=x, pente=pente, amp=amp, bias=bias):
        return amp / (1. + np.exp(pente * (x - bias)))
    #print("+++",f_error,f,x)
    y = pymc.Normal('y', mu=lgs, tau=1.0/f_error**2, value=f, observed=True)
    return locals()

x = np.arange(0,njours)
MDL = pymc.MCMC(model(x,f))
map_ = pymc.MAP(MDL)
map_.fit()

try:
    MDL.sample(5e5,burn=4.5e5)
    #MDL.sample(5e4,burn=4e4)
except OverflowError as err:
    print( 'Overflowed ', err)
   
# extract and plot results
y_min = MDL.stats()['lgs']['quantiles'][2.5]
y_max = MDL.stats()['lgs']['quantiles'][97.5]
y_fit = MDL.stats()['lgs']['mean']

## ici on prolonge la zone des observations
pente_fit = MDL.trace('pente')[:]
amp_fit = MDL.trace('amp')[:]
bias_fit = MDL.trace('bias')[:]
print("--pente",pente_fit.shape)

p025 = np.quantile(pente_fit,0.025) ; p975 = np.quantile(pente_fit,0.975) 
a025 = np.quantile(amp_fit,0.025) ; a975 = np.quantile(amp_fit,0.975) 
b025 = np.quantile(bias_fit,0.025) ; b975 = np.quantile(bias_fit,0.975)
print("--bias quantiles", j, b025, b975)

#if (j != 0):
#    sys.exit()
    
# redef de lgs() pour utilisation en dehors de model()
def lgs(x, pente, amp, bias):
    return amp / (1. + np.exp(pente * (x - bias)))
# on prolonge la fonction cible jusqu'à x=100
xetendu = np.arange(0,100)

# pymc.Matplot.plot(MDL)

# distribs des 3 paramètres
plt.subplot(311)
plt.hist(amp_fit, bins=31, label='amp')
plt.subplot(312)
plt.hist(pente_fit, bins=31, label='pente')
plt.hist(bias_fit, bins=31, label='bias')
plt.legend()

plt.subplot(313)

# observations et projections
nsamples = pente_fit.shape[0]
for i in range(-3000,0,2):
    # cumul modélisé
    yetendu = lgs(xetendu, pente_fit[nsamples+i], amp_fit[nsamples+i], bias_fit[nsamples+i])
    plt.plot(xetendu,yetendu, color='g',alpha=0.01)
    # flux journalier modélisé
    yp = yetendu[1:]
    ym = yetendu[:-1]
    plt.plot(xetendu[1:],(yp-ym)*10., color='r',alpha=0.01)

# flux journalier observé
plt.plot(x[1:],(f[1:]-f[:-1])*10.,color='b', marker='+', ls='None', ms=5, label=u'Flux /j Obs.. × 10 - '+flabel)
# flux journalier modélisé - pour la légende
plt.plot(0,0,color='r', marker='.', ls='None', ms=5, label=u'Flux /j Prév. × 10 - '+flabel)
# cumul observé
plt.plot(x,f,'b', marker='.', ls='None', ms=5, label=u'Cas obs. cumulés - '+flabel)

plt.legend()

plt.show()
