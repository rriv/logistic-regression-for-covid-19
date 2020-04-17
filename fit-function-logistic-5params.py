# -*- coding: utf-8 -*-
import pymc
import numpy as np
import matplotlib.pyplot as plt
import sys

#  repris de https://stackoverflow.com/questions/24804298/fit-a-non-linear-function-to-data-observations-with-pymcmc-pymc
# et fonction cible réimplémentée par une logistique

# on passe en arg le nbre de jours à examiner
j=int(sys.argv[1])

#### les données sont dans le code… 
# les 2 variables utilisées sont f et flabel.
# les f? et f?label sont d'autres jeux de données pour essais
# fr
# src : opencovid19-fr + extract-reg-opencovid2ICLcsv.py
flabel = u"Décès HORS EHPAD France 14/4 (centaines)"
f = np.array([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00
, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00
, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00
, 0.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00
, 1.0000e+00, 1.0000e+00, 1.0000e+00, 2.0000e+00, 2.0000e+00, 2.0000e+00
, 2.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00, 4.0000e+00, 5.0000e+00
, 9.0000e+00, 1.1000e+01, 1.9000e+01, 3.0000e+01, 3.3000e+01, 4.8000e+01
, 6.1000e+01, 7.9000e+01, 9.1000e+01, 1.2700e+02, 1.4800e+02, 1.7500e+02
, 2.4400e+02, 3.7200e+02, 4.5000e+02, 5.6200e+02, 6.7400e+02, 8.6000e+02
, 1.1000e+03, 1.3310e+03, 1.6960e+03, 1.9950e+03, 2.3140e+03, 2.6060e+03
, 3.0240e+03, 3.5230e+03, 4.0320e+03, 4.5030e+03, 5.0910e+03, 5.5320e+03
, 5.8890e+03, 6.4940e+03, 7.0910e+03, 7.6320e+03, 8.0440e+03, 8.5980e+03
, 8.9430e+03, 9.2530e+03, 9.5880e+03, 1.0129e+04]) / 100.
# src : opencovid19-fr csv
fflabel = u"Décès EN EHPAD France 15/4 (centaines)"
ff = np.array([371,
               884,
               1416,
               2028,
               2189,
               2417,
               3237,
               3237,
               4166,
               4599,
               4889,
               5140,
               5379,
               5600]) / 100.

# paca depuis opencovid19-fr
fplabel = u"Décès Paca 10/4 (dizaines)"
fp=  np.array([0,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              4,
              7,
              9,
              11,
              13,
              15,
              20,
              26,
              33,
              44,
              48,
              55,
              65,
              80,
              103,
              124,
              141,
              161,
              178,
              195,
              231,
              253,
              269,
              286,
              299])/10.

# décès Hubei https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv 29/3
fhlabel = u"Hubei Deaths (hundreds)"
fh = np.array([17, 17, 24, 40, 52, 76, 125, 125, 162, 204, 249, 350, 414, 479, 549, 618, 699, 780, 871, 974, 1068, 1068, 1310, 1457, 1596, 1696, 1789, 1921, 2029, 2144, 2144, 2346, 2346, 2495, 2563, 2615, 2641, 2682, 2727, 2761, 2803, 2835, 2871, 2902, 2931, 2959, 2986, 3008, 3024, 3046, 3056, 3062, 3075, 3085, 3099, 3111, 3122, 3130, 3133, 3139, 3153, 3153, 3160, 3163, 3169,3174,3177,3182]) / 100.

# usa src CSSEGISandData
fulabel = u"Décès USA 13/4 (milliers)"
fu = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,6,7,11,12,14,17,21,22,28,36,40,47,54,63,85,108,118,200,244,307,417,557,706,942,1209,1581,2026,2467,2978,3873,4757,5926,7087,8407,9619,10783,12722,14695,16478,18586,20463,22020,23529]) /1000.

# italie src CSSEGISandData
filabel = u"Décès Italie 13/4 (milliers)"
fi = np.array([0,0,0,0,0,0,0,0,1,2,3,7,10,12,17,21,29,34,52,79,107,148,197,233,366,463,631,827,827,1266,1441,1809,2158,2503,2978,3405,4032,4825,5476,6077,6820,7503,8215,9134,10023,10779,11591,12428,13155,13915,14681,15362,15887,16523,17127,17669,18279,18849,19468,19899,20465]) / 1000.

# corée https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv
fclabel = u"Décès Corée (dizaines)"
fc = np.array([0,0,0,0,0,0,0,1,2,2,6,8,10,12,13,13,16,17,28,28,35,35,42,44,50,53,54,60,66,66,72,75,75,81,84,91,94,102,111,111,120,126,131,139,144,152,158])/10.
# src ?
fclabel = u"Cas Corée (milliers)"
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

#### fin des données

njours = f.shape[0]
forig = f ; norig = njours
print("Run : ", flabel, u"...", f[-5:] )
print( " N jours ",njours)
if (j != 0):
    njours=j
    f=f[:njours]
    print( "  tronques a ",njours)

f_error = np.ones_like(f)*0.05*f.max()

# define the model/function to be fitted.
def model(x, f):
    # priors uniformes
    # on est sur des pentes négatives
    pente = pymc.Uniform('pente', -10., -0.0001)
    amp   = pymc.Uniform('amp', 1., 200.)
    bias  = pymc.Uniform('bias',1,150)
    # constaté expérimentalement que q est > 0
    q     = pymc.Uniform('q',0.0001,200.)
    nu    = pymc.Uniform('nu',0.0001,5.)
    
    # foncion logistique à approximer
    # «généralisée» par ajout des coeffs q et nu.
    # selon les notations de https://en.wikipedia.org/wiki/Generalised_logistic_function
    #   A = 0, K = amp, B = pente et on prend C = 1
    @pymc.deterministic(plot=False)
    def lgs(x=x, pente=pente, amp=amp, bias=bias, q=q, nu=nu):
        return amp / ((1. + q * np.exp(pente * (x - bias))) ** (1. / nu))
    y = pymc.Normal('y', mu=lgs, tau=1.0/f_error**2, value=f, observed=True)
    return locals()

x = np.arange(0,njours)
MDL = pymc.MCMC(model(x,f))
map_ = pymc.MAP(MDL)

try:
    map_.fit()
    # différentes durées de sampling
    #MDL.sample(3e7,burn=3e7 - 50000, verbose=0)
    MDL.sample(5e5,burn=4.5e5)
    #MDL.sample(5e4,burn=4e4)
except OverflowError as err:
    print( 'Overflowed ', err)

# Rappel en fin de MCMC
print("Run : ", flabel, f[:5],u"…")
print( " N jours ",njours)
 
# extract and plot results
y_min = MDL.stats()['lgs']['quantiles'][2.5]
y_max = MDL.stats()['lgs']['quantiles'][97.5]
y_fit = MDL.stats()['lgs']['mean']

pente_fit = MDL.trace('pente')[:]
amp_fit = MDL.trace('amp')[:]
bias_fit = MDL.trace('bias')[:]
q_fit = MDL.trace('q')[:]
nu_fit = MDL.trace('nu')[:]

print("--pente",pente_fit.shape)

p025 = np.quantile(pente_fit,0.025) ; p975 = np.quantile(pente_fit,0.975) 
a025 = np.quantile(amp_fit,0.025) ; a975 = np.quantile(amp_fit,0.975) 
b025 = np.quantile(bias_fit,0.025) ; b975 = np.quantile(bias_fit,0.975)
print("--bias quantiles", j, b025, b975)

#if (j != 0):
#    sys.exit()


# distribs des 5 paramètres
plt.subplot(251)
plt.hist(amp_fit,bins=31, label='amp', color='b')
plt.legend()

plt.subplot(252)
plt.hist(pente_fit, bins=31, label='pente')
plt.legend()

plt.subplot(253)
plt.hist(bias_fit, bins=31, label='bias')
plt.legend()

plt.subplot(254)
plt.hist(q_fit, bins=31, label='q')
plt.legend()

plt.subplot(255)
plt.hist(nu_fit, bins=31, label='nu')
plt.legend()

plt.subplot(212)

## ici on prolonge la zone des observations

# redef de lgs() pour utilisation en dehors de model()
def lgs(x, pente, amp, bias, q, nu):
    return amp /  ((1. + q * np.exp(pente * (x - bias))) ** (1. / nu))

# on prolonge la fonction cible jusqu'à x=100
xetendu = np.arange(0,100)

# observations et projections
nsamples = pente_fit.shape[0]
# à régler selon la durée de sampling
#for i in range(-10000,0,6):
for i in range(-40000,0,25):
    # cumul modélisé
    yetendu = lgs(xetendu, pente_fit[nsamples+i], amp_fit[nsamples+i], bias_fit[nsamples+i],q_fit[nsamples+i],nu_fit[nsamples+i] )
    plt.plot(xetendu,yetendu, color='g',alpha=0.01)
    # flux journalier modélisé
    yp = yetendu[1:]
    ym = yetendu[:-1]
    plt.plot(xetendu[1:],(yp-ym)*10., color='r',alpha=0.01)

# flux journalier observé
plt.plot(x[1:],(f[1:]-f[:-1])*10.,color='b', marker='+', ls='None', ms=5, label=u'Flux /j Obs.. × 10 - '+flabel)
# flux journalier modélisé - pour la légende
plt.plot(0,0,color='r', marker='.', ls='None', ms=5, label=u'Flux /j Prév. × 10 - '+flabel)

# cumul observé NON pris en compte dans la régression
plt.plot(np.arange(0,norig),forig,'y', marker='.', ls='None', ms=5, label=u'Cas obs. cumulés - '+flabel)
# cumul observé pris en compte dans la régression - au dessus
plt.plot(x,f,'b', marker='.', ls='None', ms=5, label=u'Cas obs. cumulés - '+flabel)

plt.legend()
plt.show()
