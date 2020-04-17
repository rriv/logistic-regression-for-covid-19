# -*- coding: utf-8 -*-
import pymc
import numpy as np
import matplotlib.pyplot as plt
import sys

# version où on reconstitue rétrospectivement les valeurs manquantes AVANT la 1ère (cas des données EHPAD)
# la valeur à x[0] n'est pas 0
# xetendu va dans les négatifs
# on ne cherche pas à tracer les faisceaux de courbes mais la valeur la plus probable pour un x donné
# => yetendu est un tableau de dim 2 qu'on va slicer sur les colonnes pour en faire un histogramme

#  repris de https://stackoverflow.com/questions/24804298/fit-a-non-linear-function-to-data-observations-with-pymcmc-pymc
# et fonction cible réimplémentée par une logistique

# on passe en arg le nbre de jours à examiner
j=int(sys.argv[1])

#### les données sont dans le code… 
# les 2 variables utilisées sont f et flabel.
# les f? et f?label sont d'autres jeux de données pour essais
# fr

# src : opencovid19-fr csv
flabel = u"Décès EN EHPAD France 15/4 (centaines)"
f = np.array([371,
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
    #MDL.sample(5e5,burn=4.5e5)
    MDL.sample(5e4,burn=4e4)
except OverflowError as err:
    print( 'Overflowed ', err)

# Rappel en fin de MCMC
print("Run : ", flabel, f[:5],u"…")
print( " N jours ",njours)
 
# extract and plot results
y_min = MDL.stats()['lgs']['quantiles'][2.5]
y_max = MDL.stats()['lgs']['quantiles'][97.5]
y_fit = MDL.stats()['lgs']['mean']
## ici on prolonge la zone des observations
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

# on prolonge la fonction cible dans les négatifs
xetendu = np.arange(-40,20)

# observations et projections
nsamples = pente_fit.shape[0]
# à régler selon la durée de sampling
# initialisation du tableau où on mémorisera tous les yetendu - faudra pas oublier d'enlever la 1ère row qui est bidon
yetendu = a=np.array([xetendu])
for i in range(-10000,0,6):
#for i in range(-40000,0,25):
    # cumul modélisé
    # .r_[] : append row
    yetendu = np.r_[yetendu,[lgs(xetendu, pente_fit[nsamples+i], amp_fit[nsamples+i], bias_fit[nsamples+i],q_fit[nsamples+i],nu_fit[nsamples+i] )]]


# ne pas oublier d'enlever la 1ère row qui est bidon
# et on revient à des unités plutôt qu'aux centaines
yetendu = np.delete(yetendu,0,0) * 100
# maintenant on slice en colonnes
for c in range(0,41):
    print(np.median(yetendu[:,c]))
    plt.hist(yetendu[:,c])


plt.show()
