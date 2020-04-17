# logistic-regression-for-covid-19
Modélisation naïve de l'épidémie Covid-19 par une fonction logistique

Implémentation en python2 (a priori compatible python3)

## Dépendances
* numpy
* pymc2
* matplotlib.pyplot
  
## Principe
On cherche à approcher la série temporelle des valeurs cumulées des {décès | hospitalisés | cas détectés | … } par une fonction logistique généralisée à 5 paramètres :

     y(t) = amp / ((1 + q * exp(pente * (x - bias))) ** (1 / nu))
     
(voir https://en.wikipedia.org/wiki/Generalised_logistic_function)

L'approche est bayesienne : on considère les 5 paramètres comme des variables stochastiques, au départ de distribution uniforme (priors). Après application d'un Monte-Calo Markov Chaining (MCMC), on obtient les distributions posterior des paramètres «fittant» les observations.

Les paramètres sont ensuite échantillonés depuis ces distributions posterior et sont utilisés pour appliquer la fonction logistique à un domaine étendu, permettant d'obtenir une famille de courbes de prévision d'évolution du phénomène.

![Exemple de résultat](Projection-France_26-03-20.png)

## Usage
`python fit-function-logistic-5params.py <N>`

Si N=0 prise en compte de l'ensemble des observations, sinon prise en compte des N premières

Le code fit-function-logistic-3params.py est une version antérieure, implémentant la fonction logistique non généralisée.

## Références
Probabilistic Programming and Bayesian Methods for Hackers, Cameron Davidson-Pilon https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC2.ipynb

## Bugs
Les bornes actuelles utilisées pour les priors provoquent souvent des RuntimeWarning overflow. En général, cela ne compromet pas la convergence du fitting.

Code à largement restructurer !