# logistic-regression-for-covid-19
Modélisation naïve de l'épidémie Covid-19 par une fonction logistique

Implémentation en python2 (a priori compatible python3)

## Dépendances
* numpy
* pymc2
* matplotlib.pyplot
  
## Principe
On cherche à approcher la série temporelle des valeurs cumulées des {décès | cas détectés} par une fonction logistique à 3 paramètres : la pente, le bias (décalage du point d'inflexion par rapport à 0) et l'amplitude (effectif concerné à l'asymptote).

L'approche est bayesienne : on considère les 3 paramètres comme des variables stochastiques, au départ de distribution uniforme (priors). Après application d'un Monte-Calo Markov Chaining (MCMC), on obtient les distributions posterior des paramètres «fittant» les observations.

Les échantillons des paramètres issus des X derniers pas (actuellement 3000) du MCMC sont utilisés pour appliquer la fonction logistique à un domaine étendu, permettant d'obtenir un nuage de prévisions d'évolution du phénomène.

![Exemple de résultat](Projection-France_26-03-20.png)

## Usage
`python rr-fit-function-logistic.py <N>`

Si N=0 prise en compte de l'ensemble des observations, sinon prise en compte des N premières
