# logistic-regression-for-covid-19
Modélisation naïve de l'épidémie Covid-19 par une fonction logistique

Mise en œuvre pour rétro-estimer les valeurs du modèle AVANT le début de la série temporelle.
(cas de la situation en EHPAD où les chiffres n'ont été consolidés qu'après le début de l'épidémie).

## Principe :

Le code est exactement le même que dans le cas standard, mais là on étend le domaine dans les x négatifs (les jours précédents la première donnée).
En sortie, on ne cherche pas à tracer les faisceaux de courbes mais la valeur la plus probable estimée pour les x négatifs.