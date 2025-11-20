import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

housing = pd.read_csv('Livro_Maos_a_obra/Housing/csv/housing.csv')

print(housing.head())

# com esse comando podemos ver quais valores tem null
print(housing.info())

print(housing.describe())

# housing.hist(bins=50, figsize=(20,15))
# plt.show()

# criar conjunto de testes
# testes criados com o criterio `aleatorio` 
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(len(train_set), "train + ", len(test_set), "test")


# rever essa parte do codigo
housing['income_cat'] = np.ceil(housing['median_income'] / 1.5).copy()
# O jeito moderno (sem aviso)
housing['income_cat'] = housing['income_cat'].where(
    housing['income_cat'] < 5, 5.0)


""" split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index] """

# esse código abaixo faz a mesma coisa que o de cima
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)


# mostra a quantidade (20% dos valores de income_cat) em cada um dos estratos
print(strat_test_set["income_cat"].value_counts())

# ele mostra a porcentagem dos valores de cada um dos estratos
# 3.0    0.350533
# 2.0    0.318798
# 4.0    0.176357
# 5.0    0.114341
# 1.0    0.039971
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

for set_ in (strat_train_set, strat_test_set):
  set_.drop("income_cat", axis=1, inplace=True)

