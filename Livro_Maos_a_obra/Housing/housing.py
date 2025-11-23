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


# isso aqui ele cria o estrato de uma categoria em relação a renda
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

#retirar o income_cat
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

# mapa das áreas 
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,s=housing["population"]/100, label="population", figsize=(10,7),c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)


# buscando correlações entre os dados

corr_matrix = housing.corr(numeric_only=True)

print(corr_matrix["median_house_value"].sort_values(ascending=False))

# criando novos atributos 
housing["rooms_per_household"] = housing["total_rooms"]/ housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["populations_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr(numeric_only=True)

print(corr_matrix["median_house_value"].sort_values(ascending=False))



# <h1> preparando os dados para o aprendizado </h1>
#print(strat_train_set.head())
#print(housing.head())

# aqui ele cria o previsor housing sem a coluna `resposta` por assim dizer (x)
housing = strat_train_set.drop("median_house_value", axis=1)
print(housing.head())
# aqui ele cria o rotulo `que guarda a resposta` (y)
housing_labels = strat_train_set["median_house_value"].copy()

#limpando dados, como já visto antes vc pode 
# - livrar dos faltantes (.dropna())
# - livra do atributo inteiro (.drop())
# - define um valor (0, media, intermediaria) (median = housing['totalbedrooms'].median() e depois housing["total_bedrooms"]. fillna(median))

# ah, e existe o imputer (para atributos numericos)
from sklearn.impute import SimpleImputer
# cria a instância
imputer = SimpleImputer(strategy="median")
# cria uma copia dos dados sem o ocean_proximity
housing_num = housing.drop("ocean_proximity", axis=1)

#print(housing_num.head())

#ajusta pára os dados de treinamento
imputer.fit(housing_num)

#aqui ele ta substituindo e guardando dentro de x com os valores já preenchidos
x = imputer.transform(housing_num)

#transformando de volta para DF
# housing_tr = pd.DataFrame(x, columns=housing_num.columns)
# print(housing_tr.head())



