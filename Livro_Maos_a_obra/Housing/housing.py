from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer

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

# retirar o income_cat
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
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / \
    housing["total_rooms"]
housing["populations_per_household"] = housing["population"] / \
    housing["households"]

corr_matrix = housing.corr(numeric_only=True)

print(corr_matrix["median_house_value"].sort_values(ascending=False))


# <h1> preparando os dados para o aprendizado </h1>
# print(strat_train_set.head())
# print(housing.head())

# aqui ele cria o previsor housing sem a coluna `resposta` por assim dizer (x)
housing = strat_train_set.drop("median_house_value", axis=1)
print(housing.head())
# aqui ele cria o rotulo `que guarda a resposta` (y)
housing_labels = strat_train_set["median_house_value"].copy()

# limpando dados, como já visto antes vc pode
# - livrar dos faltantes (.dropna())
# - livra do atributo inteiro (.drop())
# - define um valor (0, media, intermediaria) (median = housing['totalbedrooms'].median() e depois housing["total_bedrooms"]. fillna(median))

# ah, e existe o imputer (para atributos numericos)
# cria a instância
imputer = SimpleImputer(strategy="median")
# cria uma copia dos dados sem o ocean_proximity
housing_num = housing.drop("ocean_proximity", axis=1)

# print(housing_num.head())

# ajusta pára os dados de treinamento
imputer.fit(housing_num)

# aqui ele ta substituindo e guardando dentro de x com os valores já preenchidos
x = imputer.transform(housing_num)

# transformando de volta para DF
# housing_tr = pd.DataFrame(x, columns=housing_num.columns)
# print(housing_tr.head())

# Manipulando atributos categoricos
housing_cat = housing[['ocean_proximity']]

encoder = OneHotEncoder()

housing_cat_encoder = encoder.fit_transform(housing_cat)
print(housing_cat_encoder)  # retorna sparse matrix
print(encoder.categories_)

# Customizando transformadores
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nada a fazer aqui, ele não precisa aprender nada

    def transform(self, X):
        # Calcula: Cômodos por Família
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        # Calcula: População por Família
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            # Calcula: Quartos por Cômodo
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            # Retorna: Dados Originais + 3 novas colunas
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            # Retorna: Dados Originais + 2 novas colunas
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# print(housing_extra_attribs)

# pipelines de transformação
num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

print(housing.head())

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")), (
    'attribs_add', CombinedAttributesAdder()), ('std_scaler', StandardScaler())])

cat_pipeline = Pipeline([('imputer', SimpleImputer(
    strategy='most_frequent')), ('one_hot', OneHotEncoder())])

full_pipeline = ColumnTransformer(
    [('num', num_pipeline, num_attribs), ('cat', cat_pipeline, cat_attribs)])

housing_prepared = full_pipeline.fit_transform(housing)

print(housing_prepared)


# <h2> Selecione e Treine um modelo <\h2>

lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
""" print(f'predictions: {lin_reg.predict(some_data_prepared)}')
print(f'labels: {list(some_labels)}') """

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)

tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_mse = np.sqrt(tree_mse)
print(tree_mse)  # provavelmente ele se sobreajustou mal aos dados


scores = cross_val_score(tree_reg, housing_prepared,
                         housing_labels, scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)

lin_scores = cross_val_score(
    lin_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)


def display_scores(scores):
    print(f'Scores: {scores}')
    print(f'Mean: {scores.mean()}')
    print(f'Standard deviation {scores.std()}')


forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
forest_scores = cross_val_score(
    forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_rmse_scores)

""" import joblib

joblib.dump(tree_reg, "meu_modelo_arvore.pkl")
modelo_carregado = joblib.load("meu_modelo_arvore.pkl") """


param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

grid_search = GridSearchCV(forest_reg, param_grid,
                           cv=5, scoring='neg_mean_squared_error')

grid_search.fit(housing_prepared, housing_labels)

print(grid_search.best_params_)

cvres = grid_search.cv_results_

for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)

#analise os melhores modelos e seus erros
feature_importances = grid_search.best_estimator_.feature_importances_

extra_attribs = ['rooms_per_hhould', 'pop_per_hhold', 'bedrooms_per_room']
cat_encoder = cat_pipeline.named_steps['one_hot']
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs

print(sorted(zip(feature_importances, attributes), reverse=True))
