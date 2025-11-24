import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# 1. Carregar Dados
housing = pd.read_csv('Livro_Maos_a_obra/Housing/csv/housing.csv')

# 2. Criar Categoria de Renda para Split Estratificado
housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
housing['income_cat'] = housing['income_cat'].where(housing['income_cat'] < 5, 5.0)

# 3. Split Estratificado
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

# 4. Limpeza inicial
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# 5. Separar Previsores (X) e Rótulos (y)
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# 6. Definir a Classe Personalizada
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=False):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# 7. Definir Colunas
# Precisamos definir a lista de numéricos, mas sem rodar imputers manuais
housing_num = housing.drop("ocean_proximity", axis=1) # Apenas para pegar os nomes
num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

# 8. PIPELINE COMPLETO (A única parte que realmente processa)
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_add', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), # Boa prática pra garantir
    ('one_hot', OneHotEncoder())
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs)
])

# 9. Execução Final
housing_prepared = full_pipeline.fit_transform(housing)

print("Shape final:", housing_prepared.shape)