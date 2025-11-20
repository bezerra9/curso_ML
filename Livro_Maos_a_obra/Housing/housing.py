import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib
from sklearn.model_selection import train_test_split

housing = pd.read_csv('Livro_Maos_a_obra/Housing/csv/housing.csv')

print(housing.head())

# com esse comando podemos ver quais valores tem null
print(housing.info())

print(housing.describe())

#housing.hist(bins=50, figsize=(20,15))
#plt.show()

# criar conjunto de testes

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(len(train_set), "train + ", len(test_set), "test")

housing['income_cat'] = np.ceil(housing['median_income'] / 1.5).copy()
# O jeito moderno (sem aviso)
housing['income_cat'] = housing['income_cat'].where(housing['income_cat'] < 5, 5.0)

housing.hist('income_cat')
plt.show()
