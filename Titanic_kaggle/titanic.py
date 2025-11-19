import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train = pd.read_csv('Titanic_kaggle/csv/train.csv')
test = pd.read_csv('Titanic_kaggle/csv/test.csv')


variables = ['Sex', 'Age']
x = train[variables].copy()
y = train['Survived']


# tratando as colunas que estavam com valores faltantes

# print(x.loc[x['Age'].isnull()])
# print(x['Age'].mean())
x['Age'] = x['Age'].fillna(x['Age'].mean())
# print(x.loc[x['Age'].isnull()])

# convertendo o Male e Female por valores binarios
# e padronizando os valores de Age

one_hot_encoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(drop='first'), [
                                    'Sex']), ('StandardScaler', StandardScaler(), ['Age'])], remainder='passthrough')
x = one_hot_encoder.fit_transform(x)

# dividindo os dados
x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, test_size=0.25, random_state=0)
# print(x_treino.shape)
# print(y_treino.shape)
# print(x_teste.shape)
# print(y_teste.shape)


# treinamento
modelo = GaussianNB()
modelo.fit(x_treino, y_treino)

# previsão
previsao = modelo.predict(x_teste)
acuracia = accuracy_score(y_teste, previsao) * 100
print(f'A acuracia do modelo foi de {acuracia: .2f}')

#comparando Com os dados do gender_submission.csv

# prevendo os dados do CSV test e testando com o gender_submission
gender_submission = pd.read_csv('Titanic_kaggle/csv/gender_submission.csv') # lendo o CSV

x_test = test[variables].copy()

x_test['Age'] = x_test['Age'].fillna(train['Age'].mean())

print(x_test)

x_test = one_hot_encoder.transform(x_test)

previsoes_test = modelo.predict(x_test)


print(accuracy_score(gender_submission['Survived'], previsoes_test) * 100)
