import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

base = pd.read_csv('Aprendizagem_bayesiana/csv/cov_types.csv')

# print(base.columns)

# divisão entre previsores e classe
variables = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
             'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
             'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
             'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area', 'Soil_Type']

x_base = base[variables]
y_base = base['Cover_Type']

# tratamento da classe (y) (label encoder)

label_cover = LabelEncoder()
y_base = label_cover.fit_transform(y_base)

# tratamento previsores
# StandardScaler no inicio pois são atributos numericos
# LabelEncoder no final pois são atributos categóricos

x_encoder = ColumnTransformer(transformers=[('StandarScaler', StandardScaler(), slice(0, 10)), ('OneHotEncoder', OneHotEncoder(sparse_output=False), [10, 11])])
x_base = x_encoder.fit_transform(x_base)


x_treino, x_teste, y_treino, y_teste = train_test_split(x_base, y_base, test_size=0.25, random_state=0)


