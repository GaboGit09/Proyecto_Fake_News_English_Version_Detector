import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
import pickle

# Descargar stopwords de NLTK
nltk.download('stopwords')

# Cargar el dataset
path = r'C:\Users\gabri\OneDrive\Documentos\00. UNIVERSIDAD FIDELITAS\Portfolio\Completado\Proyecto_Fake_News_English_Version_Detector\data\train.csv'
news_dataset = pd.read_csv(path)

# Reemplazar valores faltantes
news_dataset = news_dataset.fillna('')

# Combinar autor y título
news_dataset['content'] = news_dataset['text'] + ' ' + news_dataset['title']

# Separar datos y etiquetas
X = news_dataset['content'].values
Y = news_dataset['label'].values

# Stemming
port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

# Convertir el texto a datos numéricos
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(news_dataset['content'])

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, stratify=Y, random_state=2)

# Entrenar el modelo
model = LogisticRegression()
model.fit(X_train, Y_train)

# Accuracy sobre los datos entrenados
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy Score of the training data: ', training_data_accuracy)

# Accuracy sobre los datos testeados
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy Score of the testing data: ', testing_data_accuracy)

# Predicción con un nuevo dato
X_new = X_test[1]  # Asegúrate de que el índice esté en el rango correcto
prediction = model.predict(X_new)
print(prediction)

if prediction[0] == 0:  # 0 es real
    print('The news is Real')
else:  # 1 es falso
    print('The news is Fake')

print(Y_test[1])

# Guardar el modelo
filename = 'modelo/model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
print(f'Modelo guardado en {filename}')

# Guardar el vectorizador
filename2 = 'modelo/vectorizer.pkl'
with open(filename2, 'wb') as file:
    pickle.dump(vectorizer, file)
print(f'Vectorizer guardado en {filename2}')
