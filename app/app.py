import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Cargar el modelo y el vectorizador
model_filename = 'modelo/model.pkl'
vectorizer_filename = 'modelo/vectorizer.pkl'

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

with open(vectorizer_filename, 'rb') as file:
    vectorizer = pickle.load(file)

# Inicializar el stemmer de Porter
port_stem = PorterStemmer()

# Definir la función de stemming (para español: "SnowballStemmer")
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Crear la interfaz de usuario de Streamlit
st.title('Fake News Detector')
st.write('Enter the text of the news to determine if it is fake or real.')

user_input = st.text_area("Text of the news")


if st.button('Predict'):
    if user_input:
        # Aplicar el mismo procesamiento al texto del usuario
        processed_input = stemming(user_input)
        vectorized_input = vectorizer.transform([processed_input])
        
        # Realizar la predicción
        prediction = model.predict(vectorized_input)
        
        if prediction[0] == 0:  # 0 es real
            st.write('The news is REAL(Remember that for a better prediction a larger dataset is necessary)')
        else:  # 1 es falso
            st.write('The news is FALSE (Remember that for a better prediction a larger dataset is necessary)')
    else:
        st.write('Please enter the text of the news.')



