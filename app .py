from flask import Flask, request, jsonify, render_template
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import numpy as np

app = Flask(__name__)


# Función para preprocesar el texto
def preprocess_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Eliminar caracteres especiales
    return text

# Ruta principal de la app para mostrar un formulario simple
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para realizar la predicción
@app.route('/predict', methods=['POST'])
def predict():
    
    # Cargar el modelo y los objetos necesarios una vez al iniciar la aplicación
    model = load_model('modelo_noticias.h5')
    tokenizer = joblib.load('tokenizer.pkl')
    encoder = joblib.load('encoder.pkl')
    maxlen = 150

    # Obtener el texto de la noticia desde el formulario
    texto_noticia = request.form['texto_noticia']
    # Preprocesar el texto de entrada
    texto_procesado = preprocess_text(texto_noticia)
    secuencia = tokenizer.texts_to_sequences([texto_procesado])
    secuencia_padded = pad_sequences(secuencia, maxlen=maxlen)

    # Hacer la predicción
    prediccion = model.predict(secuencia_padded)
    clase_predicha = np.argmax(prediccion, axis=-1)[0]

    # Decodificar la clase numérica de vuelta a la categoría original
    categoria_predicha = encoder.inverse_transform([clase_predicha])[0]

    # Devolver la categoría predicha a la página
    return render_template('index.html', categoria=categoria_predicha, texto=texto_noticia)

if __name__ == '__main__':
    app.run(debug=True)
