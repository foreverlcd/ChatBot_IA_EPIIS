import os
import streamlit as st
from gtts import gTTS
import nltk
import string
import random
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
from streamlit_chat import message

# Descargar recursos NLTK si no están disponibles
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Definir los nombres de los archivos a procesar
base_dir = os.path.dirname(os.path.abspath(__file__))
documentos = [os.path.join(base_dir, '..', 'sources', 'Corpus Refinado.txt')]

# Inicializar la variable para almacenar el contenido concatenado de los archivos
texto_completo = ''

# Leer y concatenar el contenido de los archivos
for archivo in documentos:
    with open(archivo, 'r', encoding='utf-8', errors='ignore') as f:
        texto_completo += f.read() + '\n'

# Tokenización del texto
# Lista de oraciones
oraciones = nltk.sent_tokenize(texto_completo)
# Lista de palabras
palabras = nltk.word_tokenize(texto_completo.lower())

# Crear un objeto lematizador
lematizador = nltk.stem.WordNetLemmatizer()

# Crear un diccionario para remover signos de puntuación
signos_puntuacion = dict((ord(signo), None) for signo in string.punctuation if signo != 'ñ')

# Función para lematizar una lista de tokens
def lematizar_tokens(tokens):
    return [lematizador.lemmatize(token) for token in tokens]

# Función para normalizar y lematizar el texto
def normalizar_texto(texto):
    return lematizar_tokens(nltk.word_tokenize(texto.lower().translate(signos_puntuacion)))

# Función para preprocesar el texto del usuario y calcular la matriz TF-IDF
def preprocesar_texto_usuario(respuesta_usuario):
    oraciones.append(respuesta_usuario)
    vectorizador_tfidf = TfidfVectorizer(tokenizer=normalizar_texto, stop_words=stopwords.words('spanish'), token_pattern=None)
    matriz_tfidf = vectorizador_tfidf.fit_transform([oracion.lower() for oracion in oraciones])
    return matriz_tfidf

# Función para generar una respuesta basada en la similitud del texto insertado y el corpus
def generar_respuesta(respuesta_usuario):
    respuesta_bot = ''
    matriz_tfidf = preprocesar_texto_usuario(respuesta_usuario)
    similaridades = cosine_similarity(matriz_tfidf[-1], matriz_tfidf)
    indice_mas_similar = similaridades.argsort()[0][-2]
    valores_achatados = similaridades.flatten()
    valores_achatados.sort()
    puntuacion_similaridad = valores_achatados[-2]

    if puntuacion_similaridad == 0:
        respuesta_bot = "Lo siento, no te entendí 😓. Por favor, contacta al personal asistencial."
    else:
        respuesta_bot = oraciones[indice_mas_similar]

    # Eliminar la última oración añadida para evitar duplicados
    oraciones.pop()

    return respuesta_bot

# Función para responder a saludos
def responder_saludo(oracion):
    for palabra in oracion.split():
        if palabra.lower() in SALUDOS_ENTRADAS:
            return random.choice(SALUDOS_SALIDAS)

# Función para convertir texto a habla y guardar el archivo de audio
def guardar_audio(texto):
    lenguaje = "es"
    objeto_gtts = gTTS(text=texto, lang=lenguaje)
    objeto_gtts.save("respuesta.mp3")

# Función para escuchar por micrófono
def escuchar_por_microfono():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Escuchando...")
        audio = recognizer.listen(source)
        try:
            texto = recognizer.recognize_google(audio, language='es-ES')
            st.write(f"🗣️ Has dicho: {texto}")
            return texto
        except sr.UnknownValueError:
            st.write("❌ No se pudo entender el audio")
        except sr.RequestError as e:
            st.write(f"❌ Error al solicitar resultados del servicio de reconocimiento de voz; {e}")

# Definir saludos de entrada y sus posibles respuestas
SALUDOS_ENTRADAS = ("hola", "buenas", "saludos", "qué tal", "hey", "buenos días")
SALUDOS_SALIDAS = ["Hola", "¿Qué tal?", "Hola, ¿Cómo te puedo ayudar?", "Hola, encantado de hablar contigo"]

# Interfaz de usuario con Streamlit
st.title("Asistente Académico 📚")
st.image("C:\Proyectos\IA_EPIIS\images\DALL·E 2024-12-05 00.08.52 - A beautiful and elegant academic assistant banner with a sophisticated design. The banner should have a clean gray background with smooth gradients, m.webp", use_container_width=True)
st.write("Hola 😎, soy tu asistente académico y estoy para responder todas tus preguntas.")

# Inicializar el historial de mensajes
if 'historial' not in st.session_state:
    st.session_state.historial = []

# Mostrar el historial de mensajes
for i, (usuario, bot) in enumerate(st.session_state.historial):
    message(usuario, is_user=True, key=f"usuario_{i}")
    message(bot, key=f"bot_{i}")

# Entrada de texto del usuario
entrada_usuario = st.text_input("✍️ Escribe tu pregunta aquí:")

if st.button("Enviar 🚀"):
    entrada_usuario = entrada_usuario.lower()
    if entrada_usuario not in ['salir', 'adios', 'chau']:
        if entrada_usuario in ['gracias', 'muchas gracias']:
            respuesta = 'No hay de qué.'
        elif responder_saludo(entrada_usuario):
            respuesta = responder_saludo(entrada_usuario)
        else:
            respuesta = generar_respuesta(entrada_usuario)
    else:
        respuesta = 'Nos vemos pronto, ¡Cuídate! 👋'
    
    # Guardar el mensaje en el historial
    st.session_state.historial.append((entrada_usuario, respuesta))
    
    # Mostrar la respuesta
    message(entrada_usuario, is_user=True)
    message(respuesta)
    
    guardar_audio(respuesta)
    audio_file = open("respuesta.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")

if st.button("Escuchar por micrófono 🎙️"):
    texto_escuchado = escuchar_por_microfono()
    if texto_escuchado:
        respuesta = generar_respuesta(texto_escuchado)
        
        # Guardar el mensaje en el historial
        st.session_state.historial.append((texto_escuchado, respuesta))
        
        # Mostrar la respuesta
        message(texto_escuchado, is_user=True)
        message(respuesta)
        
        guardar_audio(respuesta)
        audio_file = open("respuesta.mp3", "rb")
        st.audio(audio_file.read(), format="audio/mp3")
