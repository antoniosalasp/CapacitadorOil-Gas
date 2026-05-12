import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from engine import OilGasGeminiBot

load_dotenv()

API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

st.set_page_config(page_title="Capacitador Oil & Gas (Gemini)", page_icon="🛢️")
st.title("Capacitador Oil & Gas (Gemini)")

if not API_KEY:
    st.warning(
        "No se encontró una clave de Gemini en el entorno. Puedes ingresarla abajo o configurar GOOGLE_API_KEY/GEMINI_API_KEY en los secrets."
    )

language = st.radio("Idioma", ["Español", "Inglés"], index=0)
api_key_input = st.text_input(
    "API Key de Gemini (opcional)",
    type="password",
    placeholder="Ingresa tu clave aquí si no está en los secretos",
)
uploaded_files = st.file_uploader("Subir PDF/TXT", type=["pdf", "txt"], accept_multiple_files=True)
question = st.text_area("Tu pregunta técnica", height=150)

if st.button("Enviar Consulta"):
    effective_key = api_key_input.strip() or API_KEY
    if not question.strip():
        st.error("Escribe una pregunta técnica antes de enviar.")
    elif not effective_key:
        st.error(
            "No se puede procesar la consulta: falta la API key de Gemini. "
            "Define GOOGLE_API_KEY/GEMINI_API_KEY en los secrets o ingrésala aquí."
        )
    else:
        bot = OilGasGeminiBot(language=language, api_key=effective_key)
        try:
            if uploaded_files:
                with tempfile.TemporaryDirectory() as tmpdir:
                    paths = []
                    for uploaded_file in uploaded_files:
                        suffix = os.path.splitext(uploaded_file.name)[1]
                        tmp_path = os.path.join(tmpdir, uploaded_file.name)
                        with open(tmp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        paths.append(tmp_path)
                    bot.ingest_files(paths)
            response = bot.ask(question)
            st.success("Respuesta generada")
            st.write(response)
        except Exception as e:
            st.error(f"Error técnico: {e}")
