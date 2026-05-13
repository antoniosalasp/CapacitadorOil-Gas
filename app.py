import os
import glob
import shutil
import time
from typing import List, Tuple, Any

# Extensiones de Google y LangChain
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import gradio as gr

# --- 0. CONFIGURACIÓN ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ No se encontró GEMINI_API_KEY en el archivo .env")

# --- 1. CONFIGURACIÓN DE MODELOS ---
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=api_key,
    temperature=0.2
)

# --- 2. LÓGICA DE BASE DE DATOS VECTORIAL (RAG) ---

def get_or_create_vector_store(chunks: List[Any] = [], persist_path: str = "./db_storage") -> Chroma:
    if os.path.exists(persist_path) and not chunks:
        print("📦 Cargando base de datos existente...")
        return Chroma(persist_directory=persist_path, embedding_function=embeddings_model)
    
    if chunks:
        print(f"🚀 Iniciando indexación de {len(chunks)} fragmentos...")
        batch_size = 20
        vector_db = Chroma.from_documents(
            documents=chunks[:batch_size],
            embedding=embeddings_model,
            persist_directory=persist_path
        )
        
        for i in range(batch_size, len(chunks), batch_size):
            next_batch = chunks[i:i + batch_size]
            try:
                print(f"⏳ Subiendo lote {i} a {i + len(next_batch)}...")
                vector_db.add_documents(next_batch)
                time.sleep(12)
            except Exception as e:
                if "429" in str(e):
                    print("⚠️ Cuota alcanzada. Esperando 30 segundos...")
                    time.sleep(30)
                    vector_db.add_documents(next_batch)
                else:
                    print(f"❌ Error: {e}")
        return vector_db
    return None

def procesar_nuevos_archivos(archivos: List[Any]) -> str:
    global vector_db
    if not archivos or vector_db is None: 
        return "⚠️ Error: Sube archivos válidos o inicializa la DB."
    
    nuevos_chunks = []
    os.makedirs("documents", exist_ok=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for archivo in archivos:
        nombre_archivo = os.path.basename(archivo.name)
        ruta_destino = os.path.join("documents", nombre_archivo)
        shutil.copy(archivo.name, ruta_destino)
        
        # Lógica para elegir el cargador según la extensión
        if nombre_archivo.lower().endswith(".pdf"):
            loader = PyPDFLoader(ruta_destino)
            nuevos_chunks.extend(loader.load_and_split(text_splitter=splitter))
        elif nombre_archivo.lower().endswith(".docx") or nombre_archivo.lower().endswith(".doc"):
            loader = Docx2txtLoader(ruta_destino)
            nuevos_chunks.extend(loader.load_and_split(text_splitter=splitter))

    if not nuevos_chunks:
        return "⚠️ No se extrajo texto de los archivos. Verifica el formato."

    # Agregar a la DB por lotes
    for i in range(0, len(nuevos_chunks), 20):
        vector_db.add_documents(nuevos_chunks[i:i+20])
        time.sleep(10)
        
    return f"✅ Archivos procesados e indexados correctamente."
    
# --- 3. LÓGICA DEL CHAT Y MEMORIA ---

mi_memoria = InMemoryChatMessageHistory()

# Modifica tu prompt_template actual:
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente técnico experto en procesos oil & gas. "
               "Responde basándote exclusivamente en el contexto técnico proporcionado. "
               "Si no sabes la respuesta, admítelo.\n\n"
               "IMPORTANTE: Debes responder obligatoriamente en el idioma: {idioma}.\n\n" # <-- Nueva instrucción para traduccion
               "Contexto:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

mi_cadena = prompt_template | llm | StrOutputParser()

def responder_chatbot(pregunta: str, historial: List[dict], idioma: str) -> Tuple[str, List[dict]]:
    global vector_db
    if not pregunta.strip(): return "", historial
    
    if vector_db is None:
        error_msg = {"role": "assistant", "content": "⚠️ La base de datos no está inicializada."}
        return "", historial + [error_msg]
    
    try:
        formatted_history = []
        for msg in mi_memoria.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            formatted_history.append({"role": role, "content": msg.content})

        retriever = vector_db.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke(pregunta)
        contexto = "\n\n".join([d.page_content for d in docs])

        # Pasamos la variable 'idioma' a la cadena
        respuesta = mi_cadena.invoke({
            "question": pregunta,
            "chat_history": formatted_history,
            "context": contexto,
            "idioma": idioma  # <-- Variable dinámica
        })
        
        mi_memoria.add_messages([HumanMessage(content=pregunta), AIMessage(content=respuesta)])
        
        historial.append({"role": "user", "content": pregunta})
        historial.append({"role": "assistant", "content": respuesta})
        
        return "", historial
   
    except Exception as e:
        print(f"Error detallado en responder_chatbot: {e}") # NUEVO: Print para debug
        error_msg = {"role": "assistant", "content": f"❌ Error de conexión: {str(e)[:50]}..."}
        return "", historial + [error_msg]

# --- 4. INTERFAZ DE USUARIO (GRADIO) ---
def crear_interfaz():
    with gr.Blocks() as demo:
        gr.Markdown("# 🛰️ Asistente Técnico Industrial")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot_ui = gr.Chatbot(label="Chat de Ingeniería", height=550) 
                txt_input = gr.Textbox(
                    placeholder="Escribe tu duda técnica aquí...", 
                    show_label=False
                )
                
                # --- NUEVO: Selección de Idioma ---
                selector_idioma = gr.Radio(
                    choices=["Español", "English"], 
                    value="Español", 
                    label="Idioma de respuesta"
                )
                
                btn_enviar = gr.Button("Consultar", variant="primary")
                
            with gr.Column(scale=1):
                file_input = gr.File(label="Manuales PDF y Word", file_count="multiple")
                btn_upload = gr.Button("Indexar Documentos")
                status_msg = gr.Markdown("🟢 **Estado:** Sistema Listo")
                btn_limpiar = gr.Button("Borrar Conversación")

        # --- Actualizamos los inputs para incluir selector_idioma ---
        txt_input.submit(
            responder_chatbot, 
            [txt_input, chatbot_ui, selector_idioma], # <-- Agregado aquí
            [txt_input, chatbot_ui]
        )
        
        btn_enviar.click(
            responder_chatbot, 
            [txt_input, chatbot_ui, selector_idioma], # <-- Agregado aquí
            [txt_input, chatbot_ui]
        )
        
        btn_upload.click(procesar_nuevos_archivos, inputs=[file_input], outputs=[status_msg])
        
        def limpiar_todo():
            mi_memoria.clear()
            return [], "🟢 Estado: Memoria limpia"
            
        btn_limpiar.click(limpiar_todo, outputs=[chatbot_ui, status_msg])

    demo.launch(theme=gr.themes.Soft(), share=True)

if __name__ == "__main__":
    print("🛠️ Iniciando sistema...")
    os.makedirs("documents", exist_ok=True)
    
    # NUEVO: Búsqueda combinada de PDF y Word
    archivos_pdf = glob.glob("documents/*.pdf")
    archivos_word = glob.glob("documents/*.docx")
    existentes = archivos_pdf + archivos_word
    chunks_iniciales = []
    
    print(f"📂 Archivos detectados en /documents: {len(existentes)} ({len(archivos_pdf)} PDF, {len(archivos_word)} Word)")
    
    if existentes and not os.path.exists("./db_storage"):
        print("📄 Procesando documentos para indexación inicial...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        for archivo in existentes:
            try:
                if archivo.lower().endswith(".pdf"):
                    loader = PyPDFLoader(archivo)
                else:
                    loader = Docx2txtLoader(archivo)
                chunks_iniciales.extend(loader.load_and_split(text_splitter=splitter))
                print(f"✅ Leído correctamente: {os.path.basename(archivo)}")
            except Exception as e:
                print(f"❌ Error leyendo {archivo}: {e}")
    
    print("🤖 Inicializando Vector DB...")
    vector_db = get_or_create_vector_store(chunks_iniciales)
    
    if vector_db:
        print("🚀 Sistema RAG inicializado. Levantando interfaz...")
    else:
        print("⚠️ Vector DB iniciada en blanco. Esperando documentos desde la UI...")
        
    crear_interfaz()