import os
import gradio as gr
import pickle

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument
from langchain_unstructured import UnstructuredLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from moviepy import VideoFileClip
import whisper
import yt_dlp
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.prompts import RichPromptTemplate

VECTOR_FOLDER = 'vector_stores'
os.makedirs(VECTOR_FOLDER, exist_ok=True)
HF_TOKEN = os.getenv("HF_TOKEN")
# === Core Classes ===

class VideoProcessor:
    def __init__(self):
      pass

    def extract_audio(self, video_path, audio_path="extracted_audio.mp3"):
        with VideoFileClip(video_path) as video_clip:
            audio_clip = video_clip.audio
            if audio_clip:
                audio_clip.write_audiofile(audio_path)
                return audio_path
        return None

    def download_youtube_audio_yt_dlp(self, url):
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'extracted_audio',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
        return os.path.abspath("extracted_audio.mp3")

class AudioProcessor:
    def __init__(self, model_size="tiny.en"):
        self.model = whisper.load_model(model_size)

    def transcribe_audio(self, audio_path):
        result = self.model.transcribe(audio_path)
        return result['text']

class DocumentLoader:
    def __init__(self, file_path, filename):
        self.file_path = file_path
        self.filename = filename

    def fallback_unstructured_extraction(self):
        loader = UnstructuredLoader(self.file_path)
        return loader.load()

    def read_txt_text(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return [LangchainDocument(page_content=text, metadata={'page_no': 1, 'filename': self.filename})]

    def load_document(self):
        if self.file_path.endswith('.txt'):
            return self.read_txt_text()
        return self.fallback_unstructured_extraction()

    def chunk_text(self, documents, chunk_size=10000, chunk_overlap=1000):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = []
        for doc in documents:
            split_texts = splitter.split_text(doc.page_content)
            for chunk in split_texts:
                chunks.append(LangchainDocument(page_content=chunk, metadata=doc.metadata))
        return chunks

class VectorEmbeddingRetrieval:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def create_vector_store(self, chunks):
        return FAISS.from_documents(chunks, self.embeddings)


class QuestionAnswering:
    def __init__(self):
        self.llm = HuggingFaceInferenceAPI(
            model_name="mistralai/Mistral-7B-Instruct-v0.3",token=HF_TOKEN
        )
    def answer_question(self, contexts, question, chat_history=None):
        try:
          if not HF_TOKEN:
              raise ValueError("Hugging Face API token not found. Please set the HF_API_TOKEN environment variable.")

          prompt = RichPromptTemplate(
              """
          {% chat role="system" %}
          You are a helpful AI assistant.You are a Document Question Answer Assistant.
            Use the following pieces of retrieved context to answer the question.
            Answer the question based on the context provided.
            Reply the answer like conversation manner.
            Keep the answer concise and relevant.
            Only Return The Answer.

            Chat History:
            {{chat_history}}

            Context:
            {{context}}
          {% endchat %}

          {% chat role="user" %}
          "Question: {{question}}
            Helpful Answer:
          {% endchat %}
          """
          )
          pipeline = QueryPipeline(chain=[prompt, self.llm], verbose=True)
          chat_history = []
          full_context = ""
          for context in contexts:
            full_context += "\n" + context.page_content
          answer = pipeline.run(context=full_context, chat_history = chat_history,question=question)
          return answer.raw.choices[0].message.content
        except Exception as e:
          print(f"Error answering question: {e}")
          return "error",str(e)

# === Helper Functions ===

def get_vector_path(filename):
    return os.path.join(VECTOR_FOLDER, f"{filename}.pkl")

def write_txt_file(filename, text):
    file_path = f"{filename}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    return file_path

def video_processing(file_or_url, filename):
    try:
        if file_or_url.startswith("http"):
            audio_path = video_processor.download_youtube_audio_yt_dlp(file_or_url)
        else:
            audio_path = video_processor.extract_audio(file_or_url)
        if audio_path and os.path.exists(audio_path):
            text = audio_processor.transcribe_audio(audio_path)
            return write_txt_file(filename, text)
    except Exception as e:
        return str(e)
    return None

def vectorize_document(txt_path, filename):
    loader = DocumentLoader(txt_path, filename)
    docs = loader.load_document()
    chunks = loader.chunk_text(docs)
    vector_store = vector_retrieval.create_vector_store(chunks)
    with open(get_vector_path(filename), 'wb') as f:
        pickle.dump(vector_store, f)
    return f"{filename} indexed successfully"

def get_summary(filename):
    path = get_vector_path(filename)
    if not os.path.exists(path):
        return "Vector store not found."
    with open(path, 'rb') as f:
        vector_store = pickle.load(f)
    context = vector_store.similarity_search("Give a short summary of the content", k=3)
    return qa.answer_question(context, "Summarize the context")

def retrieve_answer(question, filename):
    path = get_vector_path(filename)
    if not os.path.exists(path):
        return "Vector store not found."
    with open(path, 'rb') as f:
        vector_store = pickle.load(f)
    context = vector_store.similarity_search(question, k=3)
    return qa.answer_question(context, question)

# === Instantiate Processors ===

video_processor = VideoProcessor()
audio_processor = AudioProcessor()
vector_retrieval = VectorEmbeddingRetrieval()
qa = QuestionAnswering()

# === Gradio Interface ===

with gr.Blocks() as demo:
    gr.Markdown("## 🎥 Video QA and Summary Tool")

    with gr.Tabs():
        with gr.Tab("Upload & Process"):
            video_input = gr.File(label="Upload MP4", type="filepath")
            url_input = gr.Textbox(label="Or Paste YouTube URL")
            filename_input = gr.Textbox(label="Filename to Save As", value="sample")
            process_btn = gr.Button("Transcribe & Embed")
            process_output = gr.Textbox(label="Status")

            def process(file_input, url_input, filename_input):
                if url_input:
                    input_path = url_input
                elif file_input:
                    input_path = file_input
                else:
                    return "Please upload a file or paste a URL"

                txt_path = video_processing(input_path, filename_input)
                if not txt_path or not os.path.exists(txt_path):
                    return "Transcription failed."
                return vectorize_document(txt_path, filename_input)

            process_btn.click(process, [video_input, url_input, filename_input], process_output)

        with gr.Tab("Ask Questions"):
            filename_q = gr.Textbox(label="Filename (same as above)", value="sample")
            chatbot = gr.Chatbot(label="Chat with Video", height=400)
            msg = gr.Textbox(label="Ask a question")
            ask_btn = gr.Button("Send")
            clear_btn = gr.Button("Clear Chat")

            chat_history_state = gr.State([])

            def chat_with_video(question, filename, chat_history):
                if not question:
                    return chat_history, chat_history

                vector_path = get_vector_path(filename)
                if not os.path.exists(vector_path):
                    bot_reply = "Vector store not found. Please process the video first."
                    chat_history.append((question, bot_reply))
                    return chat_history, chat_history

                with open(vector_path, 'rb') as f:
                    vector_store = pickle.load(f)

                context = vector_store.similarity_search(question, k=3)
                answer = qa.answer_question(context, question, chat_history)
                chat_history.append((question, answer))
                return chat_history, chat_history

            ask_btn.click(chat_with_video, [msg, filename_q, chat_history_state], [chatbot, chat_history_state])
            clear_btn.click(lambda: ([], []), None, [chatbot, chat_history_state])
            msg.submit(chat_with_video, [msg, filename_q, chat_history_state], [chatbot, chat_history_state])

        with gr.Tab("Get Summary"):
            summary_file = gr.Textbox(label="Filename (same as above)", value="sample")
            summary_output = gr.Textbox(label="Summary", lines=10)
            summary_btn = gr.Button("Generate Summary")
            summary_btn.click(get_summary, [summary_file], summary_output)

demo.launch()
