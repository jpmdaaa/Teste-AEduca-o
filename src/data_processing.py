import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import whisper
from typing import List
from langchain.schema import Document
import logging

# Configuração de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDINGS_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
    def _load_whisper_model(self):
        """Carrega o modelo Whisper (somente quando necessário para economizar memória)"""
        if self.whisper_model is None:
            logger.info("Carregando modelo Whisper...")
            try:
                self.whisper_model = whisper.load_model("base")
            except Exception as e:
                logger.error(f"Falha ao carregar modelo Whisper: {e}")
                raise
        return self.whisper_model
    
    def process_pdfs(self, pdf_folder: str) -> List[Document]:
        """Processa todos os arquivos PDF em um diretório"""
        if not os.path.exists(pdf_folder):
            logger.warning(f"Pasta de PDFs não encontrada: {pdf_folder}")
            return []
            
        # Filtra apenas arquivos PDF
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        if not pdf_files:
            logger.warning(f"Nenhum arquivo PDF encontrado em {pdf_folder}")
            return []
            
        all_docs = []
        for pdf_file in pdf_files:
            try:
                file_path = os.path.join(pdf_folder, pdf_file)
                logger.info(f"Processando PDF: {file_path}")
                
                # Extrai texto e metadados do PDF
                loader = PyPDFLoader(file_path)
                pages = loader.load_and_split(self.text_splitter)
                all_docs.extend(pages)
                
                logger.info(f"Processadas {len(pages)} páginas de {pdf_file}")
            except Exception as e:
                logger.error(f"Erro ao processar {pdf_file}: {e}")
                continue
                
        return all_docs
    
    def process_texts(self, text_folder: str) -> List[Document]:
        """Processa todos os arquivos de texto em um diretório"""
        if not os.path.exists(text_folder):
            logger.warning(f"Pasta de textos não encontrada: {text_folder}")
            return []
            
        # Filtra apenas arquivos TXT
        text_files = [f for f in os.listdir(text_folder) if f.lower().endswith('.txt')]
        if not text_files:
            logger.warning(f"Nenhum arquivo de texto encontrado em {text_folder}")
            return []
            
        all_docs = []
        for text_file in text_files:
            try:
                file_path = os.path.join(text_folder, text_file)
                logger.info(f"Processando arquivo de texto: {file_path}")
                
                # Lê e divide o conteúdo do arquivo
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
                docs = self.text_splitter.split_documents(documents)
                all_docs.extend(docs)
                
                logger.info(f"Processados {len(docs)} trechos de {text_file}")
            except Exception as e:
                logger.error(f"Erro ao processar {text_file}: {e}")
                continue
                
        return all_docs
    
    def process_videos(self, video_folder: str) -> List[Document]:
        """Processa todos os vídeos em um diretório (transcrição de áudio)"""
        if not os.path.exists(video_folder):
            logger.warning(f"Pasta de vídeos não encontrada: {video_folder}")
            return []
            
        # Formatos de vídeo suportados
        video_files = [f for f in os.listdir(video_folder) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        if not video_files:
            logger.warning(f"Nenhum arquivo de vídeo encontrado em {video_folder}")
            return []
            
        all_docs = []
        model = self._load_whisper_model()  # Carrega o modelo se necessário
        
        for video_file in video_files:
            try:
                file_path = os.path.join(video_folder, video_file)
                logger.info(f"Processando vídeo: {file_path}")
                
                # Transcreve o áudio do vídeo
                result = model.transcribe(file_path)
                transcript = result['text']
                
                # Cria documento com metadados
                doc = Document(
                    page_content=transcript,
                    metadata={
                        'source': file_path,
                        'type': 'video',
                        'segments': len(result['segments'])
                    }
                )
                all_docs.append(doc)
                
                logger.info(f"Vídeo {video_file} processado com {len(transcript)} caracteres")
            except Exception as e:
                logger.error(f"Erro ao processar {video_file}: {e}")
                continue
                
        return all_docs
    
    def create_vector_index(self, documents: List[Document]):
        """Cria índice vetorial FAISS a partir dos documentos"""
        if not documents:
            logger.warning("Nenhum documento fornecido para indexação")
            return None
            
        try:
            logger.info(f"Criando índice vetorial com {len(documents)} documentos")
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            logger.info("Índice vetorial criado com sucesso")
            return vectorstore
        except Exception as e:
            logger.error(f"Erro ao criar índice vetorial: {e}")
            raise