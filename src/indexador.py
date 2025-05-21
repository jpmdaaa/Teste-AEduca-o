from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from .audio_processor import AudioProcessor
from typing import List, Dict, Union, Optional
import logging
import os

DEFAULT_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "audio_model": "base",
    "device": "cpu"
}

class Indexador:
    def __init__(self, config: Dict = None):
        """
        Args:
            config (Dict): Configurações opcionais:
                - chunk_size: Tamanho dos chunks de texto
                - chunk_overlap: Sobreposição entre chunks
                - model_name: Nome do modelo de embeddings
                - audio_model: Tamanho do modelo Whisper
                - device: Dispositivo para processamento ('cpu' ou 'cuda')
        """
        self.logger = logging.getLogger(__name__)
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._inicializar_componentes()

    def _inicializar_componentes(self):
        """Inicializa todos os componentes do indexador"""
        try:
            self._inicializar_text_splitter()
            self._inicializar_embeddings()
            self._inicializar_processadores()
            self.banco_vetorial = None
            self.logger.info("Componentes do indexador inicializados com sucesso")
        except Exception as e:
            self.logger.critical(f"Falha na inicialização: {str(e)}")
            raise

    def _inicializar_text_splitter(self):
        """Configura o divisor de texto"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
            length_function=len,
            is_separator_regex=False
        )

    def _inicializar_embeddings(self):
        """Carrega o modelo de embeddings"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config["model_name"],
                model_kwargs={'device': self.config["device"]},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                }
            )
            self.logger.info(f"Embeddings carregados (device: {self.config['device']})")
        except Exception as e:
            self.logger.error(f"Falha ao carregar embeddings: {str(e)}")
            raise

    def _inicializar_processadores(self):
        """Inicializa todos os processadores de mídia"""
        self.audio_processor = AudioProcessor(model_size=self.config["audio_model"])
        self.logger.info(f"Processador de áudio inicializado (modelo: {self.config['audio_model']})")

    def processar_e_indexar(self, caminho_pasta: str) -> bool:
        """
        Processa e indexa todos os documentos na pasta especificada
        
        Args:
            caminho_pasta: Caminho para a pasta contendo subpastas por tipo de mídia
            
        Returns:
            bool: True se a indexação foi bem-sucedida
        """
        try:
            documentos = self._coletar_documentos(caminho_pasta)
            if documentos:
                return self.criar_indice(documentos)
            return False
        except Exception as e:
            self.logger.error(f"Erro no processamento: {str(e)}")
            return False

    def _coletar_documentos(self, caminho_pasta: str) -> List[Document]:
        """Coleta documentos de todas as subpastas"""
        documentos = []
        
        # Mapeamento de tipos de mídia para funções de processamento
        processadores = {
            'textos': self._processar_textos,
            'pdfs': self._processar_pdfs,
            'audios': self._processar_audios,
            'videos': lambda x: [],  # Placeholder - implementar se necessário
            'imagens': lambda x: []   # Placeholder - implementar se necessário
        }
        
        for tipo, processador in processadores.items():
            try:
                caminho_completo = os.path.join(caminho_pasta, tipo)
                if os.path.exists(caminho_completo):
                    docs = processador(caminho_completo)
                    documentos.extend(docs)
                    self.logger.info(f"Processados {len(docs)} documentos do tipo {tipo}")
            except Exception as e:
                self.logger.error(f"Erro ao processar {tipo}: {str(e)}")
                continue
                
        return documentos

    def _processar_audios(self, pasta_audios: str) -> List[Document]:
        """Processa arquivos de áudio e retorna documentos"""
        if not hasattr(self, 'audio_processor'):
            self.logger.warning("Processador de áudio não disponível")
            return []
            
        return self.audio_processor.processar(pasta_audios)

    def _processar_textos(self, pasta_textos: str) -> List[Document]:
        """Processa arquivos de texto (.txt, .md)"""
        documentos = []
        if not os.path.exists(pasta_textos):
            return documentos

        for arquivo in os.listdir(pasta_textos):
            if arquivo.endswith(('.txt', '.md')):
                try:
                    caminho = os.path.join(pasta_textos, arquivo)
                    with open(caminho, 'r', encoding='utf-8') as f:
                        conteudo = f.read()
                    
                    # Divide o texto em chunks
                    textos = self.text_splitter.split_text(conteudo)
                    for i, texto in enumerate(textos):
                        doc = Document(
                            page_content=texto,
                            metadata={
                                "tipo": "texto",
                                "fonte": arquivo,
                                "chunk": i+1
                            }
                        )
                        documentos.append(doc)
                        
                except Exception as e:
                    self.logger.error(f"Erro no arquivo {arquivo}: {str(e)}")
        
        return documentos

    def _processar_pdfs(self, pasta_pdfs: str) -> List[Document]:
        """Processa arquivos PDF usando PyPDF"""
        documentos = []
        try:
            from pypdf import PdfReader
        except ImportError:
            self.logger.error("PyPDF não instalado. Execute: pip install pypdf")
            return documentos

        for arquivo in os.listdir(pasta_pdfs):
            if arquivo.endswith('.pdf'):
                try:
                    caminho = os.path.join(pasta_pdfs, arquivo)
                    reader = PdfReader(caminho)
                    texto_total = "\n".join([page.extract_text() or "" for page in reader.pages])
                    
                    # Divide o texto em chunks
                    textos = self.text_splitter.split_text(texto_total)
                    for i, texto in enumerate(textos):
                        doc = Document(
                            page_content=texto,
                            metadata={
                                "tipo": "pdf",
                                "fonte": arquivo,
                                "paginas": len(reader.pages),
                                "chunk": i+1
                            }
                        )
                        documentos.append(doc)
                        
                except Exception as e:
                    self.logger.error(f"Erro no PDF {arquivo}: {str(e)}")
        
        return documentos

    def criar_indice(self, documentos: List[Document]) -> bool:
        """Cria ou atualiza o índice vetorial"""
        if not documentos:
            self.logger.warning("Nenhum documento para indexar")
            return False
            
        try:
            if self.banco_vetorial is None:
                self.banco_vetorial = FAISS.from_documents(
                    documentos, 
                    self.embeddings
                )
                self.logger.info(f"Novo índice criado com {len(documentos)} documentos")
            else:
                self.banco_vetorial.add_documents(documentos)
                self.logger.info(f"Índice atualizado com {len(documentos)} novos documentos")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na indexação: {str(e)}")
            return False

    def buscar_semelhantes(self, consulta: str, k: int = 3, filtro: Dict = None) -> List[Document]:
        """Busca documentos similares com filtros opcionais"""
        if self.banco_vetorial is None:
            self.logger.warning("Índice não inicializado")
            return []
            
        try:
            return self.banco_vetorial.similarity_search(
                query=consulta,
                k=k,
                filter=filtro
            )
        except Exception as e:
            self.logger.error(f"Erro na busca: {str(e)}")
            return []

    def salvar_indice(self, caminho: str) -> bool:
        """Salva o índice em disco"""
        try:
            if self.banco_vetorial:
                self.banco_vetorial.save_local(caminho)
                self.logger.info(f"Índice salvo em {caminho}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Erro ao salvar índice: {str(e)}")
            return False

    def carregar_indice(self, caminho: str) -> bool:
        """Carrega um índice existente"""
        try:
            self.banco_vetorial = FAISS.load_local(
                folder_path=caminho,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.logger.info(f"Índice carregado de {caminho}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao carregar índice: {str(e)}")
            return False