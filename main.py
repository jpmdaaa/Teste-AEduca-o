import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import logging
import sys
from typing import List, Dict
from src.indexador import Indexador
from src.audio_processor import AudioProcessor
from src.pdf_processor import PDFProcessor
from src.text_processor import TextProcessor
from src.video_processor import VideoProcessor
from src.image_processor import ImageProcessor
from src.tutor_adaptativo import TutorAdaptativo
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama 

class Sistema:
    def __init__(self, config: Dict = None):
        self.tutor = None
        self.logger = logging.getLogger(__name__)
        self.config = config or {
            "ollama_model": "llama2",
            "pasta_dados": "dados",
            "whisper_model": "base",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "modo_quieto": False 
        }
        self._inicializar_componentes()

    def _inicializar_componentes(self):
        """Inicializa todos os componentes com tratamento de erros"""
        try:
            # Configuração do Indexador
            self.indexador = Indexador(config={
                "audio_model": self.config["whisper_model"],
                "chunk_size": self.config["chunk_size"],
                "chunk_overlap": self.config["chunk_overlap"]
            })
            

            self.tutor = TutorAdaptativo( 
                indexador=self.indexador,
                model=self.config["ollama_model"]
            )
            self.logger.info("Tutor inicializado com sucesso")

            # Configuração do LLM local com Ollama
            self.llm = Ollama(
                base_url="http://localhost:11434",
                model=self.config["ollama_model"],
                temperature=0.7,
                timeout=600    
            )
            
            # Processadores
            self.processadores = {
                "pdf": PDFProcessor(self.indexador),
                "texto": TextProcessor(self.indexador),
                "video": VideoProcessor(self.indexador),
                "imagem": ImageProcessor(),
                "audio": AudioProcessor(model_size=self.config["whisper_model"])
            }
            
            self.logger.info("Componentes inicializados com sucesso")
            
        except Exception as e:
            self.logger.critical(f"Falha na inicialização: {str(e)}")
            raise
        
    def _verificar_ambiente(self):
        """Verifica requisitos e estrutura de pastas"""
        try:
            # Verifica dependências
            import torch
            import whisper
            from PIL import Image
            
           # Verifica se a pasta de dados foi especificada
            if "pasta_dados" not in self.config:
                raise ValueError("Configuração 'pasta_dados' não encontrada")
            
            # Cria a estrutura de pastas necessária
            pastas_necessarias = ['pdfs', 'textos', 'videos', 'audios', 'imagens']
            for pasta in pastas_necessarias:
                caminho = os.path.join(self.config["pasta_dados"], pasta)
                os.makedirs(caminho, exist_ok=True)
                self.logger.info(f"Pasta verificada/criada: {caminho}")
            
            return True
        
        except ImportError as e:
            self.logger.error(f"Dependência faltando: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Erro na verificação do ambiente: {str(e)}")
            return False

    def processar_dados(self):
        """Processa todos os tipos de dados"""
        documentos = []
        
        for tipo, processor in self.processadores.items():
            try:
                docs = processor.processar()
                if docs:
                    documentos.extend(docs)
                    self.logger.info(f"Processado {len(docs)} documentos {tipo}")
            except Exception as e:
                self.logger.error(f"Erro no {tipo}_processor: {str(e)}")
                continue
                
        return documentos

    def executar(self):
        """Fluxo principal atualizado"""
        if not self._verificar_ambiente():
            sys.exit(1)
            
        self.logger.info("=== INICIANDO SISTEMA ===")
        
        # Processamento dos dados
        documentos = self.processar_dados()
        if not documentos:
            self.logger.error("Nenhum documento válido encontrado")
            return
            
        # Criação do índice
        try:
            if not self.indexador.criar_indice(documentos):
                raise RuntimeError("Falha na criação do índice")
            self.logger.info(f"Índice criado com {len(documentos)} documentos")
        except Exception as e:
            self.logger.critical(f"Erro no índice: {str(e)}")
            return
            
        # Inicialização do Tutor
        try:
            self.tutor = TutorAdaptativo(
            indexador=self.indexador,
            model=self.config["ollama_model"])
            self._iniciar_interacao()

        except Exception as e:
            self.logger.critical(f"Falha no tutor: {str(e)}")


    def _detectar_formato_preferido(self, historico: List[str]) -> str:
        """Analisa o histórico para inferir o formato preferido (texto/vídeo/áudio)."""
        formatos = {
            "texto": sum(1 for msg in historico if "[VIDEO]" not in msg and "[AUDIO]" not in msg),
            "video": sum(1 for msg in historico if "[VIDEO]" in msg),
            "audio": sum(1 for msg in historico if "[AUDIO]" in msg)
        }
        return max(formatos.items(), key=lambda x: x[1])[0]
    
    def _iniciar_interacao(self):
        
        if self.config.get("modo_quieto"):
            return  # Não inicia o console no modo web
        
        print("\n" + "="*50)
        print("Sistema de Aprendizado Adaptativo (+A Educação)")
        print("="*50)
    
        formato_atual = "texto"
    
        while True:
            try:
                entrada = input("👤 Você: ").strip()
            
                if entrada.lower() in ('sair', 'exit', 'quit'):
                    break
                
                # Comandos de formato
                if entrada.lower().startswith('formato'):
                    partes = entrada.split()
                    if len(partes) > 1:
                        novo_formato = partes[1].lower()
                        if novo_formato in ['texto', 'vídeo', 'video', 'áudio', 'audio']:
                            formato_atual = novo_formato
                            print(f"🔹 Formato alterado para: {formato_atual}")
                            continue
                        else:
                            print("🔸 Formato inválido. Use: texto, vídeo ou áudio")
                            continue
                    else:
                        print("🔸 Especifique um formato: texto, vídeo ou áudio")
                        continue
            
                # Processa perguntas normais
                resposta = self.tutor.responder(entrada, formato=formato_atual)
                self._exibir_resposta(resposta, formato_atual)
            
            except Exception as e:
                self.logger.error(f"Erro na interação: {str(e)}")
                print("🔴 Ocorreu um erro. Por favor, tente novamente.")

    def _exibir_resposta(self, resposta: str, formato: str):
        """Formata a resposta conforme o tipo de conteúdo"""
        if formato == "víde" or formato == "video":  # Compatibilidade com ambas formas
            print("\n🎥 Assistente (Vídeo):")
            print(resposta)
            print("\n🔗 Link recomendado: [integrado com plataforma de vídeos]")
        elif formato == "áudi" or formato == "audio":
            print("\n🔊 Assistente (Áudio):")
            print(resposta)
            print("\n🎧 Versão para escutar: [player integrado]")
        else:
            print("\n📚 Assistente (Texto):")
            print(resposta)
        print("-"*50)

    def configurar_logging():
        """Configura logging detalhado"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] %(message)s',
            handlers=[
                logging.FileHandler('sistema.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

    def is_ready(self):
        """Verifica se todos os componentes estão carregados"""
        return all([
            hasattr(self, 'indexador'),
            hasattr(self, 'llm'), 
            hasattr(self, 'tutor'),
            self.indexador is not None,
            self.llm is not None,
            self.tutor is not None
    ])


    if __name__ == "__main__":
        try:
            configurar_logging()
            sistema = Sistema(config={
                "ollama_model": "llama2",
                "whisper_model": "small",
                "pasta_dados": "dados",
                "chunk_size": 1000,
                "chunk_overlap": 200
            })
            sistema.executar()
        except Exception as e:
            logging.critical(f"FALHA NO SISTEMA: {str(e)}", exc_info=True)
            sys.exit(1)