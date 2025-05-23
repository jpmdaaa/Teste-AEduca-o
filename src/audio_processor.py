import whisper
from typing import Dict, List, Optional
import logging
from pydub import AudioSegment  # Corrigido o import
import os
import json
from langchain_core.documents import Document

class AudioProcessor:

    def __init__(self, model_size: str = "base"):
        self.logger = logging.getLogger(__name__)  # Adicione esta linha
        self.model = whisper.load_model(model_size)
        self.logger.info(f"Modelo Whisper {model_size} carregado")

    
    def processar(self, caminho_pasta: str = None) -> List[Document]:
        documentos = []
        if not caminho_pasta:
            caminho_pasta = os.path.join("dados", "audios")
        
        if not os.path.exists(caminho_pasta):
            self.logger.warning(f"Pasta não encontrada: {caminho_pasta}")
            os.makedirs(caminho_pasta, exist_ok=True)
            return []
            

        for arquivo in os.listdir(caminho_pasta):
            if arquivo.endswith(('.mp3', '.wav')):
                try:
                    caminho = os.path.join(caminho_pasta, arquivo)
                    resultado = self.transcrever_audio(caminho)
                    if resultado:
                        doc = Document(
                            page_content=resultado["texto"],
                            metadata={"tipo": "audio", "fonte": arquivo}
                        )
                        documentos.append(doc)
                except Exception as e:
                    self.logger.error(f"Erro no arquivo {arquivo}: {str(e)}")
        return documentos

def transcrever_audio(self, caminho_audio: str) -> Optional[Dict]:
        try:
            audio = AudioSegment.from_file(caminho_audio)
            resultado = self.model.transcribe(caminho_audio)
            return {
                "texto": resultado["text"],
                "duracao": len(audio)/1000
            }
        except Exception as e:
            self.logger.error(f"Falha na transcrição: {str(e)}")
            return None