try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

from langchain.schema import Document
import os
import logging
from typing import List, Optional, Dict  # Adicionado Dict aqui
import whisper
import subprocess
import tempfile

class VideoProcessor:
    def __init__(self, indexador=None):
        self.indexador = indexador
        self.logger = logging.getLogger(__name__)
        self.model = None  

    def processar(self, pasta: str = "dados/videos") -> List[Document]:
        """Processa vídeos e retorna como objetos Document"""
        documentos = []
        if not os.path.exists(pasta):
            self.logger.warning(f"Pasta de vídeos não encontrada: {pasta}")
            return documentos

        for arquivo in os.listdir(pasta):
            if arquivo.endswith(('.mp4', '.avi', '.mov')):  # Adicionado mais formatos
                try:
                    path = os.path.join(pasta, arquivo)
                    
                    # Verifica se o arquivo existe e é acessível
                    if not os.access(path, os.R_OK):
                        raise PermissionError(f"Sem permissão para ler o arquivo: {path}")
                    
                    with VideoFileClip(path) as clip:
                        # Tenta extrair legendas primeiro
                        legenda = self._extrair_legendas(path)
                        
                        # Se não houver legendas, transcreve o áudio
                        if not legenda:
                            transcricao = self.transcrever_video(path)
                            conteudo = transcricao["texto"] if transcricao else f"Conteúdo do vídeo {arquivo}"
                        else:
                            conteudo = legenda

                        documento = Document(
                            page_content=conteudo,
                            metadata={
                                "tipo": "video",
                                "fonte": arquivo,
                                "duracao": clip.duration,
                                "resolucao": f"{clip.w}x{clip.h}",
                                "tem_legendas": bool(legenda)
                            }
                        )
                        documentos.append(documento)
                        self.logger.info(f"Vídeo processado: {arquivo}")

                except Exception as e:
                    self.logger.error(f"Erro ao processar {arquivo}: {str(e)}")
                    continue

        return documentos
    
    def transcrever_video(self, caminho: str) -> Optional[Dict]:
        """Transcreve o áudio do vídeo para texto"""
        try:
            if self.model is None:
                self.model = whisper.load_model("base")
            
            result = self.model.transcribe(caminho)
            return {
                "texto": result["text"],
                "segmentos": result["segments"],
                "duracao": result["segments"][-1]["end"] if result["segments"] else 0
            }
        except Exception as e:
            self.logger.error(f"Falha ao transcrever vídeo {caminho}: {str(e)}")
            return None

    def _extrair_legendas(self, caminho_video: str) -> Optional[str]:
        """Extrai legendas embutidas usando ffmpeg"""
        try:
            if not os.path.exists(caminho_video):
                raise FileNotFoundError(f"Arquivo não encontrado: {caminho_video}")

            with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as tmp:
                cmd = [
                    "ffmpeg",
                    "-i", caminho_video,
                    "-map", "0:s:0",
                    "-c:s", "srt",
                    tmp.name,
                    "-y"  # Sobrescreve se existir
                ]
                
                result = subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                if os.path.getsize(tmp.name) > 0:
                    with open(tmp.name, 'r', encoding='utf-8') as f:
                        return f.read()
                return None
                
        except subprocess.CalledProcessError as e:
            self.logger.debug(f"Não foi possível extrair legendas: {e.stderr}")
            return None
        except Exception as e:
            self.logger.error(f"Erro ao extrair legendas: {str(e)}")
            return None