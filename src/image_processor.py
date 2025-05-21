from PIL import Image
from transformers import pipeline
import magic
import os
import logging
from typing import List
from langchain.schema import Document

class ImageProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            # Modelo leve para testes (substitua por BLIP se tiver recursos)
            self.image_analyzer = pipeline("image-to-text", 
                                         model="Salesforce/blip-image-captioning-base")
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo de imagens: {e}")
            self.image_analyzer = None

    def processar(self, pasta="dados/imagens") -> List[Document]:
        """Processa imagens e gera descrições como objetos Document"""
        documentos = []
        try:
            arquivos = [f for f in os.listdir(pasta) 
                      if magic.from_file(os.path.join(pasta, f), mime=True).startswith('image/')]
            
            if not arquivos:
                self.logger.warning(f"Nenhuma imagem encontrada em {pasta}")
                return documentos

            for arquivo in arquivos:
                try:
                    path = os.path.join(pasta, arquivo)
                    img = Image.open(path)
                    
                    # Gera descrição ou usa fallback
                    descricao = self._gerar_descricao(path) if self.image_analyzer else f"Imagem: {arquivo}"
                    
                    # Cria objeto Document do LangChain
                    documento = Document(
                        page_content=descricao,
                        metadata={
                            "tipo": "imagem",
                            "fonte": arquivo,
                            "dimensoes": f"{img.width}x{img.height}",
                            "formato": img.format,
                            "tags": self._extrair_tags(descricao)
                        }
                    )
                    documentos.append(documento)
                    self.logger.info(f"Imagem processada: {arquivo}")

                except Exception as e:
                    self.logger.error(f"Erro ao processar {arquivo}: {e}")

        except Exception as e:
            self.logger.error(f"Erro ao acessar imagens: {e}")
        
        return documentos

    def _gerar_descricao(self, image_path: str) -> str:
        """Gera descrição usando modelo de IA"""
        try:
            result = self.image_analyzer(image_path)
            return result[0]['generated_text']
        except Exception as e:
            self.logger.error(f"Erro na análise da imagem: {e}")
            return "Descrição não disponível"

    def _extrair_tags(self, descricao: str) -> List[str]:
        """Extrai tags relevantes da descrição"""
        tags = []
        descricao = descricao.lower()
        
        if "diagram" in descricao:
            tags.append("diagrama")
        if "code" in descricao or "program" in descricao:
            tags.append("codigo")
        if "graph" in descricao or "chart" in descricao:
            tags.append("grafico")
        if "screenshot" in descricao:
            tags.append("captura_tela")
            
        return tags