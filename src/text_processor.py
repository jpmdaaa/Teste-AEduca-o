from langchain_community.document_loaders import TextLoader
import os
import logging
from langchain.schema import Document

class TextProcessor:
    def __init__(self, indexador):
        self.indexador = indexador
        self.logger = logging.getLogger(__name__)

    def processar(self, pasta="dados/textos"):
   
        documentos = []
        try:
            for arquivo in os.listdir(pasta):
                if arquivo.endswith(".txt"):
                    try:
                        loader = TextLoader(os.path.join(pasta, arquivo), encoding='utf-8')
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata.update({
                                "tipo": "texto",
                                "fonte": arquivo
                            })
                        documentos.extend(docs)
                        self.logger.info(f"Processado texto: {arquivo}")
                    except Exception as e:
                        self.logger.error(f"Erro no texto {arquivo}: {e}")
        except Exception as e:
            self.logger.error(f"Erro ao acessar textos: {e}")
        return documentos