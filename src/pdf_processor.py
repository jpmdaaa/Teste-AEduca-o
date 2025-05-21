from langchain_community.document_loaders import PyPDFLoader
import os
import logging
from langchain.schema import Document


class PDFProcessor:
    def __init__(self, indexador):
        self.indexador = indexador
        self.logger = logging.getLogger(__name__)

    def processar(self, pasta="dados/pdfs"):

        documentos = []
        try:
            for arquivo in os.listdir(pasta):
                if arquivo.endswith(".pdf"):
                    try:
                        loader = PyPDFLoader(os.path.join(pasta, arquivo))
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata.update({
                                "tipo": "pdf",
                                "fonte": arquivo
                            })
                        documentos.extend(docs)
                        self.logger.info(f"Processado PDF: {arquivo}")
                    except Exception as e:
                        self.logger.error(f"Erro no PDF {arquivo}: {e}")
        except Exception as e:
            self.logger.error(f"Erro ao acessar PDFs: {e}")
        return documentos