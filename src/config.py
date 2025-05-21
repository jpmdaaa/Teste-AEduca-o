import os
from dotenv import load_dotenv

load_dotenv(override=True)  # Força recarregar as variáveis

class Config:
    # Configurações do Ollama
     OLLAMA_URL = "http://localhost:11434"
     LLM_MODEL = "llama3:dolphin-mistral"  # ou "dolphin-mistral" para PT-BR
     EMBEDDINGS_MODEL = "BAAI/bge-small-pt-v1.5"
     DATA_PATH = os.getenv("DATA_PATH", "dados")  # Valor padrão caso não exista


     def avaliar_resposta(self, resposta_usuario):
        prompt = """Analise esta resposta sobre {tema}:
        {resposta}
    
        Classifique:
            1. Nível (Iniciante/Intermediário/Avançado)
        2. Principais lacunas (lista)
        3. Sugestão de tópicos para reforço
    
        Retorne JSON formatado."""
    
        return self.llm.invoke(prompt).content