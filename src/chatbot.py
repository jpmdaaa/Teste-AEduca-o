from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import logging
from typing import Optional

class Chatbot:
    def __init__(self, banco_vetorial):
        self.logger = logging.getLogger(__name__)
        self.banco_dados = banco_vetorial
        try:
            self.llm = ChatOllama(
                base_url="http://localhost:11434",
                model="dolphin-mistral",
                temperature=0.7
            )
            self._configurar_prompts()
            self.logger.info("Chatbot inicializado com sucesso")
        except Exception as e:
            self.logger.error(f"Falha ao iniciar chatbot: {str(e)}")
            raise

    def _configurar_prompts(self):
        """Configura todos os templates de prompt"""
        self.prompt_base = ChatPromptTemplate.from_template("""
        Você é um tutor de programação adaptativo. 
        Contexto: {contexto}
        
        Responda considerando:
        - Nível: {nivel}
        - Formato preferido: {formato}
        
        Sua resposta deve incluir:
        1. Explicação clara
        2. Exemplo prático
        3. Recurso recomendado
        
        Formato de saída:
        - Explicação: [texto]
        - Exemplo: [código/conceito]
        - Recurso: [tipo: nome]
        """)

    def responder(self, pergunta: str, nivel: str = "intermediário", formato: str = "texto") -> Optional[str]:
        """Gera resposta adaptativa"""
        try:
            if not self.banco_dados:
                return "Sistema não está pronto para responder"
                
            contexto = self._buscar_contexto(pergunta)
            chain = self.prompt_base | self.llm
            resposta = chain.invoke({
                "contexto": contexto,
                "nivel": nivel,
                "formato": formato,
                "pergunta": pergunta
            })
            return resposta.content
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar resposta: {str(e)}")
            return "Desculpe, ocorreu um erro ao processar sua pergunta"

    def _buscar_contexto(self, pergunta: str, k: int = 3) -> str:
        """Busca documentos relevantes"""
        try:
            docs = self.banco_dados.similarity_search(pergunta, k=k)
            return "\n".join([d.page_content for d in docs])
        except Exception as e:
            self.logger.error(f"Erro na busca de contexto: {str(e)}")
            return "Sem contexto disponível"