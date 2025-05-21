from langchain_community.chat_models import ChatOllama
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import logging

class TutorAdaptativo:
    def __init__(self, indexador, model: str = "llama2"):
        self.logger = logging.getLogger(__name__)
        self.indexador = indexador
        self.model = model
        self._inicializar_llm()
        self._configurar_chain()

    def _inicializar_llm(self):
        try:
            self.llm = ChatOllama(
                model=self.model,
                temperature=0.7,
                base_url="http://localhost:11434"
            )
        except Exception as e:
            self.logger.error(f"Falha ao iniciar ChatOllama: {str(e)}")
            raise

    def _configurar_chain(self):
        template = """Você é um tutor educacional. Responda no formato solicitado.
        
        Histórico da Conversa:
        {chat_history}
        
        Pergunta: {question}
        Formato solicitado: {formato}
        
        Resposta:"""
        
        prompt = PromptTemplate(
            input_variables=["chat_history", "question", "formato"],
            template=template
        )
        
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            memory=self.memory,
            verbose=False
        )

    def responder(self, pergunta: str, formato: str = "texto") -> str:
        try:
            # Verifica se é um comando de formato
            if pergunta.lower().strip() == "formato":
                return "Por favor, especifique o formato desejado (texto, vídeo ou áudio)"
                
            resultado = self.chain.run(
                question=pergunta,
                formato=formato
            )
            return resultado
        except Exception as e:
            self.logger.error(f"Erro ao responder: {str(e)}")
            return "Ocorreu um erro ao processar sua pergunta."