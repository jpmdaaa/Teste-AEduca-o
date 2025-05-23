from langchain_ollama import ChatOllama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import logging
from typing import List, Dict

class TutorAdaptativo:
    def __init__(self, indexador, model: str = "llama2"):
        self.logger = logging.getLogger(__name__)
        self.indexador = indexador
        self.model = model
        self._inicializar_llm()
        self._configurar_prompts()

    def _inicializar_llm(self):
        try:
            self.llm = ChatOllama(
                model=self.model,
                temperature=0.7,
                base_url="http://localhost:11434"
            )
            self.logger.info(f"Modelo {self.model} carregado com sucesso")
        except Exception as e:
            self.logger.error(f"Falha ao iniciar ChatOllama: {str(e)}")
            raise

    def _configurar_prompts(self):
        """Configura templates para diferentes formatos de resposta"""
        self.prompt_base = PromptTemplate(
            input_variables=["contexto", "formato", "pergunta", "nivel"],
            template="""
            Você é um tutor educacional adaptativo da +A Educação.
            
            Contexto dos materiais:
            {contexto}
            
            Requisitos:
            - Nível do aluno: {nivel}
            - Formato solicitado: {formato}
            - Pergunta: {pergunta}
            
            Regras:
            1. Seja claro e educacional
            2. Adapte a complexidade ao nível do aluno
            3. Utilize apenas os materiais disponíveis
            4. Formate a resposta conforme solicitado
            """
        )
        
        # Chain principal
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_base,
            verbose=True
        )

    def responder(self, pergunta: str, formato: str = "texto", nivel: str = "iniciante") -> str:
        """Gera resposta adaptativa baseada nos materiais"""
        try:
            # Busca contexto relevante
            docs = self.indexador.buscar_semelhantes(pergunta, k=3)
            contexto = self._formatar_contexto(docs)
            
            if not contexto:
                return self._resposta_off_topic(formato)
            
            # Gera resposta formatada
            resposta = self.chain.run({
                "contexto": contexto,
                "formato": formato,
                "pergunta": pergunta,
                "nivel": nivel
            })
            
            return self._formatar_resposta(resposta, formato, docs)
            
        except Exception as e:
            self.logger.error(f"Erro ao responder: {str(e)}")
            return "Ocorreu um erro ao processar sua pergunta."

    def _formatar_contexto(self, docs: List) -> str:
        """Formata os documentos para contexto"""
        if not docs:
            return ""
            
        return "\n\n".join(
            f"Material: {doc.metadata.get('fonte', 'Desconhecido')}\n"
            f"Tipo: {doc.metadata.get('tipo', 'texto')}\n"
            f"Conteúdo: {doc.page_content[:1000]}..."
            for doc in docs
        )

    def _formatar_resposta(self, resposta: str, formato: str, docs: List) -> str:
        """Aplica formatação final baseada no tipo de mídia"""
        if formato in ["vídeo", "video"]:
            return (
                "🎥 **Recursos em Vídeo**\n\n" +
                "\n".join(
                    f"- {doc.metadata.get('fonte', 'Vídeo')} "
                    f"({doc.metadata.get('duracao', 'N/A')}s)\n"
                    f"  🔗 [Assistir]({doc.metadata.get('url', '#')})"
                    for doc in docs if doc.metadata.get("tipo") == "video"
                ) + "\n\n" + resposta
            )
        elif formato in ["áudio", "audio"]:
            return (
                "🔊 **Conteúdo em Áudio**\n\n" +
                "\n".join(
                    f"- {doc.metadata.get('fonte', 'Áudio')} "
                    f"({doc.metadata.get('duracao', 'N/A')}s)\n"
                    f"  🎧 [Ouvir]({doc.metadata.get('url', '#')})"
                    for doc in docs if doc.metadata.get("tipo") == "audio"
                ) + "\n\n" + resposta
            )
        else:
            return resposta

    def _resposta_off_topic(self, formato: str) -> str:
        """Resposta para tópicos fora dos materiais"""
        formatos = {
            "vídeo": "🎥 Não encontrei vídeos sobre este tema nos materiais disponíveis.",
            "video": "🎥 Não encontrei vídeos sobre este tema nos materiais disponíveis.",
            "áudio": "🔊 Não encontrei áudios sobre este tema nos materiais disponíveis.",
            "audio": "🔊 Não encontrei áudios sobre este tema nos materiais disponíveis.",
            "texto": "📚 Este assunto não está coberto nos materiais atuais."
        }
        return formatos.get(formato.lower(), "Tópico não encontrado nos materiais disponíveis.")