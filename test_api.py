from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 1. Configure com UM destes modelos:
model_name = "dolphin-mistral"  # Recomendado para PT-BR
# model_name = "phi3"           # Modelo mais leve
# model_name = "llama3"         # Modelo em inglês

# 2. Crie a cadeia de processamento
llm = ChatOllama(
    base_url="http://localhost:11434",
    model=model_name,
    temperature=0.7,
    timeout=300  # Aumente se necessário
)

# 3. Sistema de prompts
prompt = ChatPromptTemplate.from_template(
    "Você é um especialista em mercado financeiro. "
    "Explique de forma clara o que são {tema} para um iniciante."
)

# 4. Execute
chain = prompt | llm
try:
    response = chain.invoke({"tema": "ações ordinárias"})
    print("Resposta:", response.content)
except Exception as e:
    print("Erro:", e)