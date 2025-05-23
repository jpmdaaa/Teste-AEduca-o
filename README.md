# Sistema de Aprendizado Adaptativo (+A Educação)

Solução para o desafio de Engenheiro de IA da +A Educação, implementando um sistema de tutoria inteligente com adaptação dinâmica de conteúdo.

## 📋 Pré-requisitos

- Python 3.10+
- Ollama instalado localmente (com modelo Llama2 baixado)
- FFmpeg (para processamento de vídeos)

## 🚀 Instalação

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/challenge-artificial-intelligence.git
cd challenge-artificial-intelligence
```

2. Crie e ative o ambiente virtual:

**bash**

Copy

Download

```
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```

3. Instale as dependências:

**bash**

Copy

Download

```
pip install -r requirements.txt
```

4. Baixe os modelos necessários:

**bash**

Copy

Download

```
ollama pull llama2
```

## Configuração

1. Crie a estrutura de pastas:

**bash**

Copy

Download

```
mkdir -p dados/{pdfs,textos,videos,audios,imagens}
```

2. Adicione seus arquivos de mídia nas pastas correspondentes

## 🏃 Execução

Inicie o sistema:

**bash**

Copy

Download

```
venv/scripts/activate
python interface.py

```

## 🎯 Comandos do Sistema

* `formato texto` - Respostas textuais
* `formato vídeo` - Sugere recursos visuais
* `formato áudio` - Sugere recursos de áudio
* `sair` - Encerra o sistema

## 🛠️ Tecnologias Principais

* LangChain (pipelines ETL)
* FAISS (indexação vetorial)
* Ollama (modelos locais)
* Whisper (transcrição de áudio/vídeo)

## 📂 Estrutura do Projeto

Copy

Download

```
.
├── dados/               # Arquivos de mídia
├── src/
│   ├── indexador.py     # Núcleo de indexação
│   ├── tutor_adaptativo.py # Lógica do tutor
│   └── processadores/   # Módulos para cada tipo de mídia
├── main.py              # Ponto de entrada
└── requirements.txt     # Dependências
```
