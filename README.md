# tabAI

![](https://raw.githubusercontent.com/Jeiel0rbit/tabIA/refs/heads/main/VN20250529_131431.gif)

Este projeto implementa um chatbot simples utilizando Flask, `sklearn` para similaridade de texto e a `API Gemini` para gerar respostas com base em um conjunto de documentos de FAQ do TabNews. Usei https://www.firecrawl.dev/ desde quando foi apresentado por [vedovelli](https://www.tabnews.com.br/vedovelli/construi-um-chatbot-para-responder-a-perguntas-sobre-o-conteudo-tabnews).

## Pré-requisitos

Antes de começar, certifique-se de ter instalado:

* **Python 3.x** (versão 3.8 ou superior é recomendada)
* **pip** (gerenciador de pacotes do Python)

## Configuração do Ambiente

Siga os passos abaixo para configurar e executar o projeto:

### 1. Clonar o Repositório

Repositório Git, clone-o primeiro:

```bash
git clone https://github.com/Jeiel0rbit/tabIA
cd  tabIA
```

### 2. Criar um Ambiente Virtual

Use um ambiente virtual para isolar as dependências do projeto.

```bash
python3 -m venv venv
```

### 3. Ativar o Ambiente Virtual

* **No Windows:**

    ```bash
    .\venv\Scripts\activate
    ```

* **No macOS/Linux:**

    ```bash
    source venv/bin/activate
    ```

Você verá `(venv)` no início do seu prompt de comando, indicando que o ambiente virtual está ativo.

### 4. Instalar as Dependências

Veja o `requirements.txt` na raiz do seu projeto com o seguinte conteúdo:

```
Flask
scikit-learn
google-generativeai
numpy
```

Agora, instale as dependências usando pip:

```bash
pip3 install -r requirements.txt
```

### 5. Configurar a Chave da API Gemini

No seu arquivo `app.py`, você tem a seguinte linha:

```python
genai.configure(api_key="*******")
```

**É crucial que você substitua `"******"` pela sua chave de API real do Google Gemini.** Você pode obter uma chave de API no [Google AI Studio](https://aistudio.google.com/prompts/).

**Recomendação de Segurança:** Para ambientes de produção, é uma boa prática não embutir a chave de API diretamente no código. Em vez disso, use variáveis de ambiente. Esse exemplo, é apenas um teste local com Flask.

### 6. Executar a Aplicação Flask

Com o ambiente virtual ativado e as dependências instaladas, você pode executar o aplicativo:

```bash
python3 app.py
```

O aplicativo será iniciado e geralmente estará disponível em `http://127.0.0.1:5000/` no seu navegador.

### 7. Acessar a Interface Web

Abra seu navegador e navegue até `http://127.0.0.1:5000/`. Você verá uma interface simples onde poderá digitar suas perguntas sobre o TabNews e receber respostas baseadas no conteúdo do FAQ com resposta fluída por Gemini.

## Estrutura do Projeto

* `app.py`: O arquivo principal da aplicação Flask, contendo a lógica do chatbot e as rotas.
* `templates/index.html`: O arquivo HTML que define a interface do usuário.
* `requirements.txt`: Lista das dependências do Python.

## Como Funciona

1.  **Carregamento do FAQ:** O conteúdo do FAQ do TabNews é carregado e dividido em documentos.
2.  **Vetorização TF-IDF:** Os documentos do FAQ são vetorizados usando TF-IDF para criar representações numéricas do texto.
3.  **Busca de Relevância:** Quando uma pergunta é feita, ela é vetorizada e comparada com os documentos do FAQ para encontrar os mais relevantes usando similaridade de cosseno.
4.  **Geração de Resposta:** Os documentos relevantes são passados como contexto para o modelo `Gemini 1.5 Flash`, que então gera uma resposta concisa e natural para a pergunta do usuário, utilizando apenas as informações fornecidas e com modelo ultrarrápido.
