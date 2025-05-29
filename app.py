from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import numpy as np

app = Flask(__name__)

genai.configure(api_key="https://aistudio.google.com/prompts/")
model_gemini = genai.GenerativeModel('gemini-1.5-flash')

markdown_content = """
Executando verificação de segurança...

# FAQ - Perguntas Frequentes

- [O que é o TabNews?](https://www.tabnews.com.br/faq#tabnews)
- [Qual é o propósito do TabNews?](https://www.tabnews.com.br/faq#proposito-tabnews)
- [Que tipo de conteúdo eu posso publicar no TabNews?](https://www.tabnews.com.br/faq#conteudo-tabnews)
- [Que tipo de assunto é aceito no TabNews?](https://www.tabnews.com.br/faq#assunto-tabnews)
- [Como criar um bom conteúdo no TabNews?](https://www.tabnews.com.br/faq#qualidade-tabnews)
- [O que é TabCash?](https://www.tabnews.com.br/faq#tabcash)
- [Como ganhar TabCash?](https://www.tabnews.com.br/faq#ganhar-tabcash)
- [Como utilizar meu TabCash?](https://www.tabnews.com.br/faq#utilizar-tabcash)
- [Como funciona uma publicação patrocinada?](https://www.tabnews.com.br/faq#publicacao-patrocinada)
- [O que é TabCoin?](https://www.tabnews.com.br/faq#tabcoin)
- [Como ganhar TabCoins?](https://www.tabnews.com.br/faq#ganhar-tabcoins)
- [É possível perder TabCoins?](https://www.tabnews.com.br/faq#perder-tabcoins)
- [Como utilizar meus TabCoins?](https://www.tabnews.com.br/faq#utilizar-tabcoins)
- [Posso criar publicações divulgando projetos em que estou envolvido?](https://www.tabnews.com.br/faq#publicar-projeto-envolvido)
- [Posso publicar o mesmo conteúdo várias vezes?](https://www.tabnews.com.br/faq#publicar-mesmo-conteudo)
- [Não consigo criar novas publicações. O que fazer?](https://www.tabnews.com.br/faq#erro-nova-publicacao)
- [Como funciona a página "Relevantes"?](https://www.tabnews.com.br/faq#como-relevantes)
- [Onde posso fazer sugestões e/ou reportar bugs?](https://www.tabnews.com.br/faq#sugestoes-e-bugs)
- [Como posso fazer testes no site do TabNews?](https://www.tabnews.com.br/faq#testar-tabnews)
- [Como posso contribuir com o TabNews?](https://www.tabnews.com.br/faq#contribuir-tabnews)

## O que é o TabNews?

O TabNews é um site focado na comunidade da área de tecnologia, destinado a debates e troca de conhecimentos por meio de publicações e comentários criados pelos próprios usuários.

## Qual é o propósito do TabNews?

O TabNews nasceu com o objetivo de ser um local com **conteúdos de valor concreto para quem trabalha com tecnologia**.

Queremos ter conteúdo de qualidade tanto na publicação principal quanto nos comentários, e algo que contribui para isso acontecer é a plataforma dar o mesmo espaço de criação para quem está publicando ambos os tipos de conteúdo. Tudo no TabNews é considerado um **conteúdo**, tanto que um comentário possui a sua própria página (basta clicar na data de publicação do comentário).

## Que tipo de conteúdo eu posso publicar no TabNews?

Você pode publicar notícias, artigos, tutoriais, indicações, curiosidades, sugestões de software e ferramentas, perguntas bem formuladas ou outros tipos de conteúdo, desde que o assunto da publicação seja [aceito no TabNews](https://www.tabnews.com.br/faq#assunto-tabnews).

## Que tipo de assunto é aceito no TabNews?

O conteúdo publicado no TabNews deve estar diretamente relacionado à tecnologia. Alguns exemplos de assuntos diretamente relacionados à tecnologia são: desenvolvimento de software, análise de dados, design, inteligência artificial, modelagem 3D, edição de vídeo, manipulação de imagens etc. Exemplos de assuntos indiretamente relacionados à tecnologia, mas que podem ser abordados do ponto de vista da tecnologia, são: produtividade, empreendedorismo, criação de conteúdo etc.

## Como criar um bom conteúdo no TabNews?

A forma como cada pessoa avalia a qualidade de um conteúdo é subjetiva, mas temos algumas recomendações que podem ajudar a criar uma publicação mais relevante:

- **Atenção à gramática e aos erros de digitação:** antes de publicar, confirme se precisa corrigir algum erro gramatical ou de digitação. O uso correto da língua portuguesa ajudará a transmitir a sua mensagem para os leitores.
- **Formate o conteúdo para facilitar a leitura:** o editor de texto do TabNews aceita a sintaxe Markdown, então você pode usá-la para identificar no seu texto títulos e subtítulos, trechos de código, citações, enfatizar trechos específicos, exibir diagramas etc.
- **Use imagens e fontes de apoio quando for apropriado:** nem todo conteúdo precisa de imagens ou links de referência, mas isso pode ajudar a transmitir mais credibilidade e facilitar o entendimento do seu conteúdo. Você também pode disponibilizar links para o leitor se aprofundar no assunto.
- **Transmita informações corretas:** antes de compartilhar um fato ou notícia, confirme se isso é realmente verdade. Se for algo opinativo, deixe claro que está compartilhando a sua opinião ou de um terceiro.

## O que é TabCash?

O TabCash é uma moeda digital para recompensar pessoas que estão criando conteúdos com valor concreto e também ajudando a qualificar outros conteúdos. O saldo de TabCash pode ser utilizado no sistema de Revenue Share, onde você pode usar espaços de anúncio para compartilhar o que desejar, desde que respeite os [Termos de Uso](https://www.tabnews.com.br/termos-de-uso). Esse sistema está em desenvolvimento e você pode [acompanhar o progresso no GitHub](https://github.com/filipedeschamps/tabnews.com.br/issues/1490).

## Como ganhar TabCash?

Para ganhar TabCash, é necessário contribuir com a qualificação de conteúdos de outras pessoas, consumindo 2 TabCoins a cada qualificação realizada e, ao mesmo tempo, ganhando 1 TabCash.

## Como utilizar meu TabCash?

O TabCash pode ser utilizado para publicar o que você quiser em espaços de anúncio, desde que respeite os [Termos de Uso](https://www.tabnews.com.br/termos-de-uso).

Atualmente, o único espaço de anúncio disponível é o de [publicações patrocinadas](https://www.tabnews.com.br/faq#publicacao-patrocinada). Para criar esse tipo de anúncio, acesse a página [Publicar novo conteúdo](https://www.tabnews.com.br/publicar) e marque a caixa de seleção " **Criar como publicação patrocinada**". Você precisa ter ao menos **100 TabCash**, que serão consumidos ao criar a publicação patrocinada.

## Como funciona uma publicação patrocinada?

_Esse tipo de anúncio está em desenvolvimento, então está em constante evolução. Você pode acompanhar o que está sendo feito no [issue #1491 do GitHub](https://github.com/filipedeschamps/tabnews.com.br/issues/1491)._

No topo das listas de conteúdos [Relevantes](https://www.tabnews.com.br/) e [Recentes](https://www.tabnews.com.br/recentes/pagina/1), e também nas páginas de publicações e comentários, após o conteúdo principal, uma publicação patrocinada escolhida de forma aleatória é exibida como um _banner_. Caso a publicação tenha um link de " **fonte**", o visitante que clicar no título da publicação será redirecionado para o link. Caso o link seja para um site externo, o domínio será identificado após o título, por exemplo: `Título da publicação patrocinada (site-externo.com.br)`.

Para criar uma publicação patrocinada, você investirá **100 TabCash** no orçamento dela. Ainda não está definido como o orçamento será consumido e ainda não é possível alterar o valor do orçamento.

Recomendamos que o título tenha até 70 caracteres para que possa ser exibido sem reticências ao final.

## O que é TabCoin?

TabCoin é a moeda de troca no sistema de qualificação de conteúdos do TabNews. Você utiliza seus TabCoins para qualificar conteúdos dos outros e, por sua vez, recebe ou perde TabCoins com base nas qualificações recebidas em seus próprios conteúdos.

## Como ganhar TabCoins?

As formas de ganho de TabCoins são:

- **Criando um conteúdo:** existe um algoritmo que leva em consideração os TabCoins dos seus conteúdos mais recentes para definir quantos TabCoins você ganhará ao criar um novo conteúdo.
- **Recebendo votos positivos:** quando outro usuário avalia positivamente seu conteúdo.
- **Recompensa diária:** você pode ganhar TabCoins ao acessar o TabNews pelo menos uma vez no dia. Existe um algoritmo que leva em consideração as qualificações dos seus conteúdos mais recentes e também a quantidade de TabCoins que você possui. Quanto melhor avaliados forem seus conteúdos e menos TabCoins você possuir, mais receberá na recompensa diária.

## É possível perder TabCoins?

Sim, você pode perder TabCoins:

- **Ao apagar um conteúdo:** você perderá os TabCoins que ganhou ao criar o conteúdo, caso tenha ganhado algum TabCoin, e também perderá os TabCoins que ganhou com as avaliações positivas nessa publicação. O mesmo vale para caso um moderador apague um conteúdo seu.
- **Recebendo votos negativos:** você perderá 1 TabCoin a cada avaliação negativa recebida de outros usuários em seus conteúdos.

## Como utilizar meus TabCoins?

Os TabCoins são utilizados para poder qualificar conteúdos de outros usuários e ajudar a comunidade a identificar conteúdos relevantes.

Ao avaliar uma publicação, serão consumidos 2 TabCoins e creditado 1 TabCash nos seus saldos.

## Posso criar publicações divulgando projetos em que estou envolvido?

Sim, você pode criar uma publicação sobre um projeto que está envolvido desde que agregue valor ao leitor, por exemplo explicando detalhes técnicos do projeto, compartilhando suas experiências na criação, dificuldades e decisões tomadas.

Se você pretende fazer um pitch, ou seja, uma apresentação curta e direta com o objetivo despertar atenção das pessoas para o projeto em si, você deve colocar `Pitch` no título da publicação, por exemplo: `Pitch: TabInvest — Um TabNews sobre investimentos`. Mesmo sendo um pitch você deve contribuir com a comunidade como explicado no parágrafo anterior.

Uma divulgação de um projeto que você está envolvido deve seguir as mesmas regras de qualquer outra publicação: leia os [Termos de Uso](https://www.tabnews.com.br/termos-de-uso) e o tópico [Que tipo de conteúdo eu posso publicar no TabNews?](https://www.tabnews.com.br/faq#publicar-tabnews). Publicações com foco exclusivo comercial são expressamente proibidas.

## Posso publicar o mesmo conteúdo várias vezes?

Não. Se deseja criar uma nova publicação sobre o mesmo assunto, leve em consideração há quanto tempo o conteúdo foi feito e o quão diferente será a nova publicação. Lembre-se que toda publicação está sujeita à qualificação por outros usuários através do uso de TabCoins, e casos de abuso serão tratados pela moderação. Apagar um conteúdo avaliado negativamente e republicá-lo para tentar chamar mais atenção é um exemplo de **manipulação das qualificações** e poderá resultar no banimento permanente da sua conta, como dito nos [Termos de Uso](https://www.tabnews.com.br/termos-de-uso).

## Não consigo criar novas publicações. O que fazer?

Se, ao criar uma nova publicação ou comentário, você recebe uma mensagem de erro dizendo que não é possível publicar porque há outras publicações mal avaliadas que ainda não foram excluídas, revise seus conteúdos mais recentes que estão zerados ou negativados. Essa é uma proteção para o TabNews e para o usuário, impedindo a criação de muitas publicações mal recebidas e permitindo que o usuário analise o que está fazendo de errado e corrija seu comportamento.

Ao encontrar suas publicações que estão qualificadas negativamente, você poderá apagar alguma e tentar criar a publicação que deseja. O TabNews avaliará suas publicações novamente para definir se você pode ou não criar uma nova publicação. Caso receba a mesma mensagem de erro, basta realizar o processo novamente.

## Como funciona a página "Relevantes"?

A página [Relevantes](https://www.tabnews.com.br/) tem como objetivo exibir as publicações recentes que foram mais relevantes para os usuários do TabNews. O algoritmo leva em consideração diferentes fatores como: há quanto tempo a publicação foi feita, quão positivamente ela foi avaliada, se a comunidade engajou por meio de comentários etc.

## Onde posso fazer sugestões e/ou reportar bugs?

Para sugestões de melhorias ou para reportar bugs que não envolvem informações sensíveis ou falhas de segurança, você pode abrir um issue no [repositório do TabNews no GitHub](https://github.com/filipedeschamps/tabnews.com.br).

Caso você descubra alguma falha, brecha ou vulnerabilidade de segurança e encontre **informações sensíveis** (por exemplo, dados privados de outros usuários, dados sensíveis do sistema ou acesso não autorizado), pedimos que [entre em contato de forma privada pelo GitHub](https://github.com/filipedeschamps/tabnews.com.br/security/advisories/new). Você pode seguir [o tutorial do GitHub](https://docs.github.com/pt/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability#privately-reporting-a-security-vulnerability) sobre como fazer esse tipo de relato.

Após o fechamento da falha, o TabNews se compromete em criar um Postmortem público com os detalhes do que aconteceu. Não temos interesse algum em esconder esses acontecimentos e queremos compartilhar todos os conhecimentos adquiridos e estratégias adotadas, mantendo em mente que iremos proteger ao máximo dados sensíveis dos usuários.

## Como posso fazer testes no site do TabNews?

Testes das mais variadas formas devem ser feitos no ambiente de homologação. Você pode acessar a [lista de implantações](https://github.com/filipedeschamps/tabnews.com.br/deployments/activity_log?environment=Preview) e clicar em algum link da seção `Active deployments` para acessar o ambiente. Por ser um ambiente diferente, você precisará criar uma nova conta e confirmar o e-mail.

## Como posso contribuir com o TabNews?

Existem diferentes formas de participação que contribuem para a evolução do TabNews:

- **Criação de conteúdo:** você pode criar publicações ou comentários com conteúdo de valor para outros leitores.
- **Qualificação de conteúdo:** você pode usar seus TabCoins para qualificar as publicações e comentários. Ao qualificar positivamente, você reforça que aquele tipo de conteúdo é relevante e desejado no TabNews. Ao qualificar negativamente, você demonstra que aquele conteúdo não é relevante ou possui algum problema.
- **Participação no repositório:** as sugestões de melhorias e reportes de bugs são realizados no [repositório do TabNews no GitHub](https://github.com/filipedeschamps/tabnews.com.br). Você pode contribuir com detalhes para a resolução de algum problema ou com ideias de implementação de algum recurso.
- **Modificações no código:** como o TabNews é um projeto de código aberto, além de sugerir melhorias e reportar bugs, você também pode contribuir com o código do projeto. Leia o [guia de contribuição](https://github.com/filipedeschamps/tabnews.com.br/blob/main/CONTRIBUTING.md) do projeto para mais detalhes.
"""

documents = [doc.strip() for doc in markdown_content.split('\n\n') if doc.strip()]

vectorizer = TfidfVectorizer()
document_embeddings = vectorizer.fit_transform(documents)

def find_relevant_documents(query, top_k=5):
    query_embedding = vectorizer.transform([query])
    similarities = cosine_similarity(query_embedding, document_embeddings).flatten()
    top_indices = np.argsort(similarities)[::-5][:top_k]
    return [documents[i] for i in top_indices]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'response': 'Por favor, digite uma mensagem.'}), 400

    relevant_docs = find_relevant_documents(user_message)

    context = "\n\n".join(relevant_docs)
    prompt = f"""
    Você é um assistente "Tab News" amigável e prestativo.
    Sua tarefa é responder às perguntas do usuário de forma concisa e natural, utilizando *apenas* as informações fornecidas no contexto abaixo.
    Não repita o texto do contexto palavra por palavra. Em vez disso, reformule e sintetize a informação para criar uma resposta original e útil.
    Se a resposta não puder ser encontrada no contexto fornecido, diga "Desculpe, não encontrei informações sobre isso no nosso catálogo."
    Não invente informações.

    Contexto:
    {context}

    Pergunta do usuário: {user_message}
    """

    try:
        response = model_gemini.generate_content(prompt)
        return jsonify({'response': response.text})
    except Exception as e:
        return jsonify({'response': f'Ocorreu um erro ao processar sua solicitação: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
