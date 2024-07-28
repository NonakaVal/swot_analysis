import os
from crewai_tools import ScrapeWebsiteTool, WebsiteSearchTool
from langchain_openai import ChatOpenAI
import openai
from tools.browser_tools import BrowserTools
from tools.search_tools import SearchTools
from tools.PDFSelector import select_output_directory
from crewai import Agent, Task

# CONFIGURAÇÕES DE API
openai.api_key = os.getenv('OPENAI_API_KEY')
os.environ["BROWSERLESS_API_KEY"] = os.getenv('BROWSERLESS_API_KEY')
llm = ChatOpenAI(model='gpt-3.5-turbo')

# ARQUIVOS
nicho = input("Digite o nicho da pesquisa: \n")
produto_ou_servico = input("Digite o produto ou serviço da pesquisa: \n")
output_directory = select_output_directory()

# Agentes para Análise SWOT

chief_researcher = Agent(
    role=f'Pesquisador chefe Especialista em {nicho}',
    goal='Coordenar a equipe de análise SWOT e garantir a coesão das análises realizadas pelos especialistas.',
    backstory='Com ampla experiência em pesquisa de mercado, este agente é responsável por supervisionar todas as etapas da análise SWOT.',
    llm=llm,
    allow_delegation=True,
    tools=[SearchTools.search_internet, BrowserTools.scrape_and_summarize_website, ScrapeWebsiteTool(), WebsiteSearchTool()]
)

strengths_expert = Agent(
    role=f'Especialista senior em {nicho}',
    goal=f'Identificar e analisar detalhadamente os pontos fortes do {produto_ou_servico}.',
    backstory=f'mais de 20 anos de experiência em análise de vantagens competitivas de {nicho}',
    llm=llm,
    allow_delegation=False,
    tools=[SearchTools.search_internet, BrowserTools.scrape_and_summarize_website, ScrapeWebsiteTool(), WebsiteSearchTool()]
)

weaknesses_expert = Agent(
    role=f'Especialista senior em {nicho}',
    goal=f'Identificar as principais fraquezas e limitações do {produto_ou_servico}.',
    backstory=f'mais de 20 anos de experiência em análise de vantagens competitivas de {nicho}',
    llm=llm,
    allow_delegation=False,
    tools=[SearchTools.search_internet, BrowserTools.scrape_and_summarize_website, ScrapeWebsiteTool(), WebsiteSearchTool()]
)

opportunities_expert = Agent(
    role=f'Especialista senior em estratégias de inovação em {nicho}',
    goal=f'identificar e explorar as principais oportunidades de crescimento e expansão no mercado de {produto_ou_servico}',
    backstory=f'mais de 20 anos de experiência em identificar oportunidades de mercado de {produto_ou_servico}, este agente pode explorar novas tendências e áreas de inovação.',
    llm=llm,
    allow_delegation=False,
    tools=[SearchTools.search_internet, BrowserTools.scrape_and_summarize_website, ScrapeWebsiteTool(), WebsiteSearchTool()]
)

threats_expert = Agent(
    role=f'Especialista senior em {nicho}',
    goal=f'Identificar ameaças externas que podem impactar negativamente o {nicho} e o {produto_ou_servico}.',
    backstory=f'Com 20 anos de experiência em análise de ameaças do mercado, este agente pode identificar fatores externos que impactam negativamente o {nicho} e o {produto_ou_servico}.',
    llm=llm,
    allow_delegation=False,
    tools=[SearchTools.search_internet, BrowserTools.scrape_and_summarize_website, ScrapeWebsiteTool(), WebsiteSearchTool()]
)

# Tarefas para Análise SWOT

chief_researcher_task = Task(
    description=f"""
    Coordenar e supervisionar todas as etapas da análise SWOT para o {produto_ou_servico} no mercado de {nicho}. As etapas incluem:
    1. Definição de Objetivos: Clarificar os objetivos específicos da análise SWOT.
    2. Planejamento da Pesquisa: Desenvolver um plano detalhado, incluindo metodologias, fontes de dados, cronograma e responsáveis.
    3. Coleta de Dados: Supervisionar a coleta de dados primários e secundários.
    4. Análise de Dados: Coordenar a análise dos dados para gerar insights significativos.
    5. Relatórios e Apresentações: Preparar relatórios detalhados e apresentações dos resultados.
    6. Validação dos Dados: Garantir a precisão dos relatórios gerados.
    7. Referências: Incluir links e referências para todas as fontes e sites utilizados na análise.
    """,
    expected_output=f"Relatório de planejamento de análise SWOT do {produto_ou_servico} no mercado de {nicho}, integrando as análises de forças, fraquezas, oportunidades e ameaças, e incluindo links para as referências.",
    agent=chief_researcher,
    output_file=os.path.join(output_directory, "1-planejamento.md")
 
)


strengths_task = Task(
    description=f"""
    Identificar e analisar detalhadamente os pontos fortes do {produto_ou_servico} no mercado de {nicho}. complete os seguintes tópicos:
    1. Missão e visão das principais empresas do {nicho} para entender sua finalidade e objetivos.
    2. Coletar informações sobre os {produto_ou_servico} ou iniciativas oferecidas.
    3. Analisar dados de desempenho anteriores, como vendas, crescimento, satisfação do cliente ou métricas de sucesso.
    4. identificar recursos únicos, como tecnologias, habilidades ou experiências da equipe que conferem vantagem competitiva
    """,
    expected_output=f"Relatório detalhada com uma lista dos principais pontos fortes do {produto_ou_servico} considerando seu impacto e relevância no {nicho}",
    agent=strengths_expert,
    output_file=os.path.join(output_directory, "2-strengths.md"),
    contex = [chief_researcher_task]
)

weaknesses_task = Task(
    description=f"""
    Identificar e analisar as principais fraquezas e limitações do {produto_ou_servico} no mercado de {nicho}. complete os seguintes tópicos:
    1. Revisar as informações e dados existentes sobre a {nicho} ou projeto para entender o contexto atual de {produto_ou_servico}.
    2. Revisar as informações e dados existentes sobre {nicho} para entender o contexto atual.
    3. Observar e avaliar a concorrência para entender as lacunas em comparação com outras empresas ou projetos similares.
    4. Priorizar os pontos fracos identificados com base em seu impacto e relevância para a organização ou projeto
    """,
    expected_output=f"Relatório detalhada com uma lista dos principais pontos fracos do {produto_ou_servico} considerando seu impacto e relevância no {nicho}",
    agent=weaknesses_expert,
    output_file=os.path.join(output_directory, "3-weakness.md"),
    contex = [chief_researcher_task]
)

opportunities_task = Task(
    description=f"""
    Explorar e identificar oportunidades de crescimento e expansão para o {produto_ou_servico} no mercado de {nicho}. complete os seguintes tópicos:
    1. Realizar uma pesquisa de mercado para compreender as tendências atuais no {nicho} e {produto_ou_servico}.
    2. Analisar os principais empresas de {nicho} e suas estratégias.
    3. Identificar lacunas no mercado de {nicho} que não estão sendo atendidas.
    4. Avaliar mudanças econômicas, sociais e tecnológicas que podem criar novas oportunidades.
    5. Classificar as oportunidades encontradas com base em sua viabilidade e potencial de impacto
    """,
    expected_output=f"Relatório detalhado com uma lista das principais oportunidades identificadas para o {produto_ou_servico}, classificadas com base na viabilidade e potencial de impacto no {nicho}.",
    agent=opportunities_expert,
    output_file=os.path.join(output_directory, "4-opportunities.md"),
    contex = [chief_researcher_task]
)

threats_task = Task(
    description=f"""
    Identificar e analisar ameaças externas que podem impactar negativamente o {produto_ou_servico} no mercado de {nicho}. complete os seguintes tópicos:
    1. usar o contexto dos resultados anteriores
    2. Revisar informações sobre o {nicho}, concorrência e tendências que possam indicar ameaças
    3. Listar ameaças externas, como mudanças regulamentares, crises econômicas ou desastres naturais
    4. listar ameaças internas, como falta de pessoal qualificado ou problemas de financiamento
    5. identificar vulnerabilidades que possam ser exploradas
    """,
    expected_output=f"Relatório detalhado com uma lista das principais ameaças identificadas para o {produto_ou_servico}, ameaças com base em seu impacto e probabilidade de ocorrência para o mercado {nicho}",
    agent=threats_expert,
    output_file=os.path.join(output_directory, "5-ameacas_analise_swot.md"),
    contex = [chief_researcher_task, strengths_task, weaknesses_task, opportunities_task]
)

