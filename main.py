from crewai import Crew, Process
from langchain_openai import ChatOpenAI

# Importando agentes e tarefas específicos do seu código
from crew import (
   chief_researcher,strengths_expert, weaknesses_expert, opportunities_expert, threats_expert,
   chief_researcher_task, strengths_task, weaknesses_task,opportunities_task,threats_task
)

# Configurando o modelo LLM
llm = ChatOpenAI(model='gpt-3.5-turbo')

# Configurando a equipe (Crew)
crew = Crew(
    agents=[chief_researcher,strengths_expert, weaknesses_expert, opportunities_expert, threats_expert],
    tasks=[chief_researcher_task, strengths_task, weaknesses_task,opportunities_task,threats_task],
    process=Process.sequential,  # Processo sequencial para execução das tarefas
    manager_llm=llm  # Definindo o modelo LLM para gerenciar a equipe
)

# Iniciando o processo com um caminho de transcrição específico (pdf_path)
result = crew.kickoff()
print(result)
