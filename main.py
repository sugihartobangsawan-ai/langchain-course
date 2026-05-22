from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

class Source(BaseModel):
    """
    Schema for a source used by the agent
    """
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """
    Schema for agent response with answer and sources
    """
    answer:str = Field(description="The agent's answer to the query")
    source: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools=[TavilySearch()]
agent = create_agent(llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from react-search-agent!")
    result = agent.invoke({"messages":HumanMessage(content="search for 3 most common monetization using langchain skill")})
    print(result)
    # Assuming 'result' is the variable holding your agent's output
    response_object = result["structured_response"]

    # 1. Get the answer
    final_answer = response_object.answer
    print("Answer:", final_answer)

    # 2. Get the sources (looping through the list of Source objects)
    for s in response_object.source:
        print("Source URL:", s.url)


if __name__ == "__main__":
    main()
