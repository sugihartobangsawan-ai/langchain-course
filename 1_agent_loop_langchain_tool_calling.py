from dotenv import load_dotenv

load_dotenv()

#to initialize chat model with just a string
from langchain.chat_models import init_chat_model

from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MAX_ITERATIONS = 10
MODEL = "qwen3.5:4b"

#import langsmith traceable func
from langsmith import traceable

# -- TOOL DECORATOR --
# tool decorator in langchain will format the function name, input, data type, docstring and pass it into llm

@tool
def get_product_price(product: str) -> float:
    """ look up the price of a product in the catalog"""
    print(f">>Executing get_product_price (product='{product}')")
    prices = { "laptop":1299, "headphones":149.95, "keyboard":89.5}
    return prices.get(product,0)

@tool
def apply_discount(price: float, discount_tier: str ) -> float:
    """ Appy a discount tier to a price and return the final price
        Available tiers: bronze, silver, gold."""
    print(f">>Executing apply_discount(price={price}, discount_tier={discount_tier})")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1-discount/100),2)


# -- AGENT LOOP --
@traceable(name="LangChain Agent Loop")
def run_agent(question:str):
    tools = [get_product_price,apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm=init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f'Question: {question}')
    print("="*60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant."
                "You have access to a product catalog tool"
                "and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1.NEVER guess or assume any product price."
                "You must call get_product_price first to get the real price \n"
                "2. Only call apply_discount AFTER you have received"
                "a price from get_product_price. Pass the exact price"
                "returned by get_product_price - do NOT pass a made-up number\n"
                "3. NEVER calculate discount yourself using math."
                "Always use the apply_discount tool\n"
                "4. If the user does not specify a discount tier,"
                "ask them which tier to use - do Not assume one"
            )
        ),
        HumanMessage(content=question)
    ]
    for iteration in range(1,MAX_ITERATIONS+1):
        print(f"\n--Iteration {iteration}--")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        #if no tool calls, this is the final answer
        if not tool_calls:
            print(f"\n Final Answer: {ai_message.content}")
            return ai_message.content

        #process only the first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")

        print(f"[Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} does not exist")
        observation=tool_to_use.invoke(tool_args)

        print(f"[Tool Result] {observation}")

        #so ai can remember the result of each iteration
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id = tool_call_id)
        )

    print("Error: Max Iterations reached")
    return None

if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount")

