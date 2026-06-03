from dotenv import load_dotenv

load_dotenv()

#using ollama python sdk
import ollama

MAX_ITERATIONS = 10
MODEL = "qwen3.5:4b"

#import langsmith traceable func
from langsmith import traceable

@traceable(run_type='tool')
def get_product_price(product: str) -> float:
    """ look up the price of a product in the catalog"""
    print(f">>Executing get_product_price (product='{product}')")
    prices = { "laptop":1299, "headphones":149.95, "keyboard":89.5}
    return prices.get(product,0)

@traceable(run_type='tool')
def apply_discount(price: float, discount_tier: str ) -> float:
    """ Appy a discount tier to a price and return the final price
        Available tiers: bronze, silver, gold."""
    print(f">>Executing apply_discount(price={price}, discount_tier={discount_tier})")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1-discount/100),2)

#Difference 2: Without @tool, we must MANUALLY define the JSON schema for each func.
# This is exactly what Langchain's @tool decorator generates
# automatically from the func type hints and docstring

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.

#-- HELPER: TRACED OLLAMA CALL --
# Difference 3: without langchain, we must manually trace LLM calls for langsmith
@traceable(name='Ollama Chat', run_type='llm')
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, messages=messages, tools=tools_for_llm)

# -- AGENT LOOP --
@traceable(name="LangChain Agent Loop")
def run_agent(question:str):
    tools = [get_product_price,apply_discount]
    tools_dict = {
        "get_product_price":get_product_price,
        "apply_discount":apply_discount,
    }

    #Difference 4: bind_tools()
    # llm_with_tools = llm.bind_tools(tools)

    print(f'Question: {question}')
    print("="*60)

    messages = [
        {
            "role":"system",
            "content": (
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
                "ask them which tier to use - do Not assume one\n"
                "IMPORTANT:\n"
                "If the user mentions 'laptop', 'headphones', or 'keyboard', "
                "you MUST immediately call get_product_price.\n"
                "Do not ask clarifying questions.\n"
            ),
         },
        {"role":"user","content":question},
    ]
    for iteration in range(1,MAX_ITERATIONS+1):
        print(f"\n--Iteration {iteration}--")

        #DIFFERENCE 5: ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(messages=messages)
        ai_message = response['message']

        tool_calls = ai_message.tool_calls

        #if no tool calls, this is the final answer
        if not tool_calls:
            print(f"\n Final Answer: {ai_message.content}")
            return ai_message.content

        #process only the first tool call - force one tool per iteration
        tool_call = tool_calls[0]

        #difference 6: Attribute access (.function.name) instead of dict aceess (.get('name'))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"[Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} does not exist")


        #Difference 7: Direct function call instead tool.invoke()
        observation = tool_to_use(**tool_args)
        print(f"[Tool Result] {observation}")

        # -- OBSERVATION --
        #   feeding the tool result back to LLM memory
        #   so ai can remember the result of each iteration
        messages.append(ai_message)
        messages.append(
            {'role':'tool',
             'content':str(observation),
             }
        )

    print("Error: Max Iterations reached")
    return None

if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a headphones after applying a gold discount")

