import google.generativeai as genai
import json
import math
import os

# 1. Configure Gemini
gemini_api_key =  os.environ["GEMINI_API_KEY"]
genai.configure(api_key=gemini_api_key)

# -------- Tools --------
def calculator(expression: str):
    """Evaluates a mathematical expression."""
    return str(eval(expression))


def word_count(text: str):
    return str(len(text.split()))


TOOLS = {
    "calculator": calculator,
    "word_count": word_count,
}

# -------- Agent --------
SYSTEM_PROMPT = """
You are an autonomous agent.

You have access to the following tools:

1. calculator
   - description: Evaluates mathematical expressions
   - input: a string mathematical expression

2. word_count
   - description: Counts words in a sentence
   - input: a string of text

When you need a tool, respond EXACTLY in JSON:
{
  "action": "tool_name",
  "input": "tool input"
}

If no tool is needed, respond with:
FINAL ANSWER: <your answer>
"""

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_PROMPT
)


def run_agent(task: str, max_steps=5):
    context = task

    for step in range(max_steps):
        print(f"\n--- Step {step+1} ---")
        response = model.generate_content(context).text
        print("LLM:", response)

        if response.startswith("FINAL ANSWER"):
            return response

        try:
            tool_request = json.loads(response)
            tool_name = tool_request["action"]
            tool_input = tool_request["input"]

            if tool_name not in TOOLS:
                context = f"""
                Error:
                The tool '{tool_name}' is not available.
                Available tools are: {', '.join(TOOLS.keys())}
                Please choose a valid tool or finish without using a tool.
                """
                continue

            print(f"🔧 Using tool: {tool_name}")
            observation = TOOLS[tool_name](tool_input)
            print(f"📌 Observation: {observation}")

            context = f"""
            Tool result:
            {observation}

            Continue reasoning.
            """
        except json.JSONDecodeError:
            return "❌ Invalid agent response format"

    return "❌ Agent did not finish."


if __name__ == "__main__":
    print(
        run_agent(
            "How many words are in this sentence: 'Agentic AI changes software design forever'"
        )
    )
