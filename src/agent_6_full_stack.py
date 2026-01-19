import google.generativeai as genai
import json
import os

# 1. Configure Gemini
gemini_api_key =  os.environ["GEMINI_API_KEY"]
genai.configure(api_key=gemini_api_key)

# -------- Tools --------
def calculator(expression: str):
    return str(eval(expression))


def word_count(text: str):
    return str(len(text.split()))

# Mapping tool names to functions
TOOLS = {
    "calculator": calculator,
    "word_count": word_count
}

# -------- Planner Agent --------
PLANNER_PROMPT = """
You are a planning agent.

Your job:
- Break the user goal into clear, ordered steps
- Do NOT execute the steps
- Output steps as a numbered list
"""

planner = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=PLANNER_PROMPT
)

# -------- Executor Agent --------
EXECUTOR_PROMPT = f"""
You are an execution agent.

Available tools:
{list(TOOLS.keys())}

When using a tool, respond in JSON:
{{ "action": "<tool_name>", "input": "<input>" }}

Otherwise respond:
FINAL ANSWER: <answer>
"""

executor = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=EXECUTOR_PROMPT
)


# -------- Critic Agent --------
CRITIC_PROMPT = """
You are a critic agent.

You will be given:
- The original user goal
- The executor's final answer

Your job:
- Judge whether the answer satisfies the goal
- Do NOT assume tool usage unless explicitly required
- Do NOT invent missing context
- If the answer is clear, relevant, and complete for the goal, respond with:
  PASS
- Otherwise respond with:
  CRITIQUE:
  - specific issue

Do NOT suggest tools unless the goal explicitly requires them.
"""

critic = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=CRITIC_PROMPT
)

# Multi-Agent Full Stack System
def run_full_agent(goal: str, max_retries=3):
    plan = planner.generate_content(goal).text
    print("\n🧠 PLAN:\n", plan)

    steps = [s for s in plan.split("\n") if s.strip().startswith(tuple("123456789"))]
    outputs = []

    for step in steps:
        context = step
        for _ in range(max_retries):
            response = executor.generate_content(context).text
            print("\nExecutor:", response)

            if response.startswith("FINAL ANSWER"):
                answer = response.replace("FINAL ANSWER:", "").strip()

                critique = critic.generate_content(f"""
                            GOAL: 
                            {goal}

                            ANSWER:
                            {answer}
                            """).text
                
                print("Critic:", critique)

                if critique.startswith("PASS"):
                    outputs.append(answer)
                    break
                else:
                    context = f"""
                                {answer}
                                Improve based on critique:
                                {critique}
                                """
            else:
                try:
                    tool_req = json.loads(response)
                    tool = tool_req["action"]

                    if tool not in TOOLS:
                        context = f"""
                                    ERROR: Tool '{tool}' not available.
                                    Available tools: {list(TOOLS.keys())}
                                    """
                        continue

                    observation = TOOLS[tool](tool_req["input"])
                    context = f"Tool result: {observation}"

                except Exception as e:
                    context = f"""
                            Original task:
                            {step}

                            An error occurred while handling a tool request:
                            {e}

                            If a tool is not required, answer directly.
                            If a tool is required, issue a correct tool request.
                            """

    return "\n\n".join(outputs)


if __name__ == "__main__":
    print(
        run_full_agent(
            "Talk about recursion in programming in 30 words",
        )
    )
