import google.generativeai as genai
import os

# 1. Configure Gemini
gemini_api_key =  os.environ["GEMINI_API_KEY"]
genai.configure(api_key=gemini_api_key)

# 2. Define the Multi-Agent System
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
EXECUTOR_PROMPT = """
You are an execution agent.

Your job:
- Execute ONLY the given step
- Be precise and concise
"""

executor = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=EXECUTOR_PROMPT
)


def run_multi_agent_system(task: str):

    # 1. Planning
    plan = planner.generate_content(task).text  #(like response.text from planner response object)
    print("\n🧠 PLAN:\n", plan)

    # 2. Execution
    steps = [line for line in plan.split("\n") if line.strip().startswith(tuple("123456789"))]

    results = []
    for step in steps:
        print(f"\n⚙️ Executing: {step}")
        result = executor.generate_content(step).text
        results.append(result)

    return "\n\n".join(results)


if __name__ == "__main__":
    output = run_multi_agent_system(
        "Write a short explanation of Agentic AI for senior backend engineers"
    )
    print("\n✅ FINAL OUTPUT:\n", output)
