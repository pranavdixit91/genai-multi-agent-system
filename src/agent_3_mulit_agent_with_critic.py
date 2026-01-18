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

# -------- Critic Agent --------
CRITIC_PROMPT = """
You are a critic agent.

Your job:
- Review the given output
- Make sure plot summary, main characters, and overall rating are covered in Movie Review
- Identify missing points, weak reasoning, or clarity issues
- Do NOT rewrite the content
- Respond with:
  - PASS (if output is good)
  - or CRITIQUE with bullet points
"""

critic = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=CRITIC_PROMPT
)

def critique_output(output: str):
    return critic.generate_content(output).text


 # Run the multi-agent system
def run_multi_agent_system(task: str):

    # 1. Planning
    print("Planning the task...")
    plan = planner.generate_content(task).text  #(like response.text from planner response object)
    print("\n🧠 PLAN:\n", plan)

    # 2. Execution
    print("Executing the plan...")
    steps = [line for line in plan.split("\n") if line.strip().startswith(tuple("123456789"))]

    results = []
    for step in steps:
        print(f"\n⚙️ Executing: {step}")
        result = executor.generate_content(step).text
        results.append(result)

    # 3. Critique
    print("Critiquing the output...")
    critique_output_text = critique_output("\n\n".join(results))
    print("\n🧐 CRITIQUE:\n", critique_output_text)    

    if "PASS" in critique_output_text: 
        print("Output passed the critique.")
        return "\n\n".join(results)
    else:
        print("Output did not pass the critique.")
        return "Output did not pass the critique. Please refine below points and try again.".join(critique_output_text)

if __name__ == "__main__":
    output = run_multi_agent_system(
        "Write a review of Movie Fight Club including plot summary, main characters, and overall rating"
    )
    print("\n✅ FINAL OUTPUT:\n", output)
