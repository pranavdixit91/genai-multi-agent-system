import google.generativeai as genai
import os

# 1. Configure Gemini
gemini_api_key =  os.environ["GEMINI_API_KEY"]
genai.configure(api_key=gemini_api_key)

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
- Review the given output which is Movie Review
- Make sure plot summary, main characters, and overall ratings are covered in output
- Make sure output is in Hindi Language
- Do NOT rewrite the content
- Respond with:
  - PASS (if output is good)
  - or CRITIQUE with bullet points
"""

critic = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=CRITIC_PROMPT
)


# Multi-Agent Fully Autonomous System
def autonomous_multi_agent_run(goal: str, max_retries=3):
    plan = planner.generate_content(goal).text
    print("\n🧠 PLAN:\n", plan)

    steps = [s for s in plan.split("\n") if s.strip().startswith(tuple("123456789"))]

    final_output = []

    for step in steps:
        print(f"\n⚙️ Step: {step}")
        for attempt in range(max_retries):
            result = executor.generate_content(step).text
            critique = critic.generate_content(result).text

            print(f"\nAttempt {attempt+1}")
            print("Result:", result)
            print("Critic:", critique)

            if critique.strip().startswith("PASS"):
                final_output.append(result)
                break
            else:
                # Feed critique back to executor (not rewrite)
                step = f"""
                Original task:
                {step}

                Critic feedback:
                {critique}

                Improve your execution.
                """

    return "\n\n".join(final_output)


if __name__ == "__main__":

    user_instruction = "Review the Movie Fight Club including plot summary, main characters, and overall rating in Hindi"
    print("Multi-Agent Fully Autonomous System Running for instruction:", user_instruction)
    
    # call the autonomous run function
    result = autonomous_multi_agent_run(
            goal=user_instruction
            , max_retries=3
        )
    print("\n✅ FINAL OUTPUT:\n", result)
