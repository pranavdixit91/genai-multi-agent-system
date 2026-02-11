import google.generativeai as genai
import json
import os

# ================== CONFIG ==================
gemini_api_key =  os.environ["GEMINI_API_KEY"]
genai.configure(api_key=gemini_api_key)
MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3

# ================== PLANNER ==================
planner = genai.GenerativeModel(
    model_name=MODEL,
    system_instruction="""
You are a planning agent.

Your job:
- Understand the goal thoroughly
- Break it into small, achievable steps
- Do NOT execute any step

Output format:
1. Step one
2. Step two
...
"""
)

# ================== EXECUTOR ==================
executor = genai.GenerativeModel(
    model_name=MODEL,
    system_instruction="""
You are an execution agent.

Rules:
- You are given a GOAL, a CURRENT STEP, and OBSERVATIONS
- Use OBSERVATIONS only as feedback, not as a new task
- If the task can be answered directly, respond with:

FINAL ANSWER:
<answer>

- Do NOT assume tool usage unless explicitly required
- If an error occurred previously, re-evaluate calmly
"""
)

# ================== CRITIC ==================
critic = genai.GenerativeModel(
    model_name=MODEL,
    system_instruction="""
You are a critic agent.

You will receive:
- GOAL
- CURRENT STEP
- ANSWER

Your job:
- Check whether the answer satisfies the goal or at least current step of the Goal
- Check whether the answer partially satisfies the goal
- Do NOT invent missing context
- Do NOT suggest tools unless the goal explicitly needs them

Respond ONLY with:
PASS
or
CRITIQUE:
- specific issue
"""
)

# ================== ORCHESTRATOR ==================
def run_agent_with_short_memory(goal: str):
    print("\n🎯 GOAL:\n", goal)

    plan_text = planner.generate_content(goal).text
    print("\n🧠 PLAN:\n", plan_text)

    steps = [
        line for line in plan_text.split("\n")
        if line.strip().startswith(tuple("123456789"))
    ]

    final_outputs = []

    for step in steps:
        print(f"\n➡️ EXECUTING STEP: {step}")

        agent_state = {
            "goal": goal,
            "step": step,
            "observations": [],
            "attempts": 0
        }

        while agent_state["attempts"] < MAX_RETRIES:
            agent_state["attempts"] += 1

            response = executor.generate_content(f"""
                                                    GOAL:
                                                    {agent_state['goal']}

                                                    CURRENT STEP:
                                                    {agent_state['step']}

                                                    OBSERVATIONS:
                                                    {agent_state['observations']}

                                                    Decide next action.
                                                    """).text.strip()

            print(f"\nExecutor Attempt {agent_state['attempts']}:\n{response}")

            # ---- Final Answer Path ----
            if response.startswith("FINAL ANSWER"):
                answer = response.replace("FINAL ANSWER:", "").strip()

                critique = critic.generate_content(f"""
                                                    GOAL:
                                                    {goal}

                                                    Current Step:
                                                    {agent_state['step']}

                                                    ANSWER:
                                                    {answer}
                                                    """).text.strip()

                print("\n🧐 CRITIC:", critique)

                if critique == "PASS":
                    final_outputs.append(answer)
                    break
                else:
                    agent_state["observations"].append(critique)
                    continue

            # ---- Invalid / Unexpected Output ----
            else:
                agent_state["observations"].append(
                    "Executor did not provide FINAL ANSWER. Retry with clarity."
                )

        else:
            final_outputs.append(
                "❌ Step failed after retries."
            )

    return "\n\n".join(final_outputs)


# ================== RUN ==================
if __name__ == "__main__":
    result = run_agent_with_short_memory(
        "Explain the theory of relativity in simple terms"
    )

    print("\n✅ FINAL OUTPUT:\n")
    print(result)
