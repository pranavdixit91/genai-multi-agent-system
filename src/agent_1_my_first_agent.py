import google.generativeai as genai
import os

# 1. Configure Gemini
gemini_api_key =  os.environ["GEMINI_API_KEY"]
genai.configure(api_key=gemini_api_key)

# 2. Define the Agent's "role" with iterative thinking and self-refinement
SYSTEM_PROMPT = """
You are an autonomous AI agent.

You must:
1. Think about the task
2. Decide if the task is fully completed
3. If not, refine your answer

When you are done, clearly say:
FINAL ANSWER:
"""

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_PROMPT
)

def run_agent(task: str, max_steps=3):
    context = task

    for step in range(max_steps):
        print(f"\n--- Agent Step {step+1} ---")
        response = model.generate_content(context)
        output = response.text
        print(output)

        if "FINAL ANSWER:" in output:
            return output

        # Feed output back into itself (memory + loop)
        context = f"""
                    Previous attempt:
                    {output}

                    Improve or complete the task.
                    """

    return "Agent stopped without final answer."


if __name__ == "__main__":
    print(run_agent("Explain Piyush Mishra's writing style in 5 bullet points"))

