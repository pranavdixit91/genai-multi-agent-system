import os
import google.generativeai as genai

# 1. Configure Gemini
gemini_api_key =  os.environ["GEMINI_API_KEY"]
genai.configure(api_key=gemini_api_key)

# 2. Define the Agent's "role"
SYSTEM_PROMPT = """
You are a careful, structured AI agent.
You think step-by-step before answering.
Your goal is to give precise, actionable answers.
"""

# 3. Create the model
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_PROMPT
)

# 4. Run the agent
def run_agent(user_task: str):
    response = model.generate_content(user_task)
    return response.text


if __name__ == "__main__":
    task = "Explain Agentic AI in 5 bullet points"
    print(run_agent(task))


### So it's basically as good as doing a simple LLM call (Chat Completions), but with a system prompt to set the "role" ###
### So this is still not an "agent" in the sense of having tools, memory, or planning. ###