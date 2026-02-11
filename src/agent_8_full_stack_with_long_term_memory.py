import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from datetime import datetime
import os

# ================== CONFIG ====================
gemini_api_key =  os.environ["GEMINI_API_KEY"]
genai.configure(api_key=gemini_api_key)
MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3
TOP_K = 3
RELEVANCE_THRESHOLD = 0.55
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ================= MEMORY STORE =================
dimension = 384  # embedding size of MiniLM
index = faiss.IndexFlatIP(dimension)
memory_store = []  # holds metadata + content

def embed(text):
    vec = EMBED_MODEL.encode([text])[0]
    return vec / np.linalg.norm(vec)


def store_memory(content, topic="general"):
    vector = embed(content)
    index.add(np.array([vector]).astype("float32"))
    memory_store.append({
        "content": content,
        "topic": topic,
        "timestamp": str(datetime.now())
    })


def retrieve_relevant_memories(goal):
    if len(memory_store) == 0:
        return []

    goal_vec = embed(goal)
    scores, ids = index.search(np.array([goal_vec]).astype("float32"), TOP_K)

    memories = []

    # Finding out relevant memories based on cosine similarity scores from Embedded Goal using Faiss. Only consider those above a certain relevance threshold.
    for score, idx in zip(scores[0], ids[0]):
        if score > RELEVANCE_THRESHOLD:
            memories.append(memory_store[idx]["content"])

    return memories


# Seed some long-term knowledge
store_memory(
    "Agentic AI systems rely on orchestration logic to manage planning, execution, retries, and role separation.",
    topic="agentic_ai"
)

store_memory(
    "Critic agents should evaluate output without rewriting it to avoid role leakage.",
    topic="agent_design"
)


store_memory(
    "Vector databases enable efficient similarity search for unstructured data, which is crucial for AI applications like recommendation systems and semantic search.",     
    topic="vector_databases"
)

# ================= PLANNER =================
planner = genai.GenerativeModel(
    model_name=MODEL,
    system_instruction="""
You are a planning agent.

Use any provided MEMORY if relevant.

Break the goal into small steps.
Do NOT execute them.
"""
)

# ================= EXECUTOR =================
executor = genai.GenerativeModel(
    model_name=MODEL,
    system_instruction="""
You are an execution agent.

You will receive:
- GOAL
- STEP
- OBSERVATIONS

Keep your answer SHORT (1-2 sentences max).

Answer directly with:

FINAL ANSWER:
<answer>
"""
)

# ================= CRITIC =================
critic = genai.GenerativeModel(
    model_name=MODEL,
    system_instruction="""
You are a critic agent.

You will receive GOAL, One of the STEPs of Overall Goal and ANSWER.

Be lenient. Only critique if the answer is:
- Completely irrelevant to the goal
- Answering a different step than the one provided
- Missing key information
- Factually incorrect

Respond with:
PASS
or
CRITIQUE:
- issue
"""
)

# ================= ORCHESTRATOR =================
def run_agent(goal: str):
    print("\n🎯 GOAL:\n", goal)

    # ---- Long-term memory retrieval BEFORE planning ----
    memories = retrieve_relevant_memories(goal)
    memory_block = "\n".join(memories)

    print("\n🧠 RELEVANT MEMORIES:\n", memory_block if memory_block else "None")

    # ---- Planning with memory ----
    plan = planner.generate_content(f"""
                                    GOAL:
                                    {goal}

                                    MEMORY:
                                    {memory_block}
                                    """).text

    print("\n📝 PLAN:\n", plan)

    steps = [line for line in plan.split("\n") if line.strip().startswith(tuple("123456789"))]
    outputs = []

    for step in steps:
        print(f"\n➡️ STEP: {step}")

        state = {
            "goal": goal,
            "step": step,
            "observations": [],
            "attempts": 0
        }

        while state["attempts"] < MAX_RETRIES:
            state["attempts"] += 1

            response = executor.generate_content(f"""
                                                GOAL:
                                                {state['goal']}

                                                STEP:
                                                {state['step']}

                                                OBSERVATIONS:
                                                {state['observations']}
                                                """).text.strip()

            print(f"\nExecutor Attempt {state['attempts']}:\n{response}")

            if response.startswith("FINAL ANSWER"):
                answer = response.replace("FINAL ANSWER:", "").strip()

                critique = critic.generate_content(f"""
                                                    GOAL:
                                                    {goal}

                                                    STEP:
                                                    {state['step']}

                                                    ANSWER:
                                                    {answer}
                                                    """).text.strip()

                print("\nCritic says:", critique)

                if critique == "PASS":
                    outputs.append(answer)

                    # Store useful knowledge back into memory
                    store_memory(answer, topic="learned_answer")
                    break
                else:
                    state["observations"].append(critique)

        else:
            outputs.append("❌ Failed after retries")

    print ("\n🧠 Storing final output in memory for future retrieval.")
    
    print("\n📚 Current Memory Store:")
    for memory in memory_store:
        if memory["topic"] == "learned_answer":
            print("Stored memory:", memory["content"])

    return "\n\n".join(outputs)

# ================= RUN =================
if __name__ == "__main__":
    result = run_agent(
        "List the top 3 benefits of using vector databases in AI systems"
    )

    print("\n✅ FINAL OUTPUT:\n", result)