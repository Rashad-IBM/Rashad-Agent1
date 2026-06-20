import ast
import asyncio
import json
import operator
import os
import re
import uuid

import httpx
from dotenv import load_dotenv
from fastmcp.client import Client
from fastmcp.client.transports import StreamableHttpTransport
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

load_dotenv()

# ── Identity ──────────────────────────────────────────────
AGENT_NAME   = "RashadAgent-v1"
AGENT_STACK  = "Python / ADK / Gemini"
MODEL        = "gemini-2.5-pro"
LINKEDIN_URL = "www.linkedin.com/in/rashad-muhammed-6373aa69"
GITHUB_URL   = "https://github.com/Rashad-IBM/Rashad-Agent1"

# ── Arena ─────────────────────────────────────────────────
MCP_ENDPOINT = os.environ.get(
    "ARENA_ENDPOINT",
    "https://agent-arena-623774504237.asia-southeast1.run.app/mcp",
)
ID_TOKEN     = os.environ.get("ID_TOKEN", "")
MAX_TURNS    = 20
APP_NAME     = "agent-arena"

# ── API keys ──────────────────────────────────────────────
GEMINI_API_KEY    = os.environ["GEMINI_API_KEY"]


# ─────────────────────────────────────────────────────────
# RunState
# ─────────────────────────────────────────────────────────

class RunState:
    """Shared mutable state — passed into every tool via closure."""

    def __init__(self) -> None:
        self.run_id          = str(uuid.uuid4())
        self.agent_id        = ""
        self.task_id         = ""
        self.current_level   = 1
        self.total_score     = 0
        self.tasks_attempted = 0
        self.level_history: list[dict] = []

    def record(self, level: int, title: str, score: int, levelled_up: bool) -> None:
        self.tasks_attempted += 1
        self.total_score     += score
        if levelled_up:
            self.current_level = level + 1
        self.level_history.append(
            {"level": level, "task": title, "score": score, "up": levelled_up}
        )
        icon = "✓" if levelled_up else ("~" if score >= 70 else "✗")
        print(f"  {icon} L{level}  score={score}/100")


# ─────────────────────────────────────────────────────────
# MCP caller
# ─────────────────────────────────────────────────────────

async def mcp_call(tool: str, args: dict, state: RunState) -> str:
    """Open a fresh MCP session, call one tool, return text result."""
    transport = StreamableHttpTransport(url=MCP_ENDPOINT)
    async with Client(transport, name="arena-agent") as c:
        result = await c.call_tool(tool, args)
    return "\n".join(
        getattr(b, "text", "")
        for b in result.content
        if getattr(b, "text", None)
    )

# ─────────────────────────────────────────────────────────
# Arena tools
# ─────────────────────────────────────────────────────────

def make_arena_tools(state: RunState) -> list:
    """Returns the four Arena tool functions with state captured via closure."""

    async def register_agent(name: str, stack: str) -> str:
        """Register this agent with the Arena. Call once at the start of every run.
        Returns AGENT_ID and current level. Safe to call again — will not duplicate."""
        result = await mcp_call(
            "register_agent",
            {"idToken": ID_TOKEN, "name": name, "stack": stack},
            state,
        )
        m = re.search(r"AGENT_ID:\s*(\S+)", result)
        if m:
            state.agent_id = m.group(1)
        return result

    async def get_tasks(agent_id: str) -> str:
        """Fetch the current assigned task for this agent.
        Returns JSON with: id, title, description, level, points.
        The same task is returned until you skip or submit it."""
        result = await mcp_call(
            "get_tasks",
            {"idToken": ID_TOKEN, "agentId": agent_id},
            state,
        )
        try:
            data = json.loads(result)
            if "id" in data:
                state.task_id = data["id"]
        except Exception:
            pass
        return result

    async def submit_task(agent_id: str, task_id: str, content: str) -> str:
        """Submit your answer for AI evaluation. Scored 0-100.
        Score >= 70 triggers LEVEL_UP. Each task can only be submitted once."""
        result = await mcp_call(
            "submit_task",
            {
                "idToken": ID_TOKEN,
                "agentId": agent_id,
                "taskId": task_id,
                "content": content,
                "metadata": {
                    "agent_name": AGENT_NAME,
                    "model": MODEL,
                    "linkedin": LINKEDIN_URL,
                    "github": GITHUB_URL,
                },
            },
            state,
        )
        return result

    async def skip_task(agent_id: str, task_id: str) -> str:
        """Skip the current task without penalty. Call when a task is already
        submitted or cannot be solved. Unlocks a fresh task from get_tasks."""
        return await mcp_call(
            "skip_task",
            {"idToken": ID_TOKEN, "agentId": agent_id, "taskId": task_id},
            state,
        )

    return [register_agent, get_tasks, submit_task, skip_task]

# ─────────────────────────────────────────────────────────
# Helper tools
# ─────────────────────────────────────────────────────────

async def web_search(query: str) -> str:
    """Search the internet for current facts. Use for any factual, knowledge-based,
    or research task. Returns a summary of the top results."""
    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            data = resp.json()
    except Exception as exc:
        return f"Search failed: {exc}"

    results: list[str] = []
    if data.get("AbstractText"):
        results.append(data["AbstractText"])
    for topic in data.get("RelatedTopics", [])[:5]:
        if isinstance(topic, dict) and topic.get("Text"):
            results.append(topic["Text"])
    for item in data.get("Results", [])[:3]:
        if isinstance(item, dict) and item.get("Text"):
            results.append(item["Text"])
    return "\n".join(results) if results else "No results found for that query."


_SAFE_OPS: dict = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.Pow:      operator.pow,
    ast.Mod:      operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub:     operator.neg,
}


def _eval_node(node: ast.expr) -> float:
    """Recursively evaluate a safe subset of AST nodes (no eval())."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Non-numeric constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


async def calculate(expression: str) -> str:
    """Evaluate a numeric math expression safely. Supports +, -, *, /, **, %, //.
    Use this for any task requiring precise arithmetic or calculation."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree.body)
        formatted: int | float = int(result) if result == int(result) else result
        return f"Result: {formatted}"
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


# ─────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are an autonomous AI agent competing in Agent Arena.
Your goal: earn the highest possible score by completing tasks accurately and thoroughly.

## Sequence to follow on every turn

1. **Register** (first turn only): call register_agent(name="{AGENT_NAME}", stack="{AGENT_STACK}")
   and note the AGENT_ID returned.

2. **Fetch task**: call get_tasks(agent_id=<AGENT_ID>).
   Parse the JSON — note the id, title, description, level, and points.

3. **Research & reason**:
   - For factual, historical, scientific, or knowledge questions → use web_search.
   - For any arithmetic, statistics, or numeric calculation → use calculate.
   - Think step by step. Write a comprehensive, well-structured answer.
   - Aim for completeness: include definitions, examples, and context where relevant.

4. **Submit**: call submit_task(agent_id=<AGENT_ID>, task_id=<task id>, content=<your answer>).
   A score of 0–100 will be returned. Score ≥ 70 triggers LEVEL_UP.

5. **Report**: state the score received and whether you levelled up.

## Edge cases
- If submit_task returns an "already submitted" or "task not found" error → call skip_task,
  then call get_tasks again to get a fresh task and solve it.
- If get_tasks returns no task, wait and retry once.

## Quality tips
- Be thorough. Longer, more detailed answers score higher.
- Cross-check facts with web_search before submitting.
- For math tasks, always use calculate — never estimate.
"""


# ─────────────────────────────────────────────────────────
# Agent builder
# ─────────────────────────────────────────────────────────

def build_agent(state: RunState) -> LlmAgent:
    tools = make_arena_tools(state) + [web_search, calculate]
    return LlmAgent(
        model=MODEL,
        name="arena_agent",
        instruction=SYSTEM_PROMPT,
        tools=tools,
    )


# ─────────────────────────────────────────────────────────
# Runner helpers
# ─────────────────────────────────────────────────────────

async def run_turn(
    runner: Runner,
    user_id: str,
    session_id: str,
    message: str,
) -> str:
    """Send one user message; collect and return the agent's final text reply."""
    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=message)],
    )
    reply = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                reply = event.content.parts[0].text or ""
            break
    return reply


# ─────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────

async def main() -> None:
    if not ID_TOKEN:
        print("⚠  ID_TOKEN is not set. Export ID_TOKEN=<your Firebase JWT> and retry.")
        return

    state    = RunState()
    agent    = build_agent(state)
    sessions = InMemorySessionService()
    user_id  = state.run_id

    await sessions.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=state.run_id,
    )

    runner = Runner(app_name=APP_NAME, agent=agent, session_service=sessions)

    print(f"▶  Run {state.run_id[:8]}  agent={AGENT_NAME}  model={MODEL}")
    print(f"   MAX_TURNS={MAX_TURNS}\n")

    # Turn 1 — register + first task
    reply = await run_turn(
        runner, user_id, state.run_id,
        f"Start now. Register as '{AGENT_NAME}' with stack '{AGENT_STACK}', "
        "then get your first task, solve it thoroughly using all available tools, "
        "and submit it.",
    )
    print(f"Agent: {reply}\n")

    # Turns 2 … MAX_TURNS — one task per turn
    for turn_num in range(2, MAX_TURNS + 1):
        print(f"─── Turn {turn_num} " + "─" * 40)
        reply = await run_turn(
            runner, user_id, state.run_id,
            f"Continue. Get the next task (attempt #{turn_num}), "
            "solve it thoroughly using web_search and calculate as needed, "
            "and submit it.",
        )
        print(f"Agent: {reply}\n")

    # Final summary
    avg = state.total_score / state.tasks_attempted if state.tasks_attempted else 0.0
    print(
        f"\n{'=' * 50}\n"
        f"DONE  level={state.current_level}  tasks={state.tasks_attempted}"
        f"  total={state.total_score}  avg={avg:.1f}\n"
        f"{'=' * 50}"
    )


if __name__ == "__main__":
    asyncio.run(main())
