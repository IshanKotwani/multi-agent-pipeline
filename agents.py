import time
import os
from openai import OpenAI
from dotenv import load_dotenv
from debugger import log_debug

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.aicredits.in/v1",
)

HAIKU  = "anthropic/claude-haiku-4-5"
SONNET = "anthropic/claude-sonnet-4-5"

# ── LLM Router ────────────────────────────────────────────────────────────────

def route_model(topic: str) -> tuple[str, int, str]:
    response = client.chat.completions.create(
        model=HAIKU,
        max_tokens=100,
        messages=[
            {"role": "system", "content": """You are a complexity classifier. Given a research topic, rate its complexity from 1-10:
1-5 = Simple (well-known topic, straightforward facts, limited domain knowledge needed)
6-10 = Complex (technical, multi-domain, nuanced analysis required, specialised knowledge needed)
Respond in exactly this format:
SCORE: <number>
REASON: <one sentence>"""},
            {"role": "user", "content": f"Topic: {topic}"}
        ]
    )
    output = response.content[0]['text']
    lines = output.strip().split('\n')
    score = 5
    reason = "Default routing"
    for line in lines:
        if line.startswith("SCORE:"):
            try:
                score = int(line.replace("SCORE:", "").strip())
            except:
                score = 5
        if line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()
    model = SONNET if score > 5 else HAIKU
    return model, score, reason


# ── Agents ────────────────────────────────────────────────────────────────────

def research_agent(topic: str, model: str, prior_context: list[dict] = None, task_id: str = "", attempt: int = 0) -> dict:
    start = time.time()
    memory_block = ""
    if prior_context:
        memory_block = "\n\n## Relevant Prior Research (from memory)\n"
        for item in prior_context:
            memory_block += f"\n**Similar topic ({item['similarity']*100:.0f}% match): {item['topic']}**\n"
            memory_block += f"Summary: {item['summary']}\n"
        memory_block += "\nUse this as background context where relevant, but focus on the new topic.\n"

    prompt = f"Research this topic: {topic}{memory_block}"
    system = """You are a research agent. Given a topic, produce a structured research document with exactly these sections:
## Background
## Key Concepts
## Market Dynamics
## Risks & Considerations
## Conclusion
Be factual, thorough, and concise within each section."""

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
        )
        output = response.content[0]['text']
        latency = round((time.time() - start) * 1000)
        log_debug(task_id, "research", prompt, raw_output=output, attempt=attempt, latency_ms=latency, model=model)
        return {
            "output": output,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency_ms": latency,
            "model": model
        }
    except Exception as e:
        latency = round((time.time() - start) * 1000)
        log_debug(task_id, "research", prompt, error=str(e), attempt=attempt, latency_ms=latency, model=model)
        raise

def summarise_agent(research_text: str, model: str, task_id: str = "", attempt: int = 0) -> dict:
    start = time.time()
    prompt = f"Summarise this:\n\n{research_text}"
    system = """You are a summarisation agent. Given a structured research document, extract exactly 5 bullet points — one key insight per section.
Be accurate and faithful to the source material. Format as a simple bullet list."""

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=500,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
        )
        output = response.content[0]['text']
        latency = round((time.time() - start) * 1000)
        log_debug(task_id, "summarise", prompt, raw_output=output, attempt=attempt, latency_ms=latency, model=model)
        return {
            "output": output,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency_ms": latency,
            "model": model
        }
    except Exception as e:
        latency = round((time.time() - start) * 1000)
        log_debug(task_id, "summarise", prompt, error=str(e), attempt=attempt, latency_ms=latency, model=model)
        raise

def validate_agent(research_text: str, summary_text: str, model: str, task_id: str = "", attempt: int = 0) -> dict:
    start = time.time()
    prompt = f"""Original research:\n{research_text}\n\nSummary to validate:\n{summary_text}\n\nDoes the summary accurately reflect the research?"""
    system = """You are a validation agent. Compare a summary against the original research.
PASS if the summary captures the main points accurately with no factual errors — minor omissions are acceptable.
FAIL only if the summary contains clear factual errors or hallucinations not present in the research.
Briefly note any issues, then end your response with either PASS or FAIL on its own line."""

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=500,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
        )
        output = response.content[0]['text']
        passed = output.strip().endswith("PASS")
        latency = round((time.time() - start) * 1000)
        log_debug(task_id, "validate", prompt, raw_output=output, attempt=attempt, latency_ms=latency, model=model)
        return {
            "output": output,
            "passed": passed,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency_ms": latency,
            "model": model
        }
    except Exception as e:
        latency = round((time.time() - start) * 1000)
        log_debug(task_id, "validate", prompt, error=str(e), attempt=attempt, latency_ms=latency, model=model)
        raise

def report_writer_agent(topic: str, research_text: str, summary_text: str, model: str, task_id: str = "", attempt: int = 0) -> dict:
    start = time.time()
    prompt = f"Topic: {topic}\n\nResearch:\n{research_text}\n\nValidated Summary:\n{summary_text}\n\nWrite the executive report."
    system = """You are a professional report writing agent. Given a research document and a validated summary, produce a clean, well-formatted executive report in markdown.
The report must include:
# Executive Report: [Topic]
## Overview (2-3 sentences)
## Key Findings (from the summary bullets)
## Detailed Analysis (from the research)
## Conclusion & Recommendations
Make it professional and concise — suitable for a senior stakeholder."""

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
        )
        output = response.content[0]['text']
        latency = round((time.time() - start) * 1000)
        log_debug(task_id, "report_writer", prompt, raw_output=output, attempt=attempt, latency_ms=latency, model=model)
        return {
            "output": output,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency_ms": latency,
            "model": model
        }
    except Exception as e:
        latency = round((time.time() - start) * 1000)
        log_debug(task_id, "report_writer", prompt, error=str(e), attempt=attempt, latency_ms=latency, model=model)
        raise