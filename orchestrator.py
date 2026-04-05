import uuid
from agents import route_model, research_agent, summarise_agent, validate_agent, report_writer_agent
from observability import init_db, log_agent_call
from memory import retrieve_similar, store_research
from evaluator import EVAL_CASES, log_eval_result

MAX_RETRIES = 2
PROMPT_VERSION = "v1.0"

PRICING = {
    "anthropic/claude-haiku-4-5":  {"input": 0.80,  "output": 4.00},
    "anthropic/claude-sonnet-4-5": {"input": 3.00,  "output": 15.00},
}

def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    pricing = PRICING.get(model, PRICING["anthropic/claude-haiku-4-5"])
    return (input_tokens / 1_000_000 * pricing["input"] +
            output_tokens / 1_000_000 * pricing["output"])

def run_pipeline(topic: str, max_budget_usd: float = 0.10, progress_callback=None) -> dict:
    init_db()
    task_id = str(uuid.uuid4())
    cumulative_cost = 0.0

    context = {
        "task_id": task_id,
        "topic": topic,
        "stage": "ROUTING",
        "model": None,
        "complexity_score": None,
        "routing_reason": None,
        "prior_context": [],
        "research_output": None,
        "summary_output": None,
        "validation_output": None,
        "validation_passed": False,
        "report_output": None,
        "total_cost": 0.0,
        "error": None,
        "budget_exceeded": False,
        "eval_scores": None
    }

    # --- Router ---
    if progress_callback:
        progress_callback("🧭 Router classifying topic complexity...")
    try:
        model, score, reason = route_model(topic)
        context["model"] = model
        context["complexity_score"] = score
        context["routing_reason"] = reason
    except Exception as e:
        model = "anthropic/claude-haiku-4-5"
        context["model"] = model
        context["complexity_score"] = 5
        context["routing_reason"] = f"Fallback to Haiku (router error: {e})"

    # --- Memory Retrieval ---
    if progress_callback:
        progress_callback("🧠 Searching memory for similar past research...")
    try:
        prior_context = retrieve_similar(topic, n_results=2)
        context["prior_context"] = prior_context
    except Exception:
        prior_context = []
        context["prior_context"] = []

    # --- Stage 1: Research ---
    if progress_callback:
        memory_note = f" (+ {len(prior_context)} memory hits)" if prior_context else ""
        progress_callback(f"🔍 Research Agent running on {'Sonnet ⚡' if 'sonnet' in model else 'Haiku 💨'}{memory_note}...")

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = research_agent(topic, model, prior_context=prior_context, task_id=task_id, attempt=attempt)
            cost = estimate_cost(result["input_tokens"], result["output_tokens"], model)
            cumulative_cost += cost
            if cumulative_cost > max_budget_usd:
                context["budget_exceeded"] = True
                context["error"] = f"Budget limit of ${max_budget_usd} exceeded after Research agent."
                return context
            log_agent_call(task_id, "research", result)
            context["research_output"] = result["output"]
            context["stage"] = "SUMMARISE"
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                context["error"] = f"Research agent failed: {e}"
                return context
            log_agent_call(task_id, "research", {"output": "", "input_tokens": 0, "output_tokens": 0, "latency_ms": 0, "model": model}, status="retry")

    # --- Stage 2: Summarise ---
    if progress_callback:
        progress_callback("📝 Summarisation Agent running...")

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = summarise_agent(context["research_output"], model, task_id=task_id, attempt=attempt)
            cost = estimate_cost(result["input_tokens"], result["output_tokens"], model)
            cumulative_cost += cost
            if cumulative_cost > max_budget_usd:
                context["budget_exceeded"] = True
                context["error"] = f"Budget limit of ${max_budget_usd} exceeded after Summarise agent."
                return context
            log_agent_call(task_id, "summarise", result)
            context["summary_output"] = result["output"]
            context["stage"] = "VALIDATE"
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                context["error"] = f"Summarise agent failed: {e}"
                return context

    # --- Stage 3: Validate ---
    if progress_callback:
        progress_callback("✅ Validation Agent running...")

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = validate_agent(context["research_output"], context["summary_output"], model, task_id=task_id, attempt=attempt)
            cost = estimate_cost(result["input_tokens"], result["output_tokens"], model)
            cumulative_cost += cost
            if cumulative_cost > max_budget_usd:
                context["budget_exceeded"] = True
                context["error"] = f"Budget limit of ${max_budget_usd} exceeded after Validate agent."
                return context
            log_agent_call(task_id, "validate", result, passed=1 if result["passed"] else 0)
            context["validation_output"] = result["output"]
            context["validation_passed"] = result["passed"]
            context["stage"] = "REPORT"
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                context["error"] = f"Validation agent failed: {e}"
                return context

    # --- Stage 4: Report Writer ---
    if progress_callback:
        progress_callback("📊 Report Writer Agent generating executive report...")

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = report_writer_agent(topic, context["research_output"], context["summary_output"], model, task_id=task_id, attempt=attempt)
            cost = estimate_cost(result["input_tokens"], result["output_tokens"], model)
            cumulative_cost += cost
            if cumulative_cost > max_budget_usd:
                context["budget_exceeded"] = True
                context["error"] = f"Budget limit of ${max_budget_usd} exceeded after Report Writer agent."
                return context
            log_agent_call(task_id, "report_writer", result)
            context["report_output"] = result["output"]
            context["total_cost"] = cumulative_cost
            context["stage"] = "DONE"
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                context["error"] = f"Report Writer agent failed: {e}"
                return context

    # --- Store in Memory ---
    if progress_callback:
        progress_callback("💾 Storing research in memory...")
    try:
        store_research(topic, context["research_output"], context["summary_output"])
    except Exception:
        pass

    # --- Eval Scoring ---
    matching_eval = next((e for e in EVAL_CASES if e["topic"].lower() in topic.lower()
                         or topic.lower() in e["topic"].lower()), None)
    if matching_eval and context["research_output"]:
        scores = log_eval_result(
            PROMPT_VERSION,
            matching_eval,
            context["research_output"],
            context["validation_passed"]
        )
        context["eval_scores"] = scores

    return context


def run_full_eval(progress_callback=None) -> list[dict]:
    results = []
    for case in EVAL_CASES:
        if progress_callback:
            progress_callback(f"🧪 Evaluating: {case['topic']}...")
        result = run_pipeline(case["topic"], max_budget_usd=0.20)
        if result["research_output"]:
            scores = log_eval_result(
                PROMPT_VERSION,
                case,
                result["research_output"],
                result["validation_passed"]
            )
            results.append({
                "eval_id": case["id"],
                "topic": case["topic"],
                **scores,
                "validation_passed": result["validation_passed"]
            })
    return results