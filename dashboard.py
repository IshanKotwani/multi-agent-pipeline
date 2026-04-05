import streamlit as st
import pandas as pd
from orchestrator import run_pipeline, run_full_eval
from observability import get_all_logs, init_db
from evaluator import get_eval_results, get_version_comparison
from debugger import get_debug_logs, get_failed_calls, get_recent_task_ids

st.set_page_config(page_title="Multi-Agent Pipeline", layout="wide")
st.title("🤖 Multi-Agent Research Pipeline")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pipeline Controls")
    max_budget = st.slider(
        "Max budget per run (USD)",
        min_value=0.01,
        max_value=0.50,
        value=0.10,
        step=0.01,
        help="Pipeline will stop if cumulative cost exceeds this limit"
    )
    st.caption(f"Current limit: ${max_budget:.2f}")
    st.divider()
    st.markdown("**Pipeline Stages**")
    st.markdown("0. 🧭 LLM Router")
    st.markdown("1. 🧠 Memory Retrieval")
    st.markdown("2. 🔍 Research Agent")
    st.markdown("3. 📝 Summarisation Agent")
    st.markdown("4. ✅ Validation Agent")
    st.markdown("5. 📊 Report Writer Agent")
    st.divider()
    st.markdown("**Model Routing**")
    st.markdown("💨 Haiku → simple topics")
    st.markdown("⚡ Sonnet → complex topics")

tab1, tab2, tab3, tab4 = st.tabs(["Pipeline", "Observability", "Evals", "Debug Console"])

# ── Tab 1: Pipeline ──────────────────────────────────────────────────────────
with tab1:
    topic = st.text_input("Enter a research topic", placeholder="e.g. Natural gas futures markets")

    if st.button("Run Pipeline", type="primary") and topic:
        status_box = st.empty()
        result = run_pipeline(
            topic,
            max_budget_usd=max_budget,
            progress_callback=lambda msg: status_box.info(msg)
        )
        status_box.empty()

        if result["budget_exceeded"]:
            st.warning(f"⚠️ {result['error']}")
        elif result["error"]:
            st.error(f"Pipeline failed: {result['error']}")
        else:
            st.success(f"Pipeline complete! Total cost: ${result['total_cost']:.4f} | Task ID: `{result['task_id']}`")

            model_label = "⚡ Sonnet (Complex)" if "sonnet" in result["model"] else "💨 Haiku (Simple)"
            st.info(f"**Router:** {model_label} | Score: {result['complexity_score']}/10 | {result['routing_reason']}")

            if result["prior_context"]:
                with st.expander(f"🧠 Memory: {len(result['prior_context'])} similar past research found"):
                    for item in result["prior_context"]:
                        st.markdown(f"**{item['topic']}** — {item['similarity']*100:.0f}% similar")
                        st.caption(item["summary"])
            else:
                st.caption("🧠 Memory: No similar past research found — this is a new topic.")

            if result["eval_scores"]:
                scores = result["eval_scores"]
                st.divider()
                st.subheader("🧪 Eval Scores")
                c1, c2, c3 = st.columns(3)
                c1.metric("Keyword Coverage", f"{scores['keyword_score']*100:.0f}%",
                          f"{scores['keywords_hit']}/{scores['keywords_total']} keywords")
                c2.metric("Length Score", f"{scores['length_score']*100:.0f}%",
                          f"{scores['word_count']} words")
                c3.metric("Overall Score", f"{scores['overall_score']*100:.0f}%")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("🔍 Research")
                st.write(result["research_output"])
            with col2:
                st.subheader("📝 Summary")
                st.write(result["summary_output"])
            with col3:
                st.subheader("✅ Validation")
                verdict = "🟢 PASS" if result["validation_passed"] else "🔴 FAIL"
                st.markdown(f"**{verdict}**")
                st.write(result["validation_output"])

            st.divider()
            st.subheader("📊 Executive Report")
            st.markdown(result["report_output"])

# ── Tab 2: Observability ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Agent Call Logs")
    init_db()
    logs = get_all_logs()

    if not logs:
        st.info("No logs yet. Run the pipeline first.")
    else:
        df = pd.DataFrame(logs)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Calls", len(df))
        col2.metric("Avg Latency", f"{df['latency_ms'].mean():.0f} ms")
        col3.metric("Total Cost", f"${df['cost_usd'].sum():.4f}")
        val_df = df[df["agent_name"] == "validate"]
        if len(val_df) > 0:
            pass_rate = (val_df["passed"] == 1).sum() / len(val_df) * 100
        else:
            pass_rate = 0
        col4.metric("Validation Pass Rate", f"{pass_rate:.0f}%")

        st.divider()

        st.subheader("Model Distribution")
        model_counts = df[df["model"].notna()].groupby("model").size().reset_index(name="calls")
        st.bar_chart(model_counts.set_index("model"))

        st.subheader("Per-Agent Latency")
        latency = df.groupby("agent_name")["latency_ms"].mean().reset_index()
        st.bar_chart(latency.set_index("agent_name"))

        st.subheader("Full Audit Log")
        st.dataframe(df[["timestamp", "task_id", "agent_name", "status", "passed", "model",
                          "latency_ms", "input_tokens", "output_tokens", "cost_usd"]],
                     use_container_width=True)

# ── Tab 3: Evals ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🧪 Eval Framework")
    st.caption("Run the pipeline against a fixed eval set to track quality across prompt versions.")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("▶ Run Full Eval Suite", type="primary"):
            status = st.empty()
            results = run_full_eval(
                progress_callback=lambda msg: status.info(msg)
            )
            status.empty()
            st.success(f"Eval complete — {len(results)} cases scored.")

    st.divider()

    versions = get_version_comparison()
    if versions:
        st.subheader("📈 Prompt Version Comparison")
        vdf = pd.DataFrame(versions)
        vdf["avg_score"] = vdf["avg_score"].apply(lambda x: f"{x*100:.1f}%")
        vdf["avg_keyword"] = vdf["avg_keyword"].apply(lambda x: f"{x*100:.1f}%")
        vdf["pass_rate"] = vdf["pass_rate"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(vdf[["prompt_version", "runs", "avg_score", "avg_keyword", "pass_rate"]],
                     use_container_width=True)

    eval_results = get_eval_results()
    if eval_results:
        st.subheader("📋 Individual Eval Results")
        edf = pd.DataFrame(eval_results)
        edf["overall_score"] = edf["overall_score"].apply(lambda x: f"{x*100:.1f}%")
        edf["keyword_score"] = edf["keyword_score"].apply(lambda x: f"{x*100:.1f}%")
        edf["validation_passed"] = edf["validation_passed"].apply(lambda x: "✅" if x else "❌")
        st.dataframe(edf[["prompt_version", "eval_id", "topic", "overall_score",
                           "keyword_score", "validation_passed", "timestamp"]],
                     use_container_width=True)
    else:
        st.info("No eval results yet. Click 'Run Full Eval Suite' to start.")

# ── Tab 4: Debug Console ─────────────────────────────────────────────────────
with tab4:
    st.subheader("🐛 Debug Console")

    debug_tab1, debug_tab2 = st.tabs(["Failed Calls", "Task Inspector"])

    with debug_tab1:
        st.subheader("❌ Failed & Retried Calls")
        failed = get_failed_calls()
        if not failed:
            st.success("No failed calls — all agents are running cleanly.")
        else:
            for f in failed:
                with st.expander(f"❌ {f['agent_name']} | attempt {f['attempt']} | {f['timestamp'][:19]}"):
                    st.markdown(f"**Task ID:** `{f['task_id']}`")
                    st.markdown(f"**Model:** {f['model']}")
                    st.markdown(f"**Error:**")
                    st.error(f['error'])
                    st.markdown("**Input (truncated):**")
                    st.code(f['prompt_input'][:500], language="text")

    with debug_tab2:
        st.subheader("🔍 Inspect Task by ID")
        task_ids = get_recent_task_ids()

        if not task_ids:
            st.info("No tasks logged yet. Run the pipeline first.")
        else:
            selected_task = st.selectbox(
                "Select a task to inspect",
                options=task_ids,
                format_func=lambda x: x[:16] + "..."
            )

            if selected_task:
                logs = get_debug_logs(selected_task)
                st.markdown(f"**{len(logs)} agent calls** for task `{selected_task[:16]}...`")

                for log in logs:
                    status_icon = "✅" if not log["error"] else "❌"
                    with st.expander(f"{status_icon} {log['agent_name']} | attempt {log['attempt']} | {log['latency_ms']}ms"):
                        st.markdown(f"**Model:** {log['model']}")
                        st.markdown(f"**Latency:** {log['latency_ms']}ms")

                        if log["error"]:
                            st.error(f"Error: {log['error']}")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Input (truncated):**")
                            st.code(log["prompt_input"][:600] if log["prompt_input"] else "N/A", language="text")
                        with col2:
                            st.markdown("**Output (truncated):**")
                            st.code(log["raw_output"][:600] if log["raw_output"] else "N/A", language="text")