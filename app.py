"""PromptForge AI • Prompt Engineering Studio (Streamlit entry point)."""

from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv

from promptforge.engine import (
    MAX_TASK_CHARS,
    TECHNIQUES,
    PromptConfig,
    build_messages,
    make_optimization_messages,
    messages_to_text,
)
from promptforge.evaluation import MAX_EVAL_ROWS, check_keywords, load_cases
from promptforge.llm import DEFAULT_MODEL, MODEL_OPTIONS, ModelCallError, generate

load_dotenv()
st.set_page_config(
    page_title="PromptForge AI | Prompt Engineering Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {color-scheme: dark;}
.block-container {padding-top: 1.3rem; max-width: 1300px;}
[data-testid="stApp"] {background: radial-gradient(ellipse at 87% 2%, #14263e 0, #090f1d 42%, #090d18 78%);}
h1,h2,h3 {letter-spacing: -.035em;}
[data-testid="stSidebar"] {background:#0e1728;}
.hero {border:1px solid #334966; border-radius:20px; padding:24px 29px;
       background:linear-gradient(115deg,#182b4b,#171d38 60%,#162943); margin-bottom:20px;}
.kicker {text-transform:uppercase; color:#91aefa; letter-spacing:.16em; font-size:12px; font-weight:750;}
.hero h1 {color:#fafaff; font-size:2.65rem; margin:7px 0;}
.hero p {color:#cbd5e1; margin-bottom:0;}
.tiny {font-size:.85rem;color:#9daec6;}
[data-testid="stMetric"] {background:#151e31; border:1px solid #283852; border-radius:14px; padding:15px;}
div.stButton > button[kind="primary"] {background:#7b75ff; border:0;color:white;}
div[data-testid="stTabs"] button {font-weight:600;}
</style>
""", unsafe_allow_html=True)


# ------------------------------ session & secrets ------------------------------
def init_state() -> None:
    defaults = {"history": [], "api_calls": 0, "lab_output": None,
                "compare_output": None, "opt_output": None, "eval_output": None}
    for name, default in defaults.items():
        if name not in st.session_state:
            st.session_state[name] = default


def api_key() -> str:
    entered = st.session_state.get("sidebar_api_key", "").strip()
    if entered:
        return entered
    env = os.getenv("GROQ_API_KEY", "").strip()
    if env:
        return env
    try:
        return str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except Exception:
        # Streamlit throws a custom secrets-not-found exception with no secrets file.
        return ""


def remember(label: str, task: str, result) -> None:
    st.session_state.history.insert(0, {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "technique": label,
        "input": task,
        "output": result.text,
        "model": result.model,
        "latency_seconds": result.latency_seconds,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
    })
    st.session_state.history = st.session_state.history[:50]


def run_model(messages, *, json_mode=False):
    if not api_key():
        raise ModelCallError("Add a Groq API key in the sidebar or a secrets file first.")
    if st.session_state.api_calls >= 30:
        raise ModelCallError("This browser session reached its 30-request safety cap. Start a new session later.")
    # Count attempted requests as well as successes, to avoid accidental retry loops.
    st.session_state.api_calls += 1
    return generate(
        api_key=api_key(), messages=messages,
        model=model, temperature=float(temperature),
        max_tokens=int(max_tokens), json_mode=json_mode,
    )


def show_result(result, *, key: str) -> None:
    st.markdown("#### Model response")
    st.markdown(result.text)
    c1, c2, c3 = st.columns(3)
    c1.metric("Latency", f"{result.latency_seconds:.2f}s")
    c2.metric("Input tokens", str(result.prompt_tokens or "N/A"))
    c3.metric("Output tokens", str(result.completion_tokens or "N/A"))
    st.download_button("⬇ Download response", result.text,
                       file_name=f"promptforge_{key}.txt", mime="text/plain", key=f"download_{key}")


init_state()

with st.sidebar:
    st.markdown("## 🧠 PromptForge AI")
    st.caption("LLM CONTROL CENTER")
    st.divider()
    st.text_input("Groq API key (optional if configured)",
                  type="password", key="sidebar_api_key",
                  help="Session-only entry. Never paste keys into source code or GitHub.")
    if api_key():
        st.success("API key available", icon="✅")
    else:
        st.info("Preview mode. Add a key to generate AI outputs.", icon="🔐")
    model = st.selectbox("Model", options=MODEL_OPTIONS, index=MODEL_OPTIONS.index(DEFAULT_MODEL))
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0,
                            value=0.3, step=0.1,
                            help="Lower is typically more consistent; higher is more varied.")
    max_tokens = st.slider("Max output tokens", 128, 1600, 704, step=64)
    st.divider()
    st.caption(f"Session API attempts: {st.session_state.api_calls}/30")
    st.progress(min(st.session_state.api_calls / 30, 1.0))
    st.caption("Responses stay in this browser session. No database or server-side response log is used by this project. Groq processes any prompts submitted to its API.")
    st.link_button("Get a Groq API key ↗", "https://console.groq.com/keys", use_container_width=True)

st.markdown("""
<div class="hero"><div class="kicker">Applied Generative AI • Portfolio Lab</div>
<h1>✦ PromptForge AI</h1>
<p>Design better prompts. Run real models. Compare strategies. Evaluate measurable outcomes.</p>
</div>
""", unsafe_allow_html=True)

tab_lab, tab_compare, tab_opt, tab_eval, tab_history, tab_learn = st.tabs([
    "🧪 Prompt Lab", "⚖️ A/B Compare", "✨ Optimizer", "📊 Evaluation", "🕘 History", "📚 Learn"
])

with tab_lab:
    st.subheader("Prompt Lab")
    st.caption("Create a structured chat prompt, inspect its exact messages, and call the LLM.")
    left, right = st.columns([1.13, 1], gap="large")
    with left:
        technique = st.selectbox("Prompting technique", TECHNIQUES, key="lab_technique")
        task = st.text_area("Your task", value="Explain RAG in simple terms and give a real-life example.",
                            max_chars=MAX_TASK_CHARS, height=110, key="lab_task")
        with st.expander("Advanced settings (role, context, examples, constraints)", expanded=False):
            role = st.text_input("Role / persona", "an experienced GenAI engineer")
            audience = st.text_input("Audience", "beginner developers")
            tone = st.selectbox("Tone", ["clear and professional", "friendly", "technical", "concise", "creative"])
            context = st.text_area("Reference context (used especially by Context-grounded)",
                                   placeholder="Paste trusted source material here...", max_chars=12000, height=120)
            examples = st.text_area("Few-shot examples: one `input => output` per line",
                                    placeholder="happy => positive\nsad => negative", height=85)
            constraints = st.text_area("Constraints", placeholder="Max 100 words; use three bullets...", height=75)
        cfg = PromptConfig(technique=technique, task=task, role=role,
                           audience=audience, tone=tone, context=context,
                           examples=examples, constraints=constraints)
        try:
            prompt_messages = build_messages(cfg)
            prompt_error = None
        except ValueError as exc:
            prompt_messages = None
            prompt_error = str(exc)
        if prompt_error:
            st.warning(prompt_error)
        pressed = st.button("🚀 Generate with AI", type="primary", use_container_width=True,
                            key="lab_generate", disabled=bool(prompt_error))
        if pressed and prompt_messages:
            try:
                with st.spinner("Calling Groq..."):
                    result = run_model(prompt_messages, json_mode=(technique == "Structured JSON"))
                remember(technique, task, result)
                st.session_state.lab_output = result
            except ModelCallError as exc:
                st.error(str(exc))
    with right:
        st.markdown("#### Prompt preview")
        st.caption("This is the exact system/user/few-shot structure sent to the API.")
        if prompt_messages:
            st.code(messages_to_text(prompt_messages), language="text", wrap_lines=True)
        else:
            st.info("Fix the prompt settings to see a preview.")
    if st.session_state.lab_output:
        st.divider()
        show_result(st.session_state.lab_output, key="lab")

with tab_compare:
    st.subheader("A/B Prompt Comparison")
    st.caption("Two model calls on the same task, changing only the prompting technique. Compare results manually.")
    compare_task = st.text_area("Comparison task", "Explain what vector databases do using an analogy.",
                                max_chars=MAX_TASK_CHARS, key="compare_task")
    ca, cb = st.columns(2)
    with ca:
        technique_a = st.selectbox("Technique A", TECHNIQUES, index=0, key="technique_a")
    with cb:
        technique_b = st.selectbox("Technique B", TECHNIQUES, index=1, key="technique_b")
    st.caption("For Few-shot use two provided examples. Context-grounded uses the reference text below.")
    compare_context = st.text_area("Optional reference material", "", key="compare_context")
    compare_examples = st.text_area("Few-shot examples (`input => output` per line)",
                                    "cat => an animal\ncar => a vehicle", key="compare_examples")
    if st.button("⚖️ Run A/B comparison", type="primary", key="run_compare"):
        try:
            if technique_a == technique_b:
                raise ValueError("Choose two different techniques to make the comparison meaningful.")
            a_messages = build_messages(PromptConfig(technique_a, compare_task,
                                                      context=compare_context, examples=compare_examples))
            b_messages = build_messages(PromptConfig(technique_b, compare_task,
                                                      context=compare_context, examples=compare_examples))
            with st.spinner("Generating two outputs..."):
                result_a = run_model(a_messages, json_mode=(technique_a == "Structured JSON"))
                remember(technique_a + " (A)", compare_task, result_a)
                result_b = run_model(b_messages, json_mode=(technique_b == "Structured JSON"))
                remember(technique_b + " (B)", compare_task, result_b)
            st.session_state.compare_output = (technique_a, result_a, technique_b, result_b)
        except (ModelCallError, ValueError) as exc:
            st.error(str(exc))
    if st.session_state.compare_output:
        name_a, result_a, name_b, result_b = st.session_state.compare_output
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown(f"#### A · {name_a}")
            st.markdown(result_a.text)
            st.caption(f"{result_a.latency_seconds:.2f}s • {result_a.total_tokens or '?'} total tokens")
        with c2:
            st.markdown(f"#### B · {name_b}")
            st.markdown(result_b.text)
            st.caption(f"{result_b.latency_seconds:.2f}s • {result_b.total_tokens or '?'} total tokens")
        st.info("Lower latency/token usage does not automatically mean higher answer quality.")

with tab_opt:
    st.subheader("AI Prompt Optimizer")
    st.caption("A model rewrites a weak prompt into a more actionable prompt. Review the revision before using it.")
    weak_prompt = st.text_area("Prompt to improve", "Tell me about machine learning.",
                               max_chars=MAX_TASK_CHARS, height=125)
    if st.button("✨ Improve prompt", type="primary"):
        try:
            with st.spinner("Improving your prompt..."):
                improved = run_model(make_optimization_messages(weak_prompt))
            remember("Optimizer", weak_prompt, improved)
            st.session_state.opt_output = improved
        except (ValueError, ModelCallError) as exc:
            st.error(str(exc))
    if st.session_state.opt_output:
        st.markdown("#### Improved prompt")
        st.code(st.session_state.opt_output.text, language="text", wrap_lines=True)
        st.download_button("⬇ Download optimized prompt", st.session_state.opt_output.text,
                           "optimized_prompt.txt", "text/plain")

with tab_eval:
    st.subheader("Keyword-Coverage Evaluation")
    st.caption("Run a small labeled CSV against the model. This is a deterministic keyword check, NOT an objective measure of answer correctness.")
    st.markdown("CSV fields: `input,expected_keywords` (separate multiple expected keywords with semicolons).")
    example_csv = ("input,expected_keywords\n"
                   '"What does RAG stand for?","retrieval;generation"\n'
                   '"What does few-shot prompting use?","examples"\n')
    st.download_button("⬇ Download starter evaluation CSV", example_csv,
                       file_name="evaluation_template.csv", mime="text/csv")
    uploaded = st.file_uploader("Upload evaluation CSV (maximum five rows)", type=["csv"])
    eval_technique = st.selectbox("Technique to evaluate", ["Zero-shot", "Role-based", "Structured JSON", "Constraints-first"])
    if uploaded is not None:
        if uploaded.size > 25_000:
            st.error("CSV is too large. Keep the file below 25 KB.")
        else:
            try:
                cases = load_cases(uploaded.getvalue().decode("utf-8-sig"))
                st.success(f"Loaded {len(cases)} evaluation cases. Each uses one API call.")
                st.dataframe([{"input": c.task, "expected_keywords": ";".join(c.expected_keywords)} for c in cases],
                             use_container_width=True, hide_index=True)
                if st.button("📊 Evaluate", type="primary"):
                    rows = []
                    with st.spinner("Evaluating each example..."):
                        for item in cases:
                            try:
                                response = run_model(build_messages(PromptConfig(eval_technique, item.task)),
                                                     json_mode=(eval_technique == "Structured JSON"))
                                remember(f"Evaluation: {eval_technique}", item.task, response)
                                passed, missing = check_keywords(response.text, item.expected_keywords)
                                rows.append({"input": item.task,
                                             "expected_keywords": ";".join(item.expected_keywords),
                                             "passed": passed, "missing": ";".join(missing),
                                             "response": response.text,
                                             "latency_seconds": response.latency_seconds})
                            except ModelCallError as exc:
                                st.error(f"Stopped after {len(rows)} cases: {exc}")
                                break
                    st.session_state.eval_output = rows
            except (ValueError, UnicodeDecodeError) as exc:
                st.error(f"Could not read the CSV: {exc}")
    if st.session_state.eval_output:
        rows = st.session_state.eval_output
        passed_n = sum(row["passed"] for row in rows)
        st.metric("Keyword coverage pass rate", f"{100 * passed_n / len(rows):.0f}%",
                  help="Pass = every expected keyword appears in the response, not factual accuracy.")
        st.dataframe(rows, use_container_width=True, hide_index=True)
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        st.download_button("⬇ Export evaluation results CSV", csv_buffer.getvalue(),
                           file_name="promptforge_eval_results.csv", mime="text/csv")

with tab_history:
    st.subheader("Session History")
    st.caption("Up to 50 recent successful responses are held in session memory only. Refreshing/restarting may clear them.")
    if st.session_state.history:
        st.metric("Saved outputs", len(st.session_state.history))
        st.download_button("⬇ Export history JSON", json.dumps(st.session_state.history, indent=2),
                           file_name="promptforge_history.json", mime="application/json")
        if st.button("🗑 Clear history"):
            st.session_state.history = []
            st.rerun()
        for i, event in enumerate(st.session_state.history):
            with st.expander(f"{event['time_utc']} · {event['technique']} · {event['model']}"):
                st.markdown("**Input**")
                st.write(event["input"])
                st.markdown("**Response**")
                st.write(event["output"])
                st.caption(f"Latency: {event['latency_seconds']}s | Total tokens: {event['total_tokens']}")
    else:
        st.info("No generations yet. Try Prompt Lab or the Optimizer.")

with tab_learn:
    st.subheader("Prompt Engineering Handbook")
    st.markdown("""
**Zero-shot:** Ask a model to complete a task without examples. Best baseline.

**Role-based:** Provide a specific role or expert perspective through a system message.

**Few-shot:** Add a few input/output examples so the model can imitate the desired format.

**Structured JSON:** Ask for an exact machine-readable shape; use JSON mode on the API.

**Context-grounded:** Supply reference material and instruct the model not to go beyond it. Context is not a security boundary; always check generated facts.

**Constraints-first:** Explicitly require length, structure, audience, and other constraints.

**Temperature:** Controls generation randomness. Compare outputs at the same temperature when testing techniques.

**Evaluation:** The built-in keyword check is a basic, explainable proxy. Production-quality evaluation also needs manual review, factuality checks, safety tests, and representative datasets.

**API cost:** Free service tiers can be rate-limited or changed by the provider. A public app with a shared API key can exhaust the owner's quota.

**Privacy:** Do not paste passwords, personal records, proprietary content, or confidential documents into the prompt. Provider privacy terms apply to submitted content.

**Deployment:** See `DEPLOYMENT.md` in this project for GitHub + Streamlit Community Cloud steps.
    """)

st.markdown("---")
st.caption("PromptForge AI v1.0 · Python / Streamlit / Groq · Learning and portfolio demonstration")
