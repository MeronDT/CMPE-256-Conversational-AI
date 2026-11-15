"""Streamlit chat UI.  Run:  streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from econchat.agent import EconAgent, llm_config
from econchat.analytics import Econ
from econchat.data import SERIES, load_all

st.set_page_config(page_title="FRED Econ Chat", page_icon="📈", layout="wide")

EXAMPLES = [
    "What is the latest reported U.S. GDP?",
    "How has U.S. GDP changed over the past 10 years?",
    "What is the current Core CPI inflation rate?",
    "How has Core CPI inflation changed over the past five years?",
    "What is the relationship between GDP growth and Core CPI inflation?",
    "Generate a visualization comparing GDP growth and Core CPI inflation over time.",
    "Summarize the key economic trends observed in both datasets.",
]


@st.cache_resource(ttl=6 * 3600, show_spinner="Loading FRED data...")
def get_econ(refresh_token: int = 0) -> Econ:
    data, info = load_all(force=refresh_token > 0)
    return Econ(data, info)


def init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("refresh", 0)
    st.session_state.setdefault("pending", None)


init_state()
econ = get_econ(st.session_state.refresh)
cfg = llm_config()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("FRED Econ Chat")
    st.caption("Ask questions about U.S. GDP and Core CPI (CPILFESL) in plain English.")
    st.subheader("Try a question")
    for q in EXAMPLES:
        if st.button(q, width="stretch"):
            st.session_state.pending = q
    st.divider()
    show_tools = st.toggle("Show tool calls", value=False, help="See which calculations produced each answer.")
    if st.button("Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.session_state.pop("agent", None)
        st.rerun()
    st.divider()
    st.subheader("Data")
    for sid, meta in SERIES.items():
        s = econ.raw[sid].dropna()
        inf = econ.info.get(sid, {})
        st.markdown(f"**[{sid}]({meta.url})** · {meta.frequency == 'Q' and 'quarterly' or 'monthly'}  \n"
                    f"{s.index[0]:%Y-%m} to {s.index[-1]:%Y-%m} · source: {inf.get('source', '?')}")
        if inf.get("warning"):
            st.warning(inf["warning"])
    fetched = str(econ.info.get("GDP", {}).get("fetched_at", "?"))[:16].replace("T", " ")
    st.caption(f"Retrieved {fetched} UTC")
    if st.button("Refresh from FRED", width="stretch"):
        st.session_state.refresh += 1
        st.session_state.pop("agent", None)
        st.rerun()
    st.caption(f"Model: `{cfg['model']}`" + (f" via `{cfg['base_url']}`" if cfg["base_url"] else ""))

# ------------------------------------------------------------------ header + KPIs
st.title("U.S. GDP and Core Inflation, in conversation")
gl = econ.get_latest("GDP")
ci = econ.core_inflation()
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Nominal GDP, {gl['latest_period']}", f"${gl['latest_value_trillions']:.2f}T",
          help="Billions of dollars, seasonally adjusted annual rate")
c2.metric("GDP growth, QoQ annualized", f"{gl['qoq_annualized_growth_pct']:.1f}%",
          help=f"YoY: {gl['yoy_growth_pct']:.1f}%")
if ci["core_inflation_yoy_pct"] is not None:
    c3.metric(f"Core CPI inflation, {ci['as_of_month']}", f"{ci['core_inflation_yoy_pct']:.2f}% y/y")
else:
    lc = ci["latest_computable_yoy"]
    c3.metric(f"Core CPI inflation, {lc['month']}", f"{lc['pct']:.2f}% y/y",
              help=f"{ci['as_of_month']} YoY can't be computed: {ci['yoy_not_computable_reason']} "
                   f"Estimate with interpolated base: {ci['estimated_yoy_pct_interpolated_base']:.2f}%.")
c4.metric(f"Core CPI, 3-mo annualized, {ci['as_of_month']}", f"{ci['core_inflation_3m_annualized_pct']:.2f}%")

# ------------------------------------------------------------------ chat
if not cfg["api_key"]:
    st.error("No LLM key found. Copy `.env.example` to `.env`, add `LLM_API_KEY`, and restart. "
             "The README lists free options such as Groq.")
    st.stop()

if "agent" not in st.session_state:
    st.session_state.agent = EconAgent(econ)
agent: EconAgent = st.session_state.agent


def render(msg: dict, idx: int):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for j, fig in enumerate(msg.get("figures", [])):
            st.plotly_chart(fig, width="stretch", theme=None, key=f"fig-{idx}-{j}")
        if show_tools and msg.get("trace"):
            with st.expander(f"Tool calls ({len(msg['trace'])}) · {msg.get('seconds', 0)}s"):
                for t in msg["trace"]:
                    st.markdown(f"`{t['tool']}({', '.join(f'{k}={v!r}' for k, v in t['args'].items())})`")
                    st.json(t["result"], expanded=False)


for i, m in enumerate(st.session_state.messages):
    render(m, i)

prompt = st.chat_input("Ask about GDP, Core CPI inflation, or how they relate...")
if st.session_state.pending:
    prompt, st.session_state.pending = st.session_state.pending, None

if prompt:
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    render(user_msg, len(st.session_state.messages) - 1)
    with st.spinner("Analyzing FRED data..."):
        try:
            turn = agent.ask(prompt)
            reply = {"role": "assistant", "content": turn.answer or "_(no answer returned)_",
                     "figures": [f for f, _ in turn.figures], "trace": turn.tool_calls, "seconds": turn.seconds}
        except Exception as exc:
            reply = {"role": "assistant", "content": f"Sorry, the language model call failed: `{exc}`"}
    st.session_state.messages.append(reply)
    render(reply, len(st.session_state.messages) - 1)
