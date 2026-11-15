"""LLM agent loop over the tools. Works with any OpenAI-compatible endpoint
(OpenAI, Groq, Gemini's OpenAI endpoint, Ollama) via three env vars:
LLM_API_KEY, LLM_BASE_URL (blank for OpenAI) and LLM_MODEL."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import date

from dotenv import load_dotenv
from openai import OpenAI

from .analytics import Econ, label
from .tools import TOOLS, ToolRunner, to_json

load_dotenv()

DEFAULT_MODEL = "gpt-5-mini"
MAX_STEPS = 8

SYSTEM_PROMPT = """You are FRED Econ Chat, an economic data analyst that answers questions about two FRED series:
- GDP: U.S. Gross Domestic Product, quarterly, NOMINAL, billions of dollars at a seasonally adjusted annual rate (SAAR). Starts 1947 Q1.
- CPILFESL: Core CPI (all items less food and energy), monthly, seasonally adjusted index 1982-84=100. Starts 1957-01.

Today is {today}. Latest observations: GDP {gdp_latest}; Core CPI {cpi_latest}. Data retrieved: {retrieved}.

Rules:
1. Every number you state must come from a tool result in this conversation. Do not calculate, estimate or recall numbers yourself; call a tool instead. The only conversion you may do yourself is billions to trillions of dollars (divide by 1,000).
2. Always name the period of each figure (e.g. "2026 Q3", "October 2026") and the measure (level, YoY % change, annualized rate).
3. GDP is nominal. When discussing GDP growth alongside inflation, say so and use the real-growth proxy (nominal GDP growth minus core inflation) where relevant, noting it is an approximation.
4. "Core CPI inflation rate" means the YoY % change in the index. The index level is not an inflation rate. If the latest YoY cannot be computed because a base month is missing, say so plainly and give the latest computable YoY and the labelled estimate.
5. For "past N years" questions use years_back=N. When asked how INFLATION or GROWTH changed, describe the path of the YoY rate (rate at the start, peak or trough with its date, rate at the end), not only the change in the index level.
6. For any visualization request call make_chart. The chart appears automatically under your answer; describe what it shows in 2-4 sentences, referring to lines by the colors the tool reports. Never write image links or code.
7. Only these two series are available. For other data (unemployment, real GDP, PCE, rates) or forecasts, say it is outside the app's data and offer what is available. Do not give investment advice.
8. For summary requests, cover three time scales using trend_summary: the long run (decades: growth rates, the 1970s-80s inflation era, the low-inflation 2010s), the recent episode (pandemic shock, 2022 inflation peak and decline), and the latest readings plus how the two series relate.
9. When (and only when) discussing the relationship between the two series, note that correlation is not causation.

Style: lead with the direct answer in the first sentence (no "Short answer:" label), then 3-5 short bullets with supporting figures. Label each figure once, plainly; do not repeat units or definitions in parentheses after every number. Present GDP levels in trillions of dollars with two decimals (e.g. $14.61 trillion). Use 1-2 decimals for percentages. Keep answers under about 200 words unless the user asks for a summary or more detail."""


@dataclass
class Turn:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    figures: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    seconds: float = 0.0


def llm_config() -> dict:
    return {
        "api_key": os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("LLM_BASE_URL") or None,
        "model": os.getenv("LLM_MODEL") or DEFAULT_MODEL,
        "reasoning_effort": os.getenv("LLM_REASONING_EFFORT") or _default_effort(os.getenv("LLM_MODEL") or DEFAULT_MODEL,
                                                                                 os.getenv("LLM_BASE_URL")),
    }


def _default_effort(model: str, base_url: str | None) -> str | None:
    """OpenAI reasoning models default to 'medium'; 'low' is plenty here because
    the tools do the analysis, and it roughly halves latency. Other providers: unset."""
    if not base_url and (model.startswith("gpt-5") or model.startswith("o")):
        return "low"
    return None


class EconAgent:
    def __init__(self, econ: Econ, client: OpenAI | None = None, model: str | None = None):
        cfg = llm_config()
        if client is None:
            if not cfg["api_key"]:
                raise RuntimeError("No LLM key. Set LLM_API_KEY (or OPENAI_API_KEY) in .env; see README.")
            client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=60.0, max_retries=2)
        self.client = client
        self.model = model or cfg["model"]
        self.reasoning_effort = cfg["reasoning_effort"]
        self.econ = econ
        self.history: list[dict] = []  # user/assistant texts only, keeps follow-ups cheap

    def system_prompt(self) -> str:
        e = self.econ
        g = e.raw["GDP"].dropna()
        c = e.raw["CPILFESL"].dropna()
        retrieved = e.info.get("GDP", {}).get("fetched_at", "unknown")
        return SYSTEM_PROMPT.format(
            today=date.today().isoformat(),
            gdp_latest=label(g.index[-1], "Q"), cpi_latest=label(c.index[-1], "M"),
            retrieved=str(retrieved)[:16].replace("T", " ") + " UTC",
        )

    def ask(self, question: str) -> Turn:
        t0 = time.time()
        runner = ToolRunner(self.econ)
        messages = [{"role": "system", "content": self.system_prompt()}, *self.history[-10:],
                    {"role": "user", "content": question}]
        trace, usage = [], {"prompt_tokens": 0, "completion_tokens": 0}
        answer = ""
        for _ in range(MAX_STEPS):
            kwargs = {"model": self.model, "messages": messages, "tools": TOOLS}
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
            resp = self.client.chat.completions.create(**kwargs)
            if resp.usage:
                usage["prompt_tokens"] += resp.usage.prompt_tokens or 0
                usage["completion_tokens"] += resp.usage.completion_tokens or 0
            msg = resp.choices[0].message
            if not msg.tool_calls:
                answer = (msg.content or "").strip()
                break
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = runner.run(tc.function.name, args)
                trace.append({"tool": tc.function.name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": to_json(result)})
        else:
            answer = "I couldn't finish that analysis within the step limit. Try a more specific question."
        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        return Turn(answer, trace, runner.figures, usage, round(time.time() - t0, 1))

    def reset(self) -> None:
        self.history.clear()
