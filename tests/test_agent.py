"""Agent loop plumbing tested with a scripted fake LLM (no network, no key)."""
from types import SimpleNamespace as NS

from econchat.agent import EconAgent
from tests.test_analytics import synthetic


class FakeCompletions:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.script.pop(0)


def tool_call(name, args, cid="c1"):
    fn = NS(name=name, arguments=args)
    tc = NS(id=cid, type="function", function=fn,
            model_dump=lambda: {"id": cid, "type": "function", "function": {"name": name, "arguments": args}})
    return NS(choices=[NS(message=NS(content=None, tool_calls=[tc]))], usage=NS(prompt_tokens=10, completion_tokens=5))


def final(text):
    return NS(choices=[NS(message=NS(content=text, tool_calls=None))], usage=NS(prompt_tokens=20, completion_tokens=8))


def test_agent_runs_tools_then_answers():
    comp = FakeCompletions([
        tool_call("get_latest", '{"series": "GDP"}'),
        tool_call("make_chart", '{"kind": "comparison"}', "c2"),
        final("GDP was X."),
    ])
    agent = EconAgent(synthetic(), client=NS(chat=NS(completions=comp)), model="fake")
    turn = agent.ask("latest gdp?")
    assert turn.answer == "GDP was X."
    assert [t["tool"] for t in turn.tool_calls] == ["get_latest", "make_chart"]
    assert len(turn.figures) == 1
    assert turn.usage == {"prompt_tokens": 40, "completion_tokens": 18}
    # tool results were fed back to the model
    last_msgs = comp.calls[-1]["messages"]
    assert [m["role"] for m in last_msgs][-4:] == ["assistant", "tool", "assistant", "tool"]
    # history keeps only plain texts for follow-ups
    assert agent.history == [{"role": "user", "content": "latest gdp?"}, {"role": "assistant", "content": "GDP was X."}]


def test_bad_tool_arguments_return_error_not_crash():
    comp = FakeCompletions([tool_call("compute_change", '{"series": "GDP", "start": "1900", "end": "1901"}'), final("ok")])
    turn = EconAgent(synthetic(), client=NS(chat=NS(completions=comp)), model="fake").ask("gdp in 1900")
    assert "error" in turn.tool_calls[0]["result"]
