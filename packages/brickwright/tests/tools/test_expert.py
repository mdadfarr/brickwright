"""ask_expert: the packet (sphere, column below, connected check), the calls
(conversations, cached start, rescue model, caps, budget), and the loop's cost."""

import copy
import json

import pytest

from brickwright.driver import run_agent
from brickwright.tools import available_tools, execute_tool, expert

# A 2x2 red tower (10 bricks), a blue 1x1 floating above it with a gap, and 20
# white 1x1 bricks on the ground, far to the side.
_FLOATING_ABOVE_TOWER = """\
from bricklayer import Color, LegoBuild, Position, brick
from brickwright import TestContext

def create_build():
    m = LegoBuild()
    for z in range(10):
        m.add(brick(2, 2, color=Color.RED), position=Position(0, 0, z))
    m.add(brick(1, 1, color=Color.BLUE), position=Position(0, 0, 12))
    for k in range(20):
        m.add(brick(1, 1, color=Color.WHITE), position=Position(4 + 2 * k, 0, 0))
    return m

def run_tests():
    ctx = TestContext(create_build())
    ctx.check_no_floating()
    return ctx.report()
"""

# A 3-brick tower (the largest group) and four loose 1x1 bricks at growing gaps.
_SEPARATE_GROUPS = """\
from bricklayer import Color, LegoBuild, Position, brick
from brickwright import TestContext

def create_build():
    m = LegoBuild()
    for z in range(3):
        m.add(brick(2, 2, color=Color.RED), position=Position(0, 0, z))
    for x in (15, 6, 3, 10):
        m.add(brick(1, 1, color=Color.BLUE), position=Position(x, 0, 0))
    return m

def run_tests():
    ctx = TestContext(create_build())
    ctx.check_connected()
    return ctx.report()
"""

_FIXED = _FLOATING_ABOVE_TOWER.replace(
    "    m.add(brick(1, 1, color=Color.BLUE), position=Position(0, 0, 12))\n", ""
)

REMOVE_BLUE = {
    "diagnosis": "the blue brick floats two bricks above the tower",
    "edits": [
        {"action": "remove", "part": "3005", "color": "BLUE", "x": 0, "y": 0, "z": 12}
    ],
}


def part(x, y, z, w=1, d=1, h=1, color="RED", part_id="3005"):
    return {
        "part": part_id,
        "color": color,
        "position": [x, y, z],
        "top": [x + w, y + d, z + h],
        "rotation": 0,
    }


# --- The sphere -----------------------------------------------------------------


def test_sphere_is_the_nearest_ten_percent_with_a_minimum_of_8():
    row = [part(x, 0, 0) for x in range(30)]  # 10% is 3: the minimum of 8 applies
    chosen, below = expert.sphere(row, [15])
    assert chosen == [11, 12, 13, 14, 15, 16, 17, 18] and below == []

    field = [part(x, y, 0) for y in range(10) for x in range(10)]  # 100 parts: 10
    chosen, _ = expert.sphere(field, [0])
    assert len(chosen) == 10 and 0 in chosen and 99 not in chosen


def test_sphere_adds_the_column_below_down_to_the_ground():
    tower = [part(0, 0, z, w=2, d=2) for z in range(10)]  # indices 0..9
    floating = part(0, 0, 12, color="BLUE")  # index 10
    beside = [
        part(2, 0, z) for z in range(10)
    ]  # touching the tower, not under the part
    near = [part(x, 0, 12) for x in (2, 3, 4, 5, 6, 7, 8, 9)]  # the 8 nearest parts
    parts = tower + [floating] + beside + near
    chosen, below = expert.sphere(parts, [10])
    assert below == list(range(10))  # the whole tower, z=0 included
    assert set(range(10)) <= set(chosen) and 10 in chosen
    assert not set(range(11, 21)) & set(below)


def test_snapshot_finds_the_floating_part_and_its_column(make_record, in_process):
    record = make_record(_FLOATING_ABOVE_TOWER)
    snap = expert.snapshot(str(record.build_path), record.read_build(), "no_floating")
    parts, problem = snap["parts"], snap["problem"]
    assert [expert.part_text(parts[p]) for p in problem] == ["BLUE 3005 at (0,0,12)"]
    chosen, below = expert.sphere(parts, problem)
    reds = [i for i, p in enumerate(parts) if p["color"] == "RED"]
    assert below == reds and parts[below[0]]["position"] == [0.0, 0.0, 0.0]
    assert sorted(chosen) == sorted(reds + problem)  # the far white bricks stay out


def test_connected_check_uses_the_parts_outside_the_largest_group(
    make_record, in_process
):
    record = make_record(_SEPARATE_GROUPS)
    snap = expert.snapshot(str(record.build_path), record.read_build(), "connected")
    assert snap["found"] and not snap["passed"]
    assert "found 5" in snap["detail"]  # the detail names no parts
    problem = [expert.part_text(snap["parts"][p]) for p in snap["problem"]]
    # nearest to the tower first, at most 3: x=3, 6, 10 (not 15)
    assert problem == [
        "BLUE 3005 at (3,0,0)",
        "BLUE 3005 at (6,0,0)",
        "BLUE 3005 at (10,0,0)",
    ]


def test_other_checks_find_parts_named_in_the_detail():
    parts = [part(0, 0, 0, part_id="3001"), part(4, 0, 0, part_id="3001", color="BLUE")]
    centers = [[1, 1, 0], [5, 1, 0]]
    assert expert.named_in("3001 Blue: used 3, owned 2", parts, centers) == [1]
    assert expert.named_in("Brick @ Position(1, 1, 0)", parts, centers) == [0]


# --- The calls -------------------------------------------------------------------


class FakeExpert:
    """A scripted expert: each call returns the next reply (a dict, text or an
    exception) with the given cost, and records the call's kwargs."""

    def __init__(self, *replies, cost=0.1):
        self.replies, self.cost, self.calls = list(replies), cost, []

    def __call__(self, **kwargs):
        self.calls.append(
            copy.deepcopy({k: v for k, v in kwargs.items() if k != "response_format"})
        )
        self.calls[-1]["response_format"] = kwargs.get("response_format")
        reply = self.replies[min(len(self.calls), len(self.replies)) - 1]
        if isinstance(reply, BaseException):
            raise reply
        return _response(
            json.dumps(reply) if isinstance(reply, dict) else reply, self.cost
        )


def _response(text, cost, prompt=1000, completion=100):
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
        _hidden_params={"response_cost": cost},
    )


@pytest.fixture
def expert_on(monkeypatch):
    """Turn the tool on, with a fake context hook."""
    monkeypatch.setenv(expert.EXPERT_CONTEXT_ENV, "voxel_like.hook:context")
    hook = lambda record_dir, parts, problem: {
        "reference": "PARTS YOU OWN:\n3005 RED 40\n3005 BLUE 10",
        "details": "PARTS LEFT:\n3005 RED 30",
    }
    monkeypatch.setattr(expert, "context_hook", lambda: hook)


@pytest.fixture
def fake_expert(monkeypatch):
    """Install a FakeExpert as the completion seam ask_expert uses."""
    from brickwright.driver import loop

    def install(*replies, cost=0.1):
        fake = FakeExpert(*replies, cost=cost)
        monkeypatch.setattr(loop, "litellm_complete", lambda: fake)
        return fake

    return install


def ask(record, config, check="no_floating"):
    return execute_tool(
        record, config, "ask_expert", {"record_id": record.id, "check": check}
    )


def calls_in(record):
    return expert.load_state(record)["calls"]


def test_off_by_default(monkeypatch, make_record, config, in_process):
    monkeypatch.delenv(expert.EXPERT_CONTEXT_ENV, raising=False)
    assert "ask_expert" not in [t["function"]["name"] for t in available_tools(config)]
    record = make_record(_FLOATING_ABOVE_TOWER)
    assert "ask_expert" not in execute_tool(record, config, "validate", {}).content

    monkeypatch.setenv(expert.EXPERT_CONTEXT_ENV, "voxel_like.hook:context")
    assert "ask_expert" in [t["function"]["name"] for t in available_tools(config)]
    reply = execute_tool(record, config, "validate", {}).content
    assert f'ask_expert(record_id="{record.id}", check="no_floating")' in reply


def test_answer_is_exact_edits_and_is_logged(
    expert_on, fake_expert, make_record, config, in_process
):
    fake = fake_expert(REMOVE_BLUE, cost=0.12)
    record = make_record(_FLOATING_ABOVE_TOWER)
    outcome = ask(record, config)
    assert outcome.ok and "1. remove BLUE 3005 at (0,0,12)" in outcome.content
    assert outcome.usage.cost_usd == pytest.approx(0.12)
    call = fake.calls[0]
    assert call["model"] == "claude-sonnet-5-5" and call["timeout"] == 180
    assert call["response_format"] is expert.ExpertAnswer
    packet = call["messages"][1]["content"]
    assert "FAILED CHECK: no_floating" in packet and "BLUE 3005 at (0,0,12)" in packet
    assert "RED 3005 at (0,0,0) below" not in packet  # tower bricks are 2x2 (3003)
    assert "PARTS LEFT:\n3005 RED 30" in packet
    [entry] = calls_in(record)
    assert entry["outcome"] == "answered" and entry["model"] == "claude-sonnet-5-5"
    assert entry["cost_usd"] == pytest.approx(0.12) and entry["packet_parts"] == 11
    assert (
        entry["packet_chars"] == len(packet) and entry["answer"]["edits"][0]["z"] == 12
    )


def test_a_new_conversation_per_problem_with_an_identical_cached_start(
    expert_on, fake_expert, make_record, config, in_process
):
    fake = fake_expert(REMOVE_BLUE)
    record = make_record(_FLOATING_ABOVE_TOWER)
    ask(record, config)
    record.write_build(_FIXED)
    execute_tool(record, config, "validate", {})  # no_floating passes: problem 1 closed
    record.write_build(_FLOATING_ABOVE_TOWER)
    ask(record, config)
    first, second = fake.calls
    assert len(first["messages"]) == len(second["messages"]) == 2  # system + packet
    assert first["messages"][0] == second["messages"][0]
    system = first["messages"][0]["content"][0]
    assert system["cache_control"] == {"type": "ephemeral"}
    assert (
        system["text"].startswith(expert.INSTRUCTIONS)
        and "PARTS YOU OWN" in system["text"]
    )
    assert [c["problem"] for c in calls_in(record)] == [1, 2]
    assert [c["conversation"] for c in calls_in(record)] == ["new", "new"]


def test_five_tries_in_one_conversation_then_the_rescue_model(
    expert_on, fake_expert, make_record, config, in_process
):
    fake = fake_expert(REMOVE_BLUE)
    record = make_record(_FLOATING_ABOVE_TOWER)  # the fix is never applied
    for _ in range(7):
        assert ask(record, config).ok
    models = [c["model"] for c in fake.calls]
    assert models == ["claude-sonnet-5-5"] * 5 + ["claude-opus-5-5"] * 2
    sizes = [len(c["messages"]) for c in fake.calls]
    assert sizes == [2, 4, 6, 8, 10, 2, 4]  # same conversation; the rescue starts anew
    follow_up = fake.calls[1]["messages"][3]["content"]
    assert follow_up.startswith("The edits were applied, but the check still fails.")
    assert (
        fake.calls[5]["messages"][1] == fake.calls[0]["messages"][1]
    )  # the same packet
    assert [c["role"] for c in calls_in(record)] == ["expert"] * 5 + ["rescue"] * 2


def test_rescue_cap_refuses_and_says_to_stop(
    expert_on, fake_expert, make_record, config, in_process
):
    fake = fake_expert(REMOVE_BLUE)
    config = config.model_copy(update={"expert_max_tries": 1})
    record = make_record(_FLOATING_ABOVE_TOWER)
    for _ in range(4):  # 1 expert call, then the 3 rescues of the run
        assert ask(record, config).ok
    outcome = ask(record, config)
    assert not outcome.ok and "no expert help left" in outcome.content
    assert "Stop editing" in outcome.content and outcome.usage is None
    assert len(fake.calls) == 4
    assert calls_in(record)[-1]["outcome"] == "refused"
    assert expert.rescues_used(expert.load_state(record)) == 3


def test_budget_too_small_refuses_without_a_call(
    expert_on, fake_expert, make_record, config, in_process
):
    fake = fake_expert(REMOVE_BLUE, cost=0.3)
    record = make_record(_FLOATING_ABOVE_TOWER)
    outcome = ask(record, config.model_copy(update={"max_cost": 0.04}))
    assert (
        not outcome.ok and "refused" in outcome.content and "$0.040" in outcome.content
    )
    assert fake.calls == []
    assert ask(record, config).ok  # costs 0.3: now the estimate for this model
    outcome = ask(record, config.model_copy(update={"max_cost": 0.2}))
    assert "needs about $0.30" in outcome.content and len(fake.calls) == 1


def test_a_failed_call_tells_the_builder_to_fix_it_and_goes_on(
    expert_on, fake_expert, make_record, config, in_process
):
    fake_expert(TimeoutError("the builder call timed out 3 times (180 s each)"))
    record = make_record(_FLOATING_ABOVE_TOWER)
    outcome = ask(record, config)
    assert not outcome.ok and outcome.content == expert.UNAVAILABLE
    [entry] = calls_in(record)
    assert entry["outcome"] == "unavailable" and "timed out" in entry["reason"]
    assert expert.load_state(record)["problems"][0]["tries"] == 0


def test_a_credit_error_stops_the_run(
    expert_on, fake_expert, make_record, config, in_process
):
    fake_expert(
        RuntimeError("the API says the credit balance is too low (builder call)")
    )
    record = make_record(_FLOATING_ABOVE_TOWER)
    with pytest.raises(RuntimeError, match="credit balance"):
        ask(record, config)
    assert calls_in(record)[0]["outcome"] == "credit"


def test_an_unusable_answer_is_retried_once_in_the_same_conversation(
    expert_on, fake_expert, make_record, config, in_process
):
    wrong = {**REMOVE_BLUE, "edits": [{**REMOVE_BLUE["edits"][0], "z": 11}]}
    fake = fake_expert(wrong, REMOVE_BLUE, cost=0.05)
    record = make_record(_FLOATING_ABOVE_TOWER)
    outcome = ask(record, config)
    assert outcome.ok and outcome.usage.cost_usd == pytest.approx(0.1)
    retry = fake.calls[1]["messages"][-1]["content"]
    assert (
        "no such part is in the build" in retry and len(fake.calls[1]["messages"]) == 4
    )


def test_wrong_record_id_names_the_right_one(expert_on, make_record, config):
    record = make_record(_FLOATING_ABOVE_TOWER)
    outcome = execute_tool(
        record, config, "ask_expert", {"record_id": "nope", "check": "no_floating"}
    )
    assert not outcome.ok and record.id in outcome.content


# --- The loop: the run's cost and budget ---------------------------------------


class Builder:
    """The builder asks the expert once, then stops; expert calls (with a
    response_format) get REMOVE_BLUE."""

    def __init__(self, record, tool_call, response, expert_cost=0.2):
        self.turns = [
            response(
                tool_calls=[
                    tool_call(
                        "1", "ask_expert", record_id=record.id, check="no_floating"
                    )
                ],
                cost=0.1,
            ),
            response(content="done", cost=0.1),
        ]
        self.expert_cost, self.builder_calls, self.expert_calls, self.replies = (
            expert_cost,
            0,
            0,
            [],
        )

    def __call__(self, **kwargs):
        if "response_format" in kwargs:
            self.expert_calls += 1
            return _response(json.dumps(REMOVE_BLUE), self.expert_cost)
        if kwargs["messages"][-1].get("role") == "tool":
            self.replies.append(kwargs["messages"][-1]["content"])
        self.builder_calls += 1
        return self.turns[self.builder_calls - 1]


@pytest.mark.integration
def test_expert_cost_counts_in_the_run(
    expert_on, patch_complete, make_record, config, in_process, tool_call, response
):
    record = make_record(_FLOATING_ABOVE_TOWER)
    builder = patch_complete(Builder(record, tool_call, response))
    report = run_agent(record, config)
    assert builder.expert_calls == 1
    assert report.usage.cost_usd == pytest.approx(0.4)  # 0.1 + 0.2 + 0.1
    assert report.subagent_usage.cost_usd in (None, 0, 0.0)


@pytest.mark.integration
def test_expert_gets_only_what_is_left_of_the_budget(
    expert_on, patch_complete, make_record, config, in_process, tool_call, response
):
    record = make_record(_FLOATING_ABOVE_TOWER)
    builder = patch_complete(Builder(record, tool_call, response))
    run_agent(
        record, config.model_copy(update={"max_cost": 0.12})
    )  # 0.02 left after turn 1
    assert builder.expert_calls == 0
    assert "refused" in builder.replies[0] and "$0.020" in builder.replies[0]
