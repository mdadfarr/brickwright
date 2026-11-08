"""ask_expert: a stronger model works out the fix for a failed check.

Off by default. Setting ``BRICKWRIGHT_EXPERT_CONTEXT`` to ``"module:function"``
turns it on (the same way ``BRICKWRIGHT_REQUIRED_CHECKS`` adds checks) and names
the context hook: ``function(record_dir, parts, problem)`` returns
``{"reference": ..., "details": ...}``. ``reference`` (say, the parts the user
owns) must not change during a run: it goes in the cached start of every
prompt. ``details`` (say, the parts left and the plan section) goes in the
packet.

The builder sends only the record ID and the name of the failed check. Code
builds a small packet: the check's detail, the problem parts, a "sphere" of the
nearest parts (10% of the build, at least ``MIN_SPHERE``) plus every part in the
column below a problem part down to the ground, and the hook's details.

Each problem (a failed check, until a validate passes it) gets its own
conversation with ``config.expert_model``; a fix that did not work is followed up
in the same conversation with the new check result, up to
``config.expert_max_tries`` calls. Then ``config.expert_rescue_model`` takes over
in a new conversation; it may be called ``config.expert_max_rescues`` times per
run. The reply is structured output (:class:`ExpertAnswer`), checked in code.

Every call is logged in ``expert.json`` in the record. The loop passes the run's
remaining budget as ``config.max_cost``; a call that may not fit is refused. A
call that fails (after the completion function's own retries) is answered with
:data:`UNAVAILABLE` and the build goes on; only a credit error stops the run.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ..config import Config
from ..record import Record, Usage, UsageTracker
from . import build_execution

EXPERT_CONTEXT_ENV = "BRICKWRIGHT_EXPERT_CONTEXT"
EXPERT_FILE = "expert.json"
UNAVAILABLE = "The expert is unavailable right now; fix this yourself."
CREDIT_TEXT = "credit balance is too low"
MIN_SPHERE = 8  # parts in the sphere, at least
SPHERE_SHARE = 0.10  # of the build's parts
MAX_PROBLEM_PARTS = 3
Z_TO_STUDS = 1.2  # a brick is 1.2 studs tall: distances are in studs
MAX_DETAIL = 2000  # characters of the check's detail in the packet
MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 180.0  # seconds, when config.request_timeout is not set
COST_FLOOR = {"expert": 0.05, "rescue": 0.25}  # USD a call is assumed to cost, at least
ANSWERED = ("answered", "unusable")  # outcomes of calls that were made (and paid)
EPS = 1e-6

INSTRUCTIONS = """\
You are an expert LEGO builder. Another model is building a model with BrickLayer \
(build.py) and one of its checks failed. Work out the smallest set of edits that \
makes that check pass without breaking the rest of the build.

You get the failed check and its detail, the problem parts, the parts around them \
(and every part in the column below a problem part, down to the ground), the parts \
still available, and the part of the plan the problem is in.

Each edit is "remove" or "add":
- remove: a part that is in the build, exactly as listed (part, color, position).
- add: a new part. Only add parts listed as available, in an available color.
Positions are build.py add() positions: the corner of the part with the smallest x, \
y and z; x and y in studs, z in bricks (a plate is 1/3 brick, so use .333 and .667). \
Rotation is about the vertical axis: 0, 90, 180 or 270 degrees. Colors are \
BrickLayer Color names such as RED or LIGHT_BLUISH_GRAY.

Common fixes: a floating part needs parts under it that reach down to something \
grounded; of two colliding parts, remove or move one; separate groups need a part \
that clips into both.

Reply with JSON only: a one-line diagnosis and the edits, in the order to apply them.
"""


# --- Turning it on --------------------------------------------------------------


def expert_enabled() -> bool:
    return bool(os.environ.get(EXPERT_CONTEXT_ENV, "").strip())


def context_hook() -> Callable[..., Any]:
    """The function ``BRICKWRIGHT_EXPERT_CONTEXT`` names."""
    module_name, _, attr = os.environ[EXPERT_CONTEXT_ENV].strip().partition(":")
    return getattr(importlib.import_module(module_name), attr)


# --- The answer ------------------------------------------------------------------


class ExpertEdit(BaseModel):
    action: Literal["remove", "add"]
    part: str = Field(min_length=1, description="LDraw part id, like 3023")
    color: str = Field(min_length=1, description="BrickLayer Color name, like RED")
    x: float
    y: float
    z: float
    rotation: Literal[0, 90, 180, 270] = 0


class ExpertAnswer(BaseModel):
    diagnosis: str = Field(min_length=1, description="one line: what is wrong")
    edits: list[ExpertEdit] = Field(min_length=1, max_length=20)


def answer_problems(answer: ExpertAnswer, parts: Sequence[dict]) -> list[str]:
    """What is wrong with an answer, in words the expert can act on."""
    import bricklayer as bl

    problems = []
    for n, edit in enumerate(answer.edits, start=1):
        if edit.color not in bl.Color.__members__:
            problems.append(f"edit {n}: {edit.color!r} is not a BrickLayer Color name")
        if edit.action == "remove" and not any(_is(edit, p) for p in parts):
            problems.append(
                f"edit {n} removes {edit_text(edit)}, but no such part is in the build"
            )
    return problems


def _is(edit: ExpertEdit, part: dict) -> bool:
    x, y, z = part["position"]
    return (
        edit.part == part["part"]
        and edit.color == part["color"]
        and max(abs(edit.x - x), abs(edit.y - y), abs(edit.z - z)) < 0.01
    )


def edit_text(edit: ExpertEdit) -> str:
    turn = f" rotated {edit.rotation}" if edit.rotation else ""
    return f"{edit.color} {edit.part} at {_xyz((edit.x, edit.y, edit.z))}{turn}"


def answer_text(answer: ExpertAnswer, model: str) -> str:
    """The reply the builder copies from."""
    lines = [
        f"Expert ({model}): {answer.diagnosis}",
        "Apply these edits to build.py exactly, in this order, then validate:",
    ]
    lines += [
        f"{n}. {edit.action} {edit_text(edit)}"
        for n, edit in enumerate(answer.edits, start=1)
    ]
    lines.append(
        "Positions are add() positions: x and y in studs, z in bricks; "
        "the corner with the smallest x, y and z."
    )
    return "\n".join(lines)


# --- The packet ---------------------------------------------------------------


def _num(value: float) -> str:
    return f"{round(value, 3):g}"


def _xyz(xyz: Sequence[float]) -> str:
    return "(" + ",".join(_num(v) for v in xyz) + ")"


def part_text(part: dict) -> str:
    turn = f" rotated {part['rotation']}" if part.get("rotation") else ""
    return f"{part['color']} {part['part']} at {_xyz(part['position'])}{turn}"


def _center(part: dict) -> tuple[float, float, float]:
    (x0, y0, z0), (x1, y1, z1) = part["position"], part["top"]
    return (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2 * Z_TO_STUDS


def distance(a: dict, b: dict) -> float:
    return math.dist(_center(a), _center(b))


def sphere_size(n_parts: int) -> int:
    return min(n_parts, max(MIN_SPHERE, math.ceil(SPHERE_SHARE * n_parts)))


def column_below(parts: Sequence[dict], problem: Sequence[int]) -> list[int]:
    """Parts under a problem part's footprint, down to the ground (by shape, not by
    connection, so a missing support shows as a gap)."""
    below = set()
    for p in problem:
        (px0, py0, pz0), (px1, py1, _) = parts[p]["position"], parts[p]["top"]
        for i, part in enumerate(parts):
            (x0, y0, _), (x1, y1, z1) = part["position"], part["top"]
            overlaps = x0 < px1 - EPS and px0 < x1 - EPS and y0 < py1 - EPS
            if i != p and overlaps and py0 < y1 - EPS and z1 <= pz0 + EPS:
                below.add(i)
    return sorted(below)


def sphere(
    parts: Sequence[dict], problem: Sequence[int]
) -> tuple[list[int], list[int]]:
    """(the parts in the packet, the ones of them in a column below a problem part).

    The nearest ``sphere_size`` parts to any problem part (the problem parts
    included), plus the column below each problem part."""
    if not parts or not problem:
        return [], []

    def nearest(i: int) -> float:
        return min(distance(parts[i], parts[p]) for p in problem)

    near = sorted(range(len(parts)), key=lambda i: (nearest(i), i))
    below = column_below(parts, problem)
    chosen = set(near[: sphere_size(len(parts))]) | set(below) | set(problem)
    return sorted(chosen), below


def outside_largest(
    parts: Sequence[dict], components: Sequence[Sequence[int]]
) -> list[int]:
    """For the connected check: every part outside the largest component, nearest
    to the largest component first."""
    if len(components) < 2:
        return []
    largest = max(components, key=len)  # the first of equal size
    inside = set(largest)

    def gap(i: int) -> float:
        return min(distance(parts[i], parts[j]) for j in largest)

    outside = [i for i in range(len(parts)) if i not in inside]
    return sorted(outside, key=lambda i: (gap(i), i))


_POSITION = re.compile(r"\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)")


def named_in(
    detail: str, parts: Sequence[dict], centers: Sequence[Sequence[float]]
) -> list[int]:
    """Parts a check's detail points at: by position (corner or center), else by
    part id on the same line (and color, when the line names one)."""
    found = []
    for match in _POSITION.finditer(detail):
        pos = [float(v) for v in match.groups()]
        for i, part in enumerate(parts):
            for xyz in (part["position"], centers[i]):
                if max(abs(a - b) for a, b in zip(pos, xyz, strict=True)) < 0.01:
                    found.append(i)
    if found:
        return list(dict.fromkeys(found))
    for line in detail.splitlines():
        by_id = [
            i
            for i, part in enumerate(parts)
            if re.search(rf"(?<![\w.]){re.escape(part['part'])}(?!\w)", line)
        ]
        text = line.lower()
        by_color = [i for i in by_id if _color_words(parts[i]["color"]) in text]
        found += by_color or by_id
    return list(dict.fromkeys(found))


def _color_words(color: str) -> str:
    return color.lower().replace("_", " ")


def packet_text(
    check: str,
    detail: str,
    parts: Sequence[dict],
    problem: Sequence[int],
    details: str,
) -> tuple[str, int]:
    """(the packet, how many parts it lists)."""
    chosen, below = sphere(parts, problem)
    return (
        "\n\n".join(
            [
                _problem_text(check, detail, parts, problem),
                _sphere_text(parts, chosen, below),
            ]
            + ([details.strip()] if details.strip() else [])
        ),
        len(chosen),
    )


def follow_up_text(
    check: str, detail: str, parts: Sequence[dict], problem: Sequence[int]
) -> tuple[str, int]:
    """After a fix that did not work: the new check result and the parts now."""
    chosen, below = sphere(parts, problem)
    text = "\n\n".join(
        [
            "The edits were applied, but the check still fails. New result:\n"
            + _problem_text(check, detail, parts, problem),
            _sphere_text(parts, chosen, below),
        ]
    )
    return text, len(chosen)


def _problem_text(
    check: str, detail: str, parts: Sequence[dict], problem: Sequence[int]
) -> str:
    detail = detail.strip() or "(no detail)"
    if len(detail) > MAX_DETAIL:
        detail = detail[:MAX_DETAIL] + " ..."
    lines = [f"FAILED CHECK: {check}", f"DETAIL: {detail}", "PROBLEM PARTS:"]
    lines += [f"- {part_text(parts[p])}" for p in problem] or ["- (none found)"]
    return "\n".join(lines)


def _sphere_text(
    parts: Sequence[dict], chosen: Sequence[int], below: Sequence[int]
) -> str:
    under = set(below)
    lines = [
        (
            f"PARTS AROUND THE PROBLEM ({len(chosen)} of {len(parts)} in the build; "
            "'below' marks the column under a problem part, down to the ground). "
            "Positions are add() positions: the corner with the smallest x, y and z; "
            "x and y in studs, z in bricks."
        )
    ]
    lines += [
        f"- {part_text(parts[i])}{' below' if i in under else ''}" for i in chosen
    ]
    return "\n".join(lines)


# --- The build, read in a child process -------------------------------------------


def _rotation(leaf: Any) -> int:
    m = leaf.pose.rotation
    return round(math.degrees(math.atan2(float(m[1][0]), float(m[0][0])))) % 360


def _part(leaf: Any) -> dict:
    low, high = leaf.bounding_box
    return {
        "part": leaf.content.spec.ldraw_id,
        "color": leaf.content.color.name,
        "position": [float(low.x), float(low.y), float(low.z)],
        "top": [float(high.x), float(high.y), float(high.z)],
        "rotation": _rotation(leaf),
    }


def snapshot(build_path: str, source: str, check: str) -> dict:
    """Run build.py and its checks; return the parts, the check and its problem parts.

    Runs in a child process (build.py is the agent's code)."""
    from .validation import build_and_validate

    build, report = build_and_validate(build_path, source)
    if build is None:
        return {"error": report.build_error or "build.py did not run"}
    result = next((c for c in report.checks if c.name == check), None)
    leaves = list(build.flattened_placements)
    parts = [_part(leaf) for leaf in leaves]
    index = {id(leaf): i for i, leaf in enumerate(leaves)}

    def at(placement: Any) -> int | None:
        i = index.get(id(placement))
        if i is None:
            i = next((k for k, leaf in enumerate(leaves) if leaf == placement), None)
        return i

    detail = (result.detail or "") if result else ""
    if check == "no_floating":
        found = [at(p) for p in build.floating_placements]
    elif check == "no_collisions":
        found = [at(p) for pair in build.colliding_placements for p in pair]
    elif check == "connected":
        components = [[at(p) for p in group] for group in build.connected_components]
        found = outside_largest(
            parts, [[i for i in c if i is not None] for c in components]
        )
    else:
        centers = [[float(v) for v in leaf.pose.translation] for leaf in leaves]
        found = named_in(detail, parts, centers)
    problem = list(dict.fromkeys(i for i in found if i is not None))[:MAX_PROBLEM_PARTS]
    return {
        "error": None,
        "found": result is not None,
        "passed": bool(result and result.passed),
        "detail": detail,
        "failed": report.failed_checks,
        "parts": parts,
        "problem": problem,
    }


# --- State: expert.json ---------------------------------------------------------


def expert_path(record: Record | Path) -> Path:
    return Path(getattr(record, "path", record)) / EXPERT_FILE


def load_state(record: Record | Path) -> dict:
    path = expert_path(record)
    if not path.is_file():
        return {"problems": [], "calls": []}
    return json.loads(path.read_text())


def save_state(record: Record | Path, state: dict) -> None:
    expert_path(record).write_text(json.dumps(state, indent=2) + "\n")


def rescues_used(state: dict) -> int:
    return sum(
        1 for c in state["calls"] if c["role"] == "rescue" and c["outcome"] in ANSWERED
    )


def usage_by_model(record: Record | Path) -> dict[str, dict]:
    """Tokens and cost of a record's expert calls, per model."""
    totals: dict[str, dict] = {}
    for call in load_state(record)["calls"]:
        if not call.get("model"):
            continue
        row = totals.setdefault(
            call["model"],
            {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "calls": 0},
        )
        row["input_tokens"] += call.get("input_tokens", 0)
        row["output_tokens"] += call.get("output_tokens", 0)
        row["cost_usd"] += call.get("cost_usd") or 0.0
        row["calls"] += call["outcome"] in ANSWERED
    return totals


def close_passed(record: Record, failed_checks: Sequence[str]) -> None:
    """A validate passed these problems' checks: the next failure is a new problem."""
    state = load_state(record)
    changed = False
    for problem in state["problems"]:
        if problem["open"] and problem["check"] not in failed_checks:
            problem["open"] = False
            changed = True
    if changed:
        save_state(record, state)


# --- The tool -------------------------------------------------------------------


def ask_expert(
    record: Record, config: Config, record_id: str, check: str
) -> tuple[str, bool, Usage | None]:
    """(reply for the builder, ok, usage to add to the run)."""
    if record_id != record.id:
        return (
            f"ask_expert failed: wrong record_id; this build's record ID is {record.id!r}",
            False,
            None,
        )
    snap = build_execution.run_guarded(
        snapshot,
        (str(record.build_path), record.read_build(), check),
        config.build_timeout,
        lambda e: {"error": f"build.py did not run: {e}"},
    )
    if snap["error"]:
        return (
            f"ask_expert: build.py does not run; fix that first.\n{snap['error']}",
            False,
            None,
        )
    state = load_state(record)
    if not snap["found"]:
        failed = ", ".join(snap["failed"]) or "none"
        return (
            f"ask_expert: there is no check named {check!r}. Failed checks: {failed}.",
            False,
            None,
        )
    if snap["passed"]:
        close_passed(record, snap["failed"])
        return f"ask_expert: {check} passes now; validate again.", True, None
    try:
        context = context_hook()(record.path, snap["parts"], snap["problem"])
        reference, details = str(context["reference"]), str(context.get("details", ""))
    except Exception as exc:
        return (
            f"ask_expert failed: the context hook raised {type(exc).__name__}: {exc}",
            False,
            None,
        )

    problem = next(
        (p for p in state["problems"] if p["open"] and p["check"] == check), None
    )
    if problem is None:
        problem = {
            "id": len(state["problems"]) + 1,
            "check": check,
            "open": True,
            "tries": 0,
            "rescues": 0,
            "expert_messages": [],
            "rescue_messages": [],
        }
        state["problems"].append(problem)
    role = "expert" if problem["tries"] < config.expert_max_tries else "rescue"
    model = config.expert_model if role == "expert" else config.expert_rescue_model
    conversation = problem[f"{role}_messages"]
    if conversation:
        packet, n_parts = follow_up_text(
            check, snap["detail"], snap["parts"], snap["problem"]
        )
    else:
        packet, n_parts = packet_text(
            check, snap["detail"], snap["parts"], snap["problem"], details
        )
    entry: dict[str, Any] = {
        "problem": problem["id"],
        "check": check,
        "role": role,
        "model": model,
        "try": problem["tries" if role == "expert" else "rescues"] + 1,
        "conversation": "continued" if conversation else "new",
        "packet_chars": len(packet),
        "packet_parts": n_parts,
        "problem_parts": [part_text(snap["parts"][i]) for i in snap["problem"]],
    }

    def finish(
        outcome: str, reply: str, ok: bool, usage: Usage | None = None, **more: Any
    ):
        entry.update(outcome=outcome, **more)
        if usage is not None:
            entry.update(
                cost_usd=usage.cost_usd,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                cached_tokens=usage.prompt_tokens_details.cached_tokens,
            )
        state["calls"].append(entry)
        save_state(record, state)
        return reply, ok, usage

    if role == "rescue" and rescues_used(state) >= config.expert_max_rescues:
        entry["model"] = None
        return finish(
            "refused",
            f"ask_expert: no expert help left ({config.expert_max_rescues} rescue calls "
            f"used this run). Stop editing and say in your reply which check still "
            f"fails and why.",
            False,
            reason="rescue cap reached",
        )
    estimate = max(
        [COST_FLOOR[role]]
        + [c.get("cost_usd") or 0.0 for c in state["calls"] if c.get("model") == model]
    )
    if config.max_cost < estimate:
        entry["model"] = None
        return finish(
            "refused",
            f"ask_expert refused: ${max(config.max_cost, 0.0):.3f} of this run's budget "
            f"is left and a {model} call needs about ${estimate:.2f}. Fix it yourself "
            "if you can; otherwise stop and say which check still fails.",
            False,
            reason="budget",
        )

    system = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": INSTRUCTIONS + "\n" + reference.strip(),
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }
    messages = [*conversation, {"role": "user", "content": packet}]
    usages: list[Usage] = []
    try:
        answer, reply_text = _ask(
            config, model, system, messages, snap["parts"], usages
        )
    except Exception as exc:
        usage = _total(usages)
        if CREDIT_TEXT in str(exc).lower():
            finish("credit", "", False, usage, reason=str(exc)[:300])
            raise
        return finish(
            "unavailable",
            UNAVAILABLE,
            False,
            usage,
            reason=f"{type(exc).__name__}: {exc}"[:300],
        )
    usage = _total(usages)
    if answer is None:
        return finish(
            "unusable",
            "The expert's answer could not be used; fix this yourself.",
            False,
            usage,
            reason=reply_text[:300],
        )
    conversation += [messages[-1], {"role": "assistant", "content": reply_text}]
    problem["tries" if role == "expert" else "rescues"] += 1
    return finish(
        "answered", answer_text(answer, model), True, usage, answer=answer.model_dump()
    )


def _ask(
    config: Config,
    model: str,
    system: dict,
    messages: list[dict],
    parts: Sequence[dict],
    usages: list[Usage],
) -> tuple[ExpertAnswer | None, str]:
    """One call, and one retry in the same conversation if the answer is unusable.
    Returns (answer or None, the last reply text or the last error)."""
    from ..driver import loop  # the completion seam; imported late (loop imports tools)
    from ..driver.tracking import _usage_of

    complete = loop.litellm_complete()
    conversation = list(messages)
    error = ""
    for _ in range(2):
        response = complete(
            model=model,
            messages=[system, *conversation],
            response_format=ExpertAnswer,
            max_tokens=MAX_TOKENS,
            timeout=config.request_timeout or DEFAULT_TIMEOUT,
        )
        usages.append(_usage_of(response))
        text = _reply(response)
        try:
            answer = ExpertAnswer.model_validate_json(_strip_fences(text))
            problems = answer_problems(answer, parts)
            if not problems:
                return answer, text
            error = "; ".join(problems)
        except ValidationError as exc:
            error = f"the reply does not match the answer shape: {exc}"
        conversation += [
            {"role": "assistant", "content": text or "(empty reply)"},
            {
                "role": "user",
                "content": f"Your answer was rejected: {error}\nReply with corrected JSON only.",
            },
        ]
    return None, error


def _total(usages: Sequence[Usage]) -> Usage | None:
    """The sum of the calls' usage; None when no call came back."""
    if not usages:
        return None
    tracker = UsageTracker()
    for usage in usages:
        tracker.add(usage)
    return tracker.usage()


def _reply(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError):
        return ""


def _strip_fences(text: str) -> str:
    match = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
    return match.group(1) if match else text
