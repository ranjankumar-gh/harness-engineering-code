"""No API key, no network, no model."""

from __future__ import annotations

import pytest

from harness.audit import QUESTIONS, Area, Citation, run_audit
from harness.errors import NotACitation


def test_every_question_has_a_unique_number() -> None:
    numbers = [q.number for q in QUESTIONS]
    assert numbers == list(range(1, 16))


def test_prose_is_not_a_citation() -> None:
    with pytest.raises(NotACitation):
        Citation("the router handles that")


def test_a_path_without_a_line_is_not_a_citation() -> None:
    with pytest.raises(NotACitation):
        Citation("app/tools.py")


def test_a_citation_exposes_its_path() -> None:
    assert Citation("app/tools.py:34").path == "app/tools.py"


def test_a_control_living_in_a_prompt_is_flagged() -> None:
    assert Citation("prompts/support_agent.jinja:40").in_prompt
    assert not Citation("app/tools.py:34").in_prompt


def test_unanswered_questions_are_absent_not_failed() -> None:
    result = run_audit({6: "app/tools.py:34"})
    assert len(result.cited) == 1
    assert len(result.absent) == 14
    assert 7 in {q.number for q in result.absent}


def test_counts_are_reported_per_area() -> None:
    result = run_audit({6: "app/tools.py:34", 8: "app/limits.py:12"})
    assert result.by_area()[Area.DOES] == (2, 6)
    assert result.by_area()[Area.SEES] == (0, 5)


def test_an_unknown_question_number_raises() -> None:
    with pytest.raises(KeyError):
        run_audit({99: "app/tools.py:1"})


def test_the_worked_example_scores_five_of_fifteen() -> None:
    from examples.ch01_audit import BEFORE

    result = run_audit(BEFORE)
    assert len(result.cited) == 5
    assert result.in_prompts == (5, 8)
    assert "5 of 15 controls cited" in result.report()
