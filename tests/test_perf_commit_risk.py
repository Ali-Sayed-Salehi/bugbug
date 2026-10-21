# -*- coding: utf-8 -*-
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

import collections
import contextlib
import re
import time
from datetime import datetime, timezone
from typing import Any

import hglib
import numpy as np
import pytest
import responses

from bugbug import db, perf
from bugbug.perf import commit_risk, extraction

A, B, C, D, E = ("a" * 40, "b" * 40, "c" * 40, "d" * 40, "e" * 40)
NULL = "0" * 40
UTC = timezone.utc


def _changeset(
    node,
    desc="Bug 1 - Change something",
    author="Dev <dev@example.com>",
    parents=None,
    backsoutnodes=None,
):
    return {
        "node": node,
        "desc": desc,
        "author": author,
        "parents": parents if parents is not None else [B],
        "backsoutnodes": backsoutnodes or [],
    }


@pytest.mark.parametrize(
    "changeset, expected",
    [
        (_changeset(A), None),
        (
            _changeset(
                A,
                desc="Backed out changeset abc123 for perf regressions",
                backsoutnodes=["abc123"],
            ),
            "backout",
        ),
        (_changeset(A, parents=[B, C]), "merge"),
        (
            _changeset(
                A,
                desc="Bug 1 [wpt PR 42] - Sync tests, a=testonly",
                author="moz-wptsync-bot <wptsync@mozilla.com>",
            ),
            "wptsync",
        ),
        (_changeset(A, desc="Bug 1 - Sync tests\n\nwpt-pr: 42"), "wptsync"),
        (
            _changeset(A, desc="Bug 1 - Reformat tree. # ignore-this-changeset"),
            "ignore-this-changeset",
        ),
        # A revert hg.mozilla.org could not resolve is still a backout.
        (
            _changeset(
                A,
                desc=f'Revert "Bug 1 - Thing" for failures\n\nThis reverts commit {B}.',
            ),
            "backout",
        ),
        (_changeset(A, desc="Backed out the change by hand, no hash"), None),
        # Rules are evaluated in order: a backout of a wpt sync is a backout.
        (
            _changeset(
                A,
                desc="Backed out changeset abc123\n\nwpt-pr: 42",
                backsoutnodes=["abc123"],
            ),
            "backout",
        ),
    ],
)
def test_ignore_reason_rules(changeset, expected) -> None:
    assert commit_risk.get_ignore_reason(changeset) == expected


def test_parse_backout_nodes() -> None:
    parse = commit_risk.parse_backout_nodes
    assert parse("Backed out changeset abcdef123456 (bug 1) for failures") == [
        "abcdef123456"
    ]
    assert parse(
        "Backed out 2 changesets (bug 1) for failures\n\n"
        "Backed out changeset 0123456789ab (bug 1)\n"
        "Backed out changeset ba9876543210 (bug 1)"
    ) == ["0123456789ab", "ba9876543210"]
    assert parse(f'Revert "Bug 1 - Thing"\n\nThis reverts commit {A}.') == [A]
    assert parse("Backout of abcdef123456 and abcdef123456") == ["abcdef123456"]
    # Only a message announcing a backout counts, and it must name a changeset.
    assert parse("Bug 1 - Handle backout of abcdef123456 in the UI") == []
    assert parse("Backed out the change for failures") == []
    assert parse("") == []


class FakeHg:
    """Answers the showconfig, log and export calls the scorer makes on a clone."""

    def __init__(self, commits, patches=None, mozext=True):
        self.commits = commits
        self.patches = patches or {}
        self.mozext = mozext
        self.log_calls: list[str] = []

    def rawcommand(self, args):
        if args[0] == b"showconfig":
            if self.mozext == "disabled":
                return b"extensions.mozext=!\n"
            if self.mozext:
                return b"extensions.purge=\nextensions.mozext=/vct/hgext/mozext\n"
            raise hglib.error.CommandError(args, 1, b"", b"")
        options = dict(
            arg.decode().split("=", 1) for arg in args if arg.startswith(b"--")
        )
        template, revset = options["--template"], options["--rev"]
        self.log_calls.append(revset)
        out = ""
        for node in re.findall(r"present\(([0-9a-f]{40})\)", revset):
            commit = self.commits.get(node)
            if commit is None:
                continue
            parents = commit.get("parents", [B])
            fields = [
                node,
                commit.get("author", "Dev <dev@example.com>"),
                commit["desc"],
                parents[0] if parents else NULL,
                parents[1] if len(parents) > 1 else NULL,
            ]
            if "backsoutnodes" in template:
                fields.append(" ".join(commit.get("backsoutnodes", [])))
            out += "\0".join(fields) + "\0"
        return out.encode()

    def export(self, revs, git=True):
        return self.patches[revs[0].decode()]


def test_mozext_enabled() -> None:
    assert commit_risk.mozext_enabled(FakeHg({}, mozext=True))
    assert not commit_risk.mozext_enabled(FakeHg({}, mozext=False))
    assert not commit_risk.mozext_enabled(FakeHg({}, mozext="disabled"))


def test_changesets_from_clone() -> None:
    hg = FakeHg(
        {
            A: {"desc": "Bug 1 - Change", "parents": [B]},
            C: {"desc": "Merge autoland to central", "parents": [A, B]},
            D: {
                "desc": f"Backed out changeset {A[:12]} (bug 1)",
                "parents": [C],
                "backsoutnodes": [A],
            },
        }
    )
    changesets = commit_risk.changesets_from_clone(hg, [A, C, D, E], mozext=True)
    assert set(changesets) == {A, C, D}
    assert changesets[A] == {
        "node": A,
        "desc": "Bug 1 - Change",
        "author": "Dev <dev@example.com>",
        "parents": [B],
        "backsoutnodes": [],
    }
    assert changesets[C]["parents"] == [A, B]
    assert changesets[D]["backsoutnodes"] == [A]
    assert [commit_risk.get_ignore_reason(changesets[node]) for node in (A, C, D)] == [
        None,
        "merge",
        "backout",
    ]
    assert hg.log_calls == ["+".join(f"present({node})" for node in (A, C, D, E))]
    assert commit_risk.changesets_from_clone(hg, [], mozext=True) == {}


def test_changesets_from_clone_without_mozext() -> None:
    hg = FakeHg(
        {D: {"desc": f"Backed out changeset {A[:12]} (bug 1) for failures"}},
        mozext=False,
    )
    changesets = commit_risk.changesets_from_clone(hg, [D], mozext=False)
    # The keyword is not requested, so Mercurial cannot render it empty.
    assert "backsoutnodes" not in hg.log_calls[0]
    assert changesets[D]["backsoutnodes"] == [A[:12]]
    assert commit_risk.get_ignore_reason(changesets[D]) == "backout"


def _push(push_id, time, nodes, pushlog_id=None):
    return {
        "push_id": push_id,
        "repository": "autoland",
        "revision": nodes[-1],
        "time": time,
        "pushlog_id": pushlog_id,
        "commits": [
            {"revision": node, "desc": "Bug 1 - x", "bug_id": 1, "backout": False}
            for node in nodes
        ],
        "perf_jobs": [],
    }


PUSHES = [
    _push(2, "2026-05-02T00:00:00Z", [C], 12),
    _push(1, "2026-05-01T00:00:00Z", [A, B], 11),
    _push(3, "2026-05-03T00:00:00Z", [D]),
]


def test_select_unscored_commits() -> None:
    selected = commit_risk.select_unscored_commits(PUSHES, {B})
    assert [commit["node"] for commit in selected] == [A, C, D]
    assert selected[0] == {
        "node": A,
        "push_id": 1,
        "pushlog_id": 11,
        "push_time": "2026-05-01T00:00:00Z",
    }
    assert selected[2]["pushlog_id"] is None

    since = datetime(2026, 5, 2, tzinfo=UTC)
    until = datetime(2026, 5, 3, tzinfo=UTC)
    select = commit_risk.select_unscored_commits
    assert [c["node"] for c in select(PUSHES, set(), since, until)] == [C]
    assert [c["node"] for c in select(PUSHES, set(), until=until)] == [A, B, C]
    assert [c["node"] for c in select(PUSHES, set(), since=since)] == [C, D]


class FakeModel:
    def __init__(self, version="perf-regression-predictor-v1"):
        self.version = version
        self.items = []

    def classify(self, items, probabilities=False):
        self.items.extend(items)
        return np.array([[0.25, 0.75]])

    def get_extra_data(self):
        return {
            "model_name": "Perf Regression Predictor",
            "model_version": self.version,
        }


PATCH = (
    b"# HG changeset patch\n# User Dev <dev@example.com>\n# Date 1 0\n# Node ID "
    + A.encode()
    + b"\n# Parent "
    + B.encode()
    + b"\nBug 1 - Speed up\n\nMore details\n\n"
    b"diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n-a\n+b\n"
)

COMMITS = [
    {"node": A, "push_id": 1, "pushlog_id": 11, "push_time": "2026-05-01T00:00:00Z"},
    {"node": D, "push_id": 3, "pushlog_id": None, "push_time": "2026-05-03T00:00:00Z"},
    {"node": E, "push_id": 4, "pushlog_id": 14, "push_time": "2026-05-04T00:00:00Z"},
]

BACKOUT_DESC = f"Backed out changeset {A[:12]} (bug 1) for failures"


def _clone(monkeypatch, hg):
    monkeypatch.setattr(
        commit_risk.hglib, "open", lambda repo_dir: contextlib.nullcontext(hg)
    )


def test_score_commits(monkeypatch, caplog) -> None:
    hg = FakeHg(
        {
            A: {"desc": "Bug 1 - Speed up", "parents": [B]},
            D: {"desc": BACKOUT_DESC, "parents": [A], "backsoutnodes": [A]},
        },
        {A: PATCH},
    )
    _clone(monkeypatch, hg)
    model = FakeModel()
    stats: collections.Counter = collections.Counter()
    records = list(
        commit_risk.score_commits(
            model, "v1", "/repo", COMMITS, chunk_size=2, stats=stats
        )
    )

    assert [record["node"] for record in records] == [A, D]
    scored, ignored = records
    assert scored["ignored_reason"] is None
    assert scored["prob"] == [0.25, 0.75]
    assert scored["class"] == 1 and scored["risk_score"] == 0.75
    assert scored["push_id"] == 1 and scored["pushlog_id"] == 11
    assert scored["push_time"] == "2026-05-01T00:00:00Z"
    assert ignored["ignored_reason"] == "backout"
    assert ignored["prob"] is None and ignored["class"] is None
    assert ignored["risk_score"] is None
    assert all(r["model_version"] == "v1" and r["scored_at"] for r in records)
    # The model gets the message peeled off the patch and the whole patch as diff.
    assert model.items == [
        {"commit_message": "Bug 1 - Speed up\n\nMore details", "diff": PATCH.decode()}
    ]
    # Three commits in chunks of two take two hg log calls; the commit missing
    # from the clone is counted, not recorded.
    assert len(hg.log_calls) == 2
    assert stats == {"scored": 1, "ignored": 1, "missing": 1}
    assert "mozext" not in caplog.text


def test_score_commits_without_mozext_warns(monkeypatch, caplog) -> None:
    hg = FakeHg({D: {"desc": BACKOUT_DESC, "parents": [A]}}, mozext=False)
    _clone(monkeypatch, hg)
    records = list(commit_risk.score_commits(FakeModel(), "v1", "/repo", COMMITS[1:2]))
    assert [record["ignored_reason"] for record in records] == ["backout"]
    assert commit_risk.MOZEXT_WARNING in caplog.text


def test_score_commits_stops_at_the_deadline(monkeypatch) -> None:
    _clone(monkeypatch, FakeHg({A: {"desc": "Bug 1 - x"}}, {A: PATCH}))
    stats: collections.Counter = collections.Counter()
    records = list(
        commit_risk.score_commits(
            FakeModel(),
            "v1",
            "/repo",
            COMMITS,
            deadline=time.monotonic() - 1,
            stats=stats,
        )
    )
    assert records == []
    assert stats == {"unscored": 3}


def test_model_version() -> None:
    assert commit_risk.get_model_version(FakeModel()) == "perf-regression-predictor-v1"
    assert commit_risk.get_model_version(FakeModel("2026-10-01")) == "2026-10-01"
    with pytest.raises(ValueError):
        commit_risk.get_model_version(FakeModel(None))


def test_commit_risk_dataset_round_trip() -> None:
    record = {"node": A, "push_id": 1, "model_version": "v1", "risk_score": 0.5}
    db.write(perf.PERF_COMMIT_RISK_DB, [record])
    assert list(perf.get_perf_commit_risk()) == [record]


def test_pushlog_windows() -> None:
    def t(day, hour=0):
        return datetime(2026, 5, day, hour, tzinfo=UTC)

    windows = extraction.pushlog_windows([t(10), t(1), t(2, 12), t(20), t(3, 23)], 3)
    assert windows == [(t(1), t(4)), (t(10), t(13)), (t(20), t(23))]
    assert extraction.pushlog_windows([], 3) == []


@responses.activate
def test_fetch_and_set_pushlog_ids() -> None:
    responses.add(
        responses.GET,
        extraction.PUSHLOG_URL,
        json={
            "lastpushid": 3,
            "pushes": {
                "1": {"changesets": [A], "date": 1, "user": "a@b.c"},
                "2": {"changesets": [], "date": 2, "user": "a@b.c"},
                "3": {"changesets": [B, C], "date": 3, "user": "a@b.c"},
            },
        },
    )
    windows = [(datetime(2026, 5, 1, tzinfo=UTC), datetime(2026, 5, 4, tzinfo=UTC))]
    pushlog_ids = extraction.fetch_pushlog_ids(windows)
    # Keyed by the head changeset, since that is Treeherder's push revision.
    assert pushlog_ids == {A: 1, C: 3}
    params = responses.calls[0].request.params
    assert params["version"] == "2" and params["tipsonly"] == "1"
    assert params["startdate"] == "2026-04-30 23:00:00"
    assert params["enddate"] == "2026-05-04 01:00:00"

    pushes: list[dict[str, Any]] = [
        {"revision": A, "pushlog_id": None},
        {"revision": C, "pushlog_id": 7},
        {"revision": D},
    ]
    assert extraction.set_pushlog_ids(pushes, pushlog_ids) == 1
    assert [push["pushlog_id"] for push in pushes] == [1, 7, None]
