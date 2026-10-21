# -*- coding: utf-8 -*-
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

"""Performance regression risk of commits.

The rules deciding which commits the perf regression predictor skips and the
per-patch scoring are shared by the HTTP service endpoint
(``classify_perf_regression`` in ``bugbug_http.models``) and the batch scorer
(``scripts/perf_commit_risk_scorer.py``). The scorer runs the model once over
every commit of every push in the pushes snapshot and stores the scores in
``perf_commit_risk.json``, so evaluations replaying scheduling strategies over
history read scores instead of calling the endpoint.
"""

import collections
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator

import hglib

from bugbug import repository
from bugbug.perf.regression_labeling import is_within_window
from bugbug.utils import utcnow

logger = logging.getLogger(__name__)

# A message that announces a backout or a revert on its first line, with the
# changeset hashes it mentions. Stands in for the ``backsoutnodes`` of
# hg.mozilla.org on clones without the mozext extension, and backs it up where
# the extension resolved nothing.
BACKOUT_RE = re.compile(
    r"^\s*(?:back(?:ed|ing|s)?\s*out|backout|revert(?:ed|ing|s)?)\b", re.IGNORECASE
)
NODE_RE = re.compile(r"\b[0-9a-f]{12,40}\b")


def parse_backout_nodes(desc: str) -> list[str]:
    """Changesets a commit message says it backs out, in order of mention."""
    lines = desc.strip().splitlines()
    if not lines or not BACKOUT_RE.match(lines[0]):
        return []
    return list(dict.fromkeys(NODE_RE.findall(desc)))


# Commits the perf regression predictor must not score, keyed by the reason
# reported with the result. Rules are evaluated in order and the first match
# wins. Each predicate receives a changeset shaped like an hg.mozilla.org
# automationrelevance changeset (``node``, ``desc``, ``author``, ``parents``,
# ``backsoutnodes``); ``changesets_from_clone`` builds the same shape from a
# local clone.
IGNORED_COMMIT_RULES: dict[str, Callable[[dict[str, Any]], bool]] = {
    # Backouts revert an earlier patch, so their diff mirrors the original
    # change and tells nothing about new regressions. hg.mozilla.org resolves
    # the backed-out changesets by searching a bounded number of ancestors, so
    # a message announcing a backout counts even when it resolved none.
    "backout": lambda changeset: bool(
        changeset.get("backsoutnodes") or parse_backout_nodes(changeset.get("desc", ""))
    ),
    # Merges (e.g. mozilla-central into autoland) carry no change of their own.
    "merge": lambda changeset: len(changeset.get("parents", [])) > 1,
    # Automated web-platform-tests imports from upstream.
    "wptsync": lambda changeset: repository.is_wptsync(
        {
            "author_email": changeset.get("author", ""),
            "desc": changeset.get("desc", ""),
        }
    ),
    # Large mechanical changes (e.g. reformatting) excluded from blame.
    "ignore-this-changeset": lambda changeset: (
        "ignore-this-changeset" in changeset.get("desc", "")
    ),
}


def get_ignore_reason(changeset: dict[str, Any]) -> str | None:
    """Return why the perf regression predictor skips a commit, or None to score it."""
    for reason, matches in IGNORED_COMMIT_RULES.items():
        if matches(changeset):
            return reason
    return None


def extract_commit_message(patch_text: str) -> str:
    """The commit message of an ``hg export`` patch, empty when none is found."""
    # Imported here: the model module pulls in numpy and bugbug.model, which
    # the rest of the pipeline never needs.
    from bugbug.models.perf_regression_predictor import PatchCommitMessageExtractor

    return PatchCommitMessageExtractor()(patch_text) or ""


def score_patch(model: Any, patch: bytes) -> dict[str, Any]:
    """Score one ``hg export`` patch with the perf regression predictor.

    Returns the class probabilities, the predicted class and the risk score,
    which is the probability of the positive class. The model truncates the
    diff to its context window itself, so the whole patch is passed.
    """
    patch_text = patch.decode("utf-8", "replace")
    probabilities = model.classify(
        [{"commit_message": extract_commit_message(patch_text), "diff": patch_text}],
        probabilities=True,
    )[0]
    return {
        "prob": probabilities.tolist(),
        "class": int(probabilities.argmax()),
        "risk_score": float(probabilities[1]),
    }


LOG_FIELDS = ("node", "author", "desc", "p1node", "p2node", "backsoutnodes")
LOG_TEMPLATE = "".join(f"{{{field}}}\\0" for field in LOG_FIELDS)
# {backsoutnodes} comes from the mozext extension of version-control-tools,
# which hg.mozilla.org and the pipeline's clone run. Mercurial renders an
# unknown keyword as empty rather than failing, so a clone without the
# extension must not be asked for it.
LOG_TEMPLATE_WITHOUT_MOZEXT = "".join(f"{{{field}}}\\0" for field in LOG_FIELDS[:-1])
MOZEXT_WARNING = (
    "The clone does not enable the mozext extension; backouts are detected "
    "from commit messages instead of {backsoutnodes}"
)


def mozext_enabled(hg: hglib.client.hgclient) -> bool:
    """Whether the clone enables the mozext Mercurial extension."""
    try:
        out = hg.rawcommand([b"showconfig", b"extensions"])
    except hglib.error.CommandError:
        # Exit status 1 when the section is empty.
        return False
    for line in out.splitlines():
        name, _, value = line.partition(b"=")
        if name.strip() == b"extensions.mozext" and not value.strip().startswith(b"!"):
            return True
    return False


# Mercurial's hash for "no commit", which {p1node} and {p2node} render when a
# parent slot is empty.
NULL_NODE = "0" * 40


def changesets_from_clone(
    hg: hglib.client.hgclient, nodes: Iterable[str], mozext: bool
) -> dict[str, dict[str, Any]]:
    """Describe commits of a local clone like automationrelevance does.

    Returns a changeset per node found, keyed by node, with ``node``, ``desc``,
    ``author``, ``parents`` and ``backsoutnodes``. Nodes the clone does not
    have are left out rather than failing the whole batch. With ``mozext``
    the clone resolves ``backsoutnodes`` the way hg.mozilla.org does;
    without it they are parsed from the commit message.
    """
    nodes = list(nodes)
    if not nodes:
        return {}
    revset = "+".join(f"present({node})" for node in nodes)
    template = LOG_TEMPLATE if mozext else LOG_TEMPLATE_WITHOUT_MOZEXT
    args = hglib.util.cmdbuilder(b"log", template=template, rev=revset)
    values = [
        value.decode("utf-8", "replace") for value in hg.rawcommand(args).split(b"\x00")
    ]
    fields = template.count("\\0")
    changesets = {}
    for i in range(0, len(values) - 1, fields):
        node, author, desc, p1node, p2node = values[i : i + 5]
        parents = [p1node] if p1node != NULL_NODE else []
        if p2node != NULL_NODE:
            parents.append(p2node)
        changesets[node] = {
            "node": node,
            "desc": desc,
            "author": author,
            "parents": parents,
            "backsoutnodes": (
                values[i + 5].split() if mozext else parse_backout_nodes(desc)
            ),
        }
    return changesets


def select_unscored_commits(
    pushes: Iterable[dict[str, Any]],
    scored: set[str],
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[dict[str, Any]]:
    """Commits of the pushes in ``[since, until)`` whose node is not in ``scored``.

    Returned in push order, each with the push fields a score record carries:
    ``node``, ``push_id``, ``pushlog_id`` and ``push_time``.
    """
    end = until if until is not None else datetime.max.replace(tzinfo=timezone.utc)
    commits = []
    for push in sorted(pushes, key=lambda push: (push["time"], push["push_id"])):
        if not is_within_window(push["time"], since, end):
            continue
        for commit in push["commits"]:
            if commit["revision"] in scored:
                continue
            commits.append(
                {
                    "node": commit["revision"],
                    "push_id": push["push_id"],
                    "pushlog_id": push.get("pushlog_id"),
                    "push_time": push["time"],
                }
            )
    return commits


def score_commits(
    model: Any,
    model_version: str,
    repo_dir: str,
    commits: list[dict[str, Any]],
    chunk_size: int = 50,
    deadline: float | None = None,
    stats: collections.Counter | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield one ``perf_commit_risk.json`` record per commit found in ``repo_dir``.

    Commits are described ``chunk_size`` at a time with one ``hg log``, then
    exported and scored one by one so a single patch is in memory at a time.
    Ignored commits get their reason and null scores. Commits missing from
    the clone, such as ones not yet merged to mozilla-central, are skipped and
    counted in ``stats`` so a later run scores them. Stops cleanly once
    ``time.monotonic()`` passes ``deadline``.
    """
    stats = stats if stats is not None else collections.Counter()
    scored_at = utcnow().isoformat()
    with hglib.open(repo_dir) as hg:
        mozext = mozext_enabled(hg)
        if not mozext:
            logger.warning(MOZEXT_WARNING)
        for start in range(0, len(commits), chunk_size):
            chunk = commits[start : start + chunk_size]
            changesets = changesets_from_clone(hg, (c["node"] for c in chunk), mozext)
            for commit in chunk:
                if deadline is not None and time.monotonic() >= deadline:
                    logger.info("Time budget exhausted, stopping")
                    stats["unscored"] += len(commits) - start - chunk.index(commit)
                    return
                changeset = changesets.get(commit["node"])
                if changeset is None:
                    stats["missing"] += 1
                    continue
                record = {
                    **commit,
                    "ignored_reason": get_ignore_reason(changeset),
                    "prob": None,
                    "class": None,
                    "risk_score": None,
                    "model_version": model_version,
                    "scored_at": scored_at,
                }
                if record["ignored_reason"] is None:
                    patch = repository.get_commit_patch(
                        hg, commit["node"].encode("ascii")
                    )
                    record.update(score_patch(model, patch))
                    stats["scored"] += 1
                else:
                    stats["ignored"] += 1
                yield record


def load_model(model_dir: str | None = None) -> Any:
    """Load the perf regression predictor from ``model_dir``.

    Without a directory, the published checkpoint is downloaded into the
    model's default directory under the current directory, unless an
    up-to-date copy is already there.
    """
    from bugbug import utils
    from bugbug.models.perf_regression_predictor import (
        MODEL_IDENTIFIER,
        PerfRegressionPredictorModel,
    )

    if model_dir is None:
        model_dir = utils.download_model(MODEL_IDENTIFIER)
    return PerfRegressionPredictorModel.load(model_dir)


def get_model_version(model: Any) -> str:
    """The version the model reports, recorded with every score it produces."""
    version = model.get_extra_data().get("model_version")
    if not version:
        raise ValueError("The model reports no version")
    return str(version)
