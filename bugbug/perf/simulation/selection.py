# -*- coding: utf-8 -*-
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

"""Scheduling strategies that run only the tests the landed commits can regress.

A selector names, for each push, the labels its commits can regress:
:class:`AreaSelector` from the areas the tests declare in the tree and the
Bugzilla components of the files the commit touched, :class:`HistorySelector`
from the tests that commits in the same components regressed before, or
both. Two strategies use it. :class:`SelectedBatch` keeps a batching rule's
batches and runs in each the union of what the pushes since the last batch
selected. :class:`LabelRiskBatch` keeps one counter per label, raised by
every commit that selects it, and runs the label at the next batch once the
counter reaches a threshold. Both still run the whole suite every
``full_hours`` hours, so a regression the selector misses waits at most for
the next full run.
"""

import collections
import logging
from datetime import timedelta
from typing import Any, Callable, ClassVar, Iterable, Mapping, Protocol

from bugbug.perf.label_policy import POSITIVE_OUTCOMES
from bugbug.perf.simulation.strategies import ALL, Decision, Strategy
from bugbug.perf.simulation.timeline import Push, Timeline
from bugbug.perf.task_labels import split_task_label

logger = logging.getLogger(__name__)

MOBILE_APPS = frozenset(("fenix", "geckoview", "focus", "refbrow"))
MOBILE_COMPONENT_PREFIXES = ("Firefox for Android::", "GeckoView::", "Focus::")
# Components whose changes can move any test: the build, the test harnesses,
# the low-level runtime and the protocol browsertime drives the browser with.
CROSS_CUTTING_PREFIXES = (
    "Firefox Build System::",
    "Testing::",
    "Core::General",
    "Core::MFBT",
    "Core::mozglue",
    "Remote Protocol::",
    "Release Engineering::",
)
# A regression is known to the history selector this long after its culprit
# landed: the time sheriffs and the pipeline take to confirm and pin it.
HISTORY_SETTLE_DAYS = 14

Selection = frozenset[str] | str


class Selector(Protocol):
    def commit_selections(self, push: Push, suite: frozenset[str]) -> list[Selection]:
        """What each scored commit of the push selects out of ``suite``: labels or ``ALL``."""


def union(selections: Iterable[Selection]) -> Selection:
    selected: set[str] = set()
    for labels in selections:
        if labels == ALL:
            return ALL
        selected.update(labels)
    return frozenset(selected)


def label_identity(label: str) -> str:
    """The platform-independent test name of a label, the label itself if unparsed."""
    parts = split_task_label(label)
    return parts[2] if parts is not None else label


def catalog_identity(catalog: Any) -> Callable[[str], str]:
    """An identity from the tests a label runs, so chunks and variants of a test match.

    Falls back to :func:`label_identity` for labels the catalog does not know.
    """

    def identity(label: str) -> str:
        tests = catalog.tests_of_label(label)
        if not tests:
            return label_identity(label)
        return "|".join(sorted(f"{test['framework']}/{test['test']}" for test in tests))

    return identity


class AreaSelector:
    """Select the labels whose tests share an area with what a commit touched.

    ``label_areas`` maps each suite label to the areas of the tests it runs,
    None for a label whose tests are unknown, which is always selected.
    ``commit_info`` maps a node to its ``perf_commit_areas.json`` record. A
    commit without any mapped area selects every label (``unmapped``
    ``all``) or none (``none``); with ``mobile``, a commit in a mobile
    component selects the labels of mobile applications; with
    ``cross_cutting``, a commit in a cross-cutting component selects all.
    """

    def __init__(
        self,
        label_areas: Mapping[str, frozenset[str] | None],
        label_apps: Mapping[str, frozenset[str]],
        commit_info: Mapping[str, Mapping[str, Any]],
        unmapped: str = "all",
        mobile: bool = True,
        cross_cutting: bool = True,
    ) -> None:
        self.label_areas = label_areas
        self.label_apps = label_apps
        self.commit_info = commit_info
        self.unmapped = unmapped
        self.mobile = mobile
        self.cross_cutting = cross_cutting
        self._cache: dict[tuple[str, frozenset[str]], Selection] = {}

    def commit_labels(self, node: str, suite: frozenset[str]) -> Selection:
        key = (node, suite)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        info = self.commit_info.get(node)
        if info is None:
            result: Selection = ALL if self.unmapped == "all" else frozenset()
        else:
            result = self._select(info, suite)
        self._cache[key] = result
        return result

    def _select(self, info: Mapping[str, Any], suite: frozenset[str]) -> Selection:
        areas = frozenset(info.get("areas", ()))
        unmapped = info.get("unmapped_components", ())
        if self.cross_cutting and any(
            c.startswith(CROSS_CUTTING_PREFIXES) for c in unmapped
        ):
            return ALL
        mobile = self.mobile and any(
            c.startswith(MOBILE_COMPONENT_PREFIXES) for c in unmapped
        )
        if not areas and not mobile:
            return ALL if self.unmapped == "all" else frozenset()
        selected = set()
        for label in suite:
            label_areas = self.label_areas.get(label)
            if (
                label_areas is None
                or (label_areas & areas)
                or mobile
                and (self.label_apps.get(label, frozenset()) & MOBILE_APPS)
            ):
                selected.add(label)
        return frozenset(selected)

    def commit_selections(self, push: Push, suite: frozenset[str]) -> list[Selection]:
        return [
            self.commit_labels(commit.node, suite)
            for commit in push.commits
            if commit.scored
        ]


class HistorySelector:
    """Select the tests that commits in the same Bugzilla components regressed before.

    Built from the regression records: every positive regression with pinned
    culprits links the culprits' components to the test identities of its
    labels, and becomes known ``settle_days`` after its culprit push. A
    commit selects, out of the suite, every label whose identity any of its
    components is linked to. Commits without components select nothing.
    """

    def __init__(
        self,
        regressions: Iterable[Mapping[str, Any]],
        commit_info: Mapping[str, Mapping[str, Any]],
        settle_days: int = HISTORY_SETTLE_DAYS,
        identity: Callable[[str], str] = label_identity,
    ) -> None:
        self.commit_info = commit_info
        self.identity = identity
        self.events: list[tuple[Any, frozenset[str], frozenset[str]]] = []
        for record in regressions:
            if (
                record["outcome"] not in POSITIVE_OUTCOMES
                or not record["culprit_commits"]
            ):
                continue
            identities = frozenset(
                identity(runnable["label"])
                for runnable in record["runnables"]
                if runnable.get("label")
            )
            components = frozenset(
                component
                for node in record["culprit_commits"]
                for component in commit_info.get(node, {}).get("components", ())
            )
            if identities and components:
                self.events.append((record["push_time"], components, identities))
        self.events.sort(key=lambda event: event[0])
        self.settle = timedelta(days=settle_days)
        self.known: dict[str, set[str]] = collections.defaultdict(set)
        self._next = 0
        self._suite_identities: dict[frozenset[str], dict[str, list[str]]] = {}

    def _advance(self, push: Push) -> None:
        from bugbug.utils import parse_timestamp

        cutoff = push.time - self.settle
        while self._next < len(self.events):
            time, components, identities = self.events[self._next]
            parsed = parse_timestamp(time)
            if parsed is None or parsed > cutoff:
                break
            for component in components:
                self.known[component].update(identities)
            self._next += 1

    def _by_identity(self, suite: frozenset[str]) -> dict[str, list[str]]:
        cached = self._suite_identities.get(suite)
        if cached is None:
            cached = collections.defaultdict(list)
            for label in suite:
                cached[self.identity(label)].append(label)
            self._suite_identities[suite] = cached
        return cached

    def commit_selections(self, push: Push, suite: frozenset[str]) -> list[Selection]:
        self._advance(push)
        by_identity = self._by_identity(suite)
        selections: list[Selection] = []
        for commit in push.commits:
            if not commit.scored:
                continue
            selected: set[str] = set()
            for component in self.commit_info.get(commit.node, {}).get(
                "components", ()
            ):
                for identity in self.known.get(component, ()):
                    selected.update(by_identity.get(identity, ()))
            selections.append(frozenset(selected))
        return selections


class UnionSelector:
    """Select what any of several selectors selects."""

    def __init__(self, *selectors: Selector) -> None:
        self.selectors = selectors

    def commit_selections(self, push: Push, suite: frozenset[str]) -> list[Selection]:
        per_selector = [
            selector.commit_selections(push, suite) for selector in self.selectors
        ]
        return [union(choices) for choices in zip(*per_selector)]


class SelectionStrategy(Strategy):
    """Base of the selection strategies: a batching rule plus a selector.

    ``cadence`` is the spec of the rule saying when a batch is due, kept in
    the parameters for the name; the rule itself, the selector and the
    timeline are passed by the caller as ``cadence_strategy``, ``selector``
    and ``history``, or attached with ``bind``.
    """

    def __init__(
        self,
        cadence: str = "production",
        selector: Selector | None = None,
        history: Timeline | None = None,
        cadence_strategy: Strategy | None = None,
        **params: Any,
    ) -> None:
        self.selector = selector
        self.timeline = history
        self.cadence = cadence_strategy
        super().__init__(cadence=cadence, **params)

    def bind(
        self, selector: Selector, timeline: Timeline, cadence: Strategy
    ) -> "SelectionStrategy":
        self.selector = selector
        self.timeline = timeline
        self.cadence = cadence
        self.reset()
        return self

    def reset(self) -> None:
        self.last_full: Push | None = None
        if getattr(self, "cadence", None) is not None:
            self.cadence.reset()  # type: ignore[union-attr]

    def full_due(self, push: Push) -> bool:
        return self.last_full is None or (
            (push.time - self.last_full.time).total_seconds()
            >= self.params["full_hours"] * 3600
        )

    def selections(self, push: Push) -> list[Selection]:
        assert self.selector is not None and self.timeline is not None
        return self.selector.commit_selections(push, self.timeline.suite_at(push.index))


class SelectedBatch(SelectionStrategy):
    """The batching rule's batches run the union of what was selected since the last batch."""

    family = "selected_batch"
    search_space: ClassVar[dict] = {"full_hours": (4.0, 168.0, True)}

    def __init__(self, full_hours: float = 24.0, **context: Any) -> None:
        super().__init__(full_hours=float(full_hours), **context)

    def reset(self) -> None:
        super().reset()
        self.pending: set[str] = set()
        self.pending_all = False

    def observe(self, push: Push) -> None:
        assert self.cadence is not None
        self.cadence.observe(push)
        labels = union(self.selections(push))
        if labels == ALL:
            self.pending_all = True
        else:
            self.pending.update(labels)

    def decide(self, push: Push) -> Decision:
        assert self.cadence is not None and self.timeline is not None
        due = self.cadence.decide(push)
        if due is None:
            return None
        if due != ALL:
            # The rule's own partial runs, such as production's expanded labels.
            return due
        if self.full_due(push) or self.pending_all:
            self.last_full = push
            self.pending.clear()
            self.pending_all = False
            return ALL
        selected = frozenset(self.pending & self.timeline.suite_at(push.index))
        self.pending.clear()
        return selected or None


class LabelRiskBatch(SelectionStrategy):
    """Each label runs at a batch once enough commits selecting it have landed.

    Every commit adds 1 to the counter of each label it selects, or
    ``all_weight`` to every label when it selects the whole suite; a batch
    of the rule runs the labels whose counter reached ``threshold`` and
    resets them, so a label few commits can affect runs rarely while one
    most commits touch runs at every batch. The whole suite runs instead
    every ``full_hours`` hours.
    """

    family = "label_risk_batch"
    search_space: ClassVar[dict] = {
        "threshold": (1, 12, False),
        "full_hours": (4.0, 168.0, True),
    }

    def __init__(
        self,
        threshold: int = 2,
        full_hours: float = 48.0,
        all_weight: float = 1.0,
        **context: Any,
    ) -> None:
        super().__init__(
            threshold=int(threshold),
            full_hours=float(full_hours),
            all_weight=float(all_weight),
            **context,
        )

    def reset(self) -> None:
        super().reset()
        self.counters: collections.Counter = collections.Counter()
        self.all_pending = 0.0

    def observe(self, push: Push) -> None:
        assert self.cadence is not None
        self.cadence.observe(push)
        for labels in self.selections(push):
            if labels == ALL:
                self.all_pending += self.params["all_weight"]
            else:
                self.counters.update(labels)

    def decide(self, push: Push) -> Decision:
        assert self.cadence is not None and self.timeline is not None
        due = self.cadence.decide(push)
        if due is None:
            return None
        if due != ALL:
            return due
        if self.full_due(push):
            self.last_full = push
            self.counters.clear()
            self.all_pending = 0.0
            return ALL
        suite = self.timeline.suite_at(push.index)
        threshold = self.params["threshold"]
        selected = set()
        for label in suite:
            if self.counters[label] + self.all_pending >= threshold:
                selected.add(label)
                del self.counters[label]
        if suite and len(selected) == len(suite):
            self.last_full = push
            self.counters.clear()
            self.all_pending = 0.0
            return ALL
        return frozenset(selected) or None


SELECTION_FAMILIES = {cls.family: cls for cls in (SelectedBatch, LabelRiskBatch)}


def build_label_tables(
    timeline: Timeline, catalog: Any
) -> tuple[dict[str, frozenset[str] | None], dict[str, frozenset[str]]]:
    """Areas and applications of every label that ever ran in the timeline."""
    label_areas: dict[str, frozenset[str] | None] = {}
    label_apps: dict[str, frozenset[str]] = {}
    labels = {label for push in timeline.pushes for label in push.ran_labels}
    unknown = 0
    for label in labels:
        tests = catalog.tests_of_label(label)
        if not tests:
            label_areas[label] = None
            unknown += 1
            continue
        label_areas[label] = frozenset(area for test in tests for area in test["areas"])
        label_apps[label] = frozenset(
            app for test in tests for app in test.get("apps", ())
        )
    logger.info(
        "Label tables: %d labels, %d without a known test", len(labels), unknown
    )
    return label_areas, label_apps


def build_selector(
    kind: str,
    timeline: Timeline,
    catalog: Any,
    commit_info: Mapping[str, Mapping[str, Any]],
    regressions: Iterable[Mapping[str, Any]] = (),
    unmapped: str = "all",
) -> Selector:
    """A selector of ``kind``: ``area``, ``history`` or ``both``."""
    selectors: list[Selector] = []
    if kind in ("area", "both"):
        label_areas, label_apps = build_label_tables(timeline, catalog)
        selectors.append(AreaSelector(label_areas, label_apps, commit_info, unmapped))
    if kind in ("history", "both"):
        selectors.append(
            HistorySelector(
                regressions, commit_info, identity=catalog_identity(catalog)
            )
        )
    if not selectors:
        raise ValueError(f"Unknown selector {kind!r}; one of area, history, both")
    return selectors[0] if len(selectors) == 1 else UnionSelector(*selectors)
