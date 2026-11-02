# -*- coding: utf-8 -*-
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

from datetime import datetime, timedelta, timezone

import pytest

from bugbug.perf import commit_areas, test_categories
from bugbug.perf import simulation as sim

START = datetime(2026, 1, 5, tzinfo=timezone.utc)
PLATFORM = "test-linux2404-64-shippable/opt-"
AMAZON = PLATFORM + "browsertime-tp6-firefox-amazon"
SPEEDOMETER = PLATFORM + "browsertime-benchmark-firefox-speedometer3"
SPEEDOMETER_MOBILE = (
    "test-android-hw-a55-14-0-aarch64-shippable/opt-"
    "browsertime-benchmark-speedometer3-mobile-fenix-nofis"
)
PDFPAINT = PLATFORM + "talos-pdfpaint-3-swr"
CHROME = PLATFORM + "talos-chrome"
AWSY = PLATFORM + "awsy-base"
SW = "perftest-linux-service-worker-linux2404-64-shippable/opt"
STARTUP = (
    "perftest-android-hw-a55-aarch64-shippable-startup-fenix-cold-main-first-frame"
)
UNKNOWN = PLATFORM + "browsertime-tp6-firefox-nytimes"
SUITE = frozenset(
    {AMAZON, SPEEDOMETER, SPEEDOMETER_MOBILE, PDFPAINT, CHROME, AWSY, SW, UNKNOWN}
)

RECORDS = [
    {
        "framework": "browsertime",
        "harness": "raptor",
        "test": "amazon",
        "suite_aliases": ["amazon"],
        "areas": ["js", "dom", "layout", "network"],
        "scenario": "pageload",
        "apps": ["firefox", "fenix"],
    },
    {
        "framework": "browsertime",
        "harness": "raptor",
        "test": "speedometer3",
        "suite_aliases": ["speedometer3"],
        "areas": ["js", "dom", "layout", "graphics"],
        "scenario": "benchmark",
        "apps": ["firefox", "fenix"],
    },
    {
        "framework": "browsertime",
        "harness": "raptor",
        "test": "facebook-nav",
        "suite_aliases": ["facebook-nav"],
        "areas": ["js", "dom"],
        "scenario": "interaction",
        "apps": ["firefox"],
    },
    {
        "framework": "talos",
        "harness": "talos",
        "test": "pdfpaint",
        "suite_aliases": ["pdfpaint"],
        "areas": ["pdf", "graphics"],
        "scenario": "pageload",
        "talos_suites": ["pdfpaint"],
    },
    {
        "framework": "talos",
        "harness": "talos",
        "test": "tresize",
        "suite_aliases": ["tresize"],
        "areas": ["frontend"],
        "scenario": "interaction",
        "talos_suites": ["chromez"],
    },
    {
        "framework": "devtools",
        "harness": "talos",
        "test": "damp",
        "suite_aliases": ["damp"],
        "areas": ["devtools"],
        "scenario": "interaction",
        "talos_suites": ["damp", "damp-inspector"],
    },
    {
        "framework": "awsy",
        "harness": "awsy",
        "test": "base",
        "suite_aliases": ["base"],
        "areas": ["memory"],
        "scenario": "startup",
    },
    {
        "framework": "mozperftest",
        "harness": "mozperftest",
        "test": "Service Worker Fetch",
        "suite_aliases": ["Service Worker Fetch", "test_fetch.html", "test_fetch"],
        "areas": ["dom", "network"],
        "scenario": "benchmark",
        "perftest_tasks": ["service-worker"],
    },
    {
        "framework": "mozperftest",
        "harness": "mozperftest",
        "test": "applink-startup-first-frame",
        "suite_aliases": ["applink-startup-first-frame", "cmff.sh", "cmff"],
        "areas": ["frontend"],
        "scenario": "startup",
        "perftest_tasks": ["startup-fenix-cold-main-first-frame"],
    },
]


def _lookup(catalog, framework, suite):
    record = catalog.lookup(framework, suite)
    assert record is not None
    return record


def test_catalog_lookup_by_suite() -> None:
    catalog = test_categories.TestCatalog(RECORDS)
    assert _lookup(catalog, "browsertime", "Amazon")["test"] == "amazon"
    assert (
        _lookup(catalog, "browsertime", "facebook-nav.friends")["test"]
        == "facebook-nav"
    )
    assert _lookup(catalog, "devtools", "damp")["areas"] == ["devtools"]
    assert _lookup(catalog, "awsy", "Base Content Explicit")["test"] == "base"
    assert catalog.lookup("awsy", "Images") is None
    assert _lookup(catalog, "mozperftest", "test_fetch.html")["areas"] == [
        "dom",
        "network",
    ]
    assert catalog.lookup("browsertime", "speedometer") is None


def test_catalog_tests_of_label() -> None:
    catalog = test_categories.TestCatalog(RECORDS)
    names = lambda label: [t["test"] for t in catalog.tests_of_label(label)]  # noqa: E731
    assert names(AMAZON) == ["amazon"]
    assert names(SPEEDOMETER) == ["speedometer3"]
    assert names(SPEEDOMETER_MOBILE) == ["speedometer3"]
    assert names(PDFPAINT) == ["pdfpaint"]
    assert names(CHROME) == ["tresize"]
    assert names(PLATFORM + "talos-damp-inspector") == ["damp"]
    assert names(AWSY) == ["base"]
    assert names(SW) == ["Service Worker Fetch"]
    assert names(STARTUP) == ["applink-startup-first-frame"]
    assert names(UNKNOWN) == []
    assert catalog.areas_of_label(UNKNOWN) is None
    assert catalog.areas_of_label(PDFPAINT) == {"pdf", "graphics"}


def test_parse_script_metadata() -> None:
    js = """
    const perfMetadata = {
      owner: "Team",
      areas: ["storage", "frontend"],
      scenario: "interaction",
      name: "browser_thing_perf.js",
      options: { default: { perfherder_metrics: [{ name: "x", unit: "ms" }] } },
    };
    """
    assert test_categories.parse_script_metadata(js) == {
        "areas": ["storage", "frontend"],
        "scenario": "interaction",
        "name": "browser_thing_perf.js",
    }
    shell = "#!/bin/bash\n#name: homeview-startup\n# Areas: frontend, js\n#scenario: startup\n"
    assert test_categories.parse_script_metadata(shell) == {
        "name": "homeview-startup",
        "areas": ["frontend", "js"],
        "scenario": "startup",
    }
    assert (
        test_categories.perftest_task_name(
            "hw-a55-aarch64-shippable-startup-fenix-cold-main-first-frame"
        )
        == "startup-fenix-cold-main-first-frame"
    )


def test_component_mapping_and_area_index() -> None:
    mapping = commit_areas.ComponentMapping(
        {
            "dom/base/Document.cpp": "Core::DOM: Core & HTML",
            "dom/base/Element.cpp": "Core::DOM: Core & HTML",
            "dom/events/Event.cpp": "Core::DOM: Events",
            "js/src/vm/Interpreter.cpp": "Core::JavaScript Engine",
            "moz.configure": "Firefox Build System::General",
        }
    )
    assert mapping.component_of("dom/base/Element.cpp") == "Core::DOM: Core & HTML"
    # A file the mapping no longer lists takes its directory's usual component.
    assert mapping.component_of("dom/base/NewFile.cpp") == "Core::DOM: Core & HTML"
    assert mapping.component_of("dom/other/Thing.cpp") == "Core::DOM: Core & HTML"
    assert mapping.component_of("unknown/path.rs") is None

    index = commit_areas.AreaIndex(
        {
            "dom": ["Core::DOM: *"],
            "js": ["Core::JavaScript Engine", "Core::JavaScript: GC"],
            "memory": ["Core::JavaScript: GC"],
        }
    )
    assert index.areas_of("Core::DOM: Events") == {"dom"}
    assert index.areas_of("Core::JavaScript: GC") == {"js", "memory"}
    assert index.areas_of("Firefox Build System::General") == frozenset()

    described = commit_areas.describe_commit(
        ["dom/events/Event.cpp", "moz.configure", "nowhere/x.c"], mapping, index
    )
    assert described == {
        "components": ["Core::DOM: Events", "Firefox Build System::General"],
        "areas": ["dom"],
        "unmapped_components": ["Firefox Build System::General"],
        "unknown_files": 1,
    }


@pytest.fixture
def small_suite(monkeypatch):
    """Let a handful of labels count as the whole suite."""
    monkeypatch.setattr(sim.timeline, "FULL_SUITE_MIN_LABELS", len(SUITE))


def _timeline(count, areas_by_push, full=(0,), regressions=()):
    pushes = []
    for i in range(count):
        pushes.append(
            sim.Push(
                index=i,
                push_id=1000 + i,
                pushlog_id=100 + i,
                time=START + timedelta(hours=i),
                commits=(sim.Commit(f"{i:040x}"),),
                dontbuild=False,
                ran_labels=SUITE if i in full else frozenset(),
            )
        )
    regs = [
        sim.Regression(summary_id, index, pushes[index].time, frozenset(labels))
        for summary_id, index, labels in regressions
    ]
    info = {
        f"{i:040x}": {"areas": areas, "unmapped_components": unmapped, "components": []}
        for i, (areas, unmapped) in areas_by_push.items()
    }
    return sim.Timeline(pushes, regs), info


def _selector(timeline, info, unmapped="none"):
    catalog = test_categories.TestCatalog(RECORDS)
    label_areas, label_apps = sim.build_label_tables(timeline, catalog)
    return sim.AreaSelector(label_areas, label_apps, info, unmapped=unmapped)


def test_area_selector() -> None:
    timeline, info = _timeline(
        4,
        {
            0: (["pdf"], []),
            1: ([], ["Firefox Build System::General"]),
            2: ([], ["Firefox for Android::Toolbar"]),
            3: ([], ["Web Compatibility::Interventions"]),
        },
    )
    selector = _selector(timeline, info)
    # The unknown label is always selected; pdf selects pdfpaint.
    assert sim.selection.union(
        selector.commit_selections(timeline.pushes[0], SUITE)
    ) == {
        PDFPAINT,
        UNKNOWN,
    }
    assert selector.commit_selections(timeline.pushes[1], SUITE) == [sim.ALL]
    assert sim.selection.union(
        selector.commit_selections(timeline.pushes[2], SUITE)
    ) == {
        AMAZON,
        SPEEDOMETER,
        SPEEDOMETER_MOBILE,
        UNKNOWN,
    }
    assert selector.commit_selections(timeline.pushes[3], SUITE) == [frozenset()]
    assert _selector(timeline, info, unmapped="all").commit_selections(
        timeline.pushes[3], SUITE
    ) == [sim.ALL]


def test_selected_batch_runs_the_union_since_the_last_batch(small_suite) -> None:
    timeline, info = _timeline(
        7,
        {i: (["pdf"] if i in (1, 2) else ["memory"], []) for i in range(7)},
        regressions=((1, 1, [PDFPAINT]), (2, 3, [AMAZON])),
    )
    strategy = sim.SelectedBatch(
        full_hours=100, cadence="fixed_interval:push_interval=3"
    )
    strategy.bind(_selector(timeline, info), timeline, sim.FixedInterval(3))
    schedule = sim.replay(strategy, timeline)
    # Push 0 cold-starts a full run; the batch at 3 runs what pushes 1-3
    # selected, the one at 6 what pushes 4-6 selected.
    assert schedule.decisions[0] == sim.ALL
    assert schedule.decisions[3] == {PDFPAINT, AWSY, UNKNOWN}
    assert schedule.decisions[6] == {AWSY, UNKNOWN}
    detections = {d.regression.summary_id: d for d in sim.detect(schedule, timeline)}
    assert detections[1].detected_index == 3
    # The amazon regression is only caught by a full run, none follows.
    assert detections[2].detected_index is None


def test_label_risk_batch_counts_selecting_commits(small_suite) -> None:
    timeline, info = _timeline(
        7, {i: (["pdf"] if i in (1, 2, 4) else ["memory"], []) for i in range(7)}
    )
    strategy = sim.LabelRiskBatch(
        threshold=2, full_hours=100, cadence="fixed_interval:push_interval=3"
    )
    strategy.bind(_selector(timeline, info), timeline, sim.FixedInterval(3))
    schedule = sim.replay(strategy, timeline)
    # By push 3 pdfpaint was selected twice (pushes 1 and 2) and awsy once;
    # awsy's count carries over and reaches 2 by push 6, while pdfpaint's
    # single selection at push 4 waits. The unknown label is always selected.
    assert schedule.decisions[3] == {PDFPAINT, UNKNOWN}
    assert schedule.decisions[6] == {AWSY, UNKNOWN}
    assert strategy.counters[PDFPAINT] == 1
    assert AWSY not in strategy.counters


def test_history_selector_learns_settled_regressions() -> None:
    timeline, _ = _timeline(3, {})
    info = {
        "a" * 40: {
            "components": ["Firefox::PDF Viewer"],
            "areas": ["pdf"],
            "unmapped_components": [],
        },
        "b" * 40: {
            "components": ["Core::Layout"],
            "areas": ["layout"],
            "unmapped_components": [],
        },
    }
    regressions = [
        {
            "outcome": "concluded",
            "culprit_commits": ["a" * 40],
            "push_time": (START - timedelta(days=30)).isoformat(),
            "runnables": [{"label": PLATFORM + "talos-pdfpaint-1"}],
        },
        {
            "outcome": "concluded",
            "culprit_commits": ["b" * 40],
            "push_time": (START - timedelta(days=1)).isoformat(),
            "runnables": [{"label": AMAZON}],
        },
        {
            "outcome": "rejected",
            "culprit_commits": ["a" * 40],
            "push_time": "2025-01-01T00:00:00Z",
            "runnables": [{"label": AMAZON}],
        },
    ]
    catalog = test_categories.TestCatalog(RECORDS)
    selector = sim.HistorySelector(
        regressions, info, identity=sim.selection.catalog_identity(catalog)
    )
    push = sim.Push(
        0, 1, 100, START, (sim.Commit("a" * 40), sim.Commit("b" * 40)), False, SUITE
    )
    # The pdf regression settled; the layout one is too recent to be known.
    assert selector.commit_selections(push, SUITE) == [
        frozenset({PDFPAINT}),
        frozenset(),
    ]
    later = sim.Push(
        1, 2, 101, START + timedelta(days=20), (sim.Commit("b" * 40),), False, SUITE
    )
    assert selector.commit_selections(later, SUITE) == [frozenset({AMAZON})]


def test_parse_selection_strategy() -> None:
    strategy = sim.parse_strategy(
        "label_risk_batch:threshold=3,full_hours=48,cadence=fixed_interval:push_interval=5"
    )
    assert isinstance(strategy, sim.LabelRiskBatch)
    assert strategy.params["cadence"] == "fixed_interval:push_interval=5"
    assert strategy.params["threshold"] == 3
