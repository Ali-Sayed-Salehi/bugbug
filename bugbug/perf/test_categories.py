# -*- coding: utf-8 -*-
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

"""What each performance test covers, read from the Firefox tree.

Every performance test declares the code areas it stresses and the scenario
it runs, with values from ``testing/performance/categories.yml`` (bug
2087893): Raptor tests in their manifests, Talos tests on their classes, AWSY
in ``TEST_CATEGORIES`` and mozperftest tests in ``perfMetadata`` or a shell
header. Until Perfherder stores them on the signature (bug 2087894), this
module reads them from a checkout into the ``perf_test_categories.json``
dataset, one record per test keyed by the Perfherder framework and suite, and
``TestCatalog`` answers what a suite or a task label covers.
"""

import ast
import json
import logging
import pathlib
import re
import tomllib
from typing import Any, Iterable, Iterator, Mapping

import yaml

from bugbug.perf.task_labels import split_task_label, split_test_name

logger = logging.getLogger(__name__)

CATEGORIES_PATH = "testing/performance/categories.yml"
RAPTOR_TESTS_DIR = "testing/raptor/raptor/tests"
TALOS_TESTS_PATH = "testing/talos/talos/test.py"
TALOS_SUITES_PATH = "testing/talos/talos.json"
AWSY_INIT_PATH = "testing/awsy/awsy/__init__.py"
PERFTEST_KINDS_DIR = "taskcluster/kinds/perftest"
MOBILE_SCRIPTS_DIRS = ("testing/performance",)
SKIPPED_DIR_MARKERS = ("obj-", "objdir-", "/.claude/", "/node_modules/")

# Talos tests whose Perfherder framework is not talos.
TALOS_FRAMEWORKS = {"damp": "devtools"}
# AWSY reports measurements as suites; the test is told from the suite name.
AWSY_BASE_SUITE_PREFIX = "base content"
# Flavours appended to a Raptor or Talos test in its task label.
LABEL_VARIANT_SUFFIXES = (
    "-native-profiling",
    "-profiling",
    "-no-nv",
    "-nofis",
    "-swr",
    "-nv",
)

# Talos task labels that name a talos.json suite differently.
TALOS_LABEL_SUITES = {"chrome": "chromez"}
# Leading tokens of a perftest task name that describe the hardware, which
# the pipeline strips from labels as the platform.
PERFTEST_TASK_PLATFORM_TOKEN_RE = re.compile(
    r"^(hw|emulator|aarch64|shippable|[aps]\d+|\d+h\d+|\d+)$"
)
# Suffixes a perftest task may have gained since the labels in history ran.
PERFTEST_TASK_SUFFIXES = ("-native", "-wasm")

JS_METADATA_RE = re.compile(r"(perfMetadata|module\.exports)\s*=\s*\{", re.MULTILINE)
JS_FIELD_RE = {
    "areas": re.compile(
        r"^\s*areas\s*:\s*(\[[^\]]*\]|\"[^\"]*\"|'[^']*')", re.MULTILINE
    ),
    "scenario": re.compile(r"^\s*scenario\s*:\s*[\"']([^\"']*)[\"']", re.MULTILINE),
    "name": re.compile(r"^\s*name\s*:\s*[\"']([^\"']*)[\"']", re.MULTILINE),
}
SHELL_FIELD_RE = re.compile(
    r"^#\s*(name|areas|scenario)\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE
)
TEST_PATH_RE = re.compile(r"((?:[\w.${}-]+/)+[\w.-]+\.(?:js|html|sh|py))(?![\w/])")


def split_names(value: Any) -> list[str]:
    """A declared list of areas, as a comma-separated string or a list, as names."""
    if value is None:
        return []
    if isinstance(value, str):
        value = value.split(",")
    return [str(name).strip() for name in value if str(name).strip()]


def load_vocabulary(firefox_dir: str) -> dict[str, Any]:
    """The areas (with their components) and scenarios of ``categories.yml``."""
    with open(pathlib.Path(firefox_dir, CATEGORIES_PATH), encoding="utf-8") as fh:
        content = yaml.safe_load(fh)
    return {
        "areas": {
            name: {
                "description": (entry or {}).get("description", ""),
                "components": list((entry or {}).get("components") or []),
            }
            for name, entry in (content.get("areas") or {}).items()
        },
        "scenarios": {
            name: (entry or {}).get("description", "")
            for name, entry in (content.get("scenarios") or {}).items()
        },
    }


def vocabulary_records(vocabulary: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """``perf_test_areas.json`` records: one per area and one per scenario."""
    for name, entry in vocabulary["areas"].items():
        yield {
            "kind": "area",
            "name": name,
            "description": entry["description"],
            "components": entry["components"],
        }
    for name, description in vocabulary["scenarios"].items():
        yield {"kind": "scenario", "name": name, "description": description}


def _relative(path: pathlib.Path, firefox_dir: str) -> str:
    return path.relative_to(firefox_dir).as_posix()


def _skipped(path: pathlib.Path) -> bool:
    text = path.as_posix()
    return any(marker in text for marker in SKIPPED_DIR_MARKERS)


def iter_raptor_tests(firefox_dir: str) -> Iterator[dict[str, Any]]:
    """One record per Raptor manifest entry; Perfherder reports them as browsertime."""
    root = pathlib.Path(firefox_dir, RAPTOR_TESTS_DIR)
    for manifest in sorted(root.rglob("*.toml")):
        with open(manifest, "rb") as fh:
            content = tomllib.load(fh)
        default = content.get("DEFAULT", {})
        for name, entry in content.items():
            if name == "DEFAULT":
                continue
            yield {
                "framework": "browsertime",
                "harness": "raptor",
                "test": name,
                "suite_aliases": [name],
                "areas": split_names(entry.get("areas", default.get("areas"))),
                "scenario": entry.get("scenario", default.get("scenario")),
                "path": _relative(manifest, firefox_dir),
                "group": manifest.parent.name,
                "apps": split_names(entry.get("apps", default.get("apps"))),
            }


def _talos_classes(source: str) -> dict[str, tuple[list[str], dict[str, Any], bool]]:
    classes: dict[str, tuple[list[str], dict[str, Any], bool]] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [base.id for base in node.bases if isinstance(base, ast.Name)]
        attrs: dict[str, Any] = {}
        for statement in node.body:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id in ("areas", "scenario")
            ):
                try:
                    attrs[statement.targets[0].id] = ast.literal_eval(statement.value)
                except ValueError:
                    attrs[statement.targets[0].id] = None
        registered = any(
            isinstance(decorator, ast.Call)
            and getattr(decorator.func, "id", "") == "register_test"
            for decorator in node.decorator_list
        )
        classes[node.name] = (bases, attrs, registered)
    return classes


def iter_talos_tests(firefox_dir: str) -> Iterator[dict[str, Any]]:
    """One record per registered Talos test class, with the talos.json suites running it."""
    path = pathlib.Path(firefox_dir, TALOS_TESTS_PATH)
    classes = _talos_classes(path.read_text(encoding="utf-8"))
    with open(pathlib.Path(firefox_dir, TALOS_SUITES_PATH), encoding="utf-8") as fh:
        suites = json.load(fh)["suites"]
    suites_of_test: dict[str, list[str]] = {}
    for suite, entry in suites.items():
        for test in entry.get("tests", []):
            suites_of_test.setdefault(test, []).append(suite)

    def resolve(name: str, key: str) -> Any:
        bases, attrs, _ = classes[name]
        if attrs.get(key) is not None:
            return attrs[key]
        for base in bases:
            if base in classes:
                value = resolve(base, key)
                if value is not None:
                    return value
        return None

    for name, (_, _, registered) in classes.items():
        if not registered:
            continue
        yield {
            "framework": TALOS_FRAMEWORKS.get(name, "talos"),
            "harness": "talos",
            "test": name,
            "suite_aliases": [name],
            "areas": split_names(resolve(name, "areas")),
            "scenario": resolve(name, "scenario"),
            "path": TALOS_TESTS_PATH,
            "talos_suites": sorted(suites_of_test.get(name, [])),
        }


def iter_awsy_tests(firefox_dir: str) -> Iterator[dict[str, Any]]:
    path = pathlib.Path(firefox_dir, AWSY_INIT_PATH)
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and getattr(node.targets[0], "id", "") == "TEST_CATEGORIES"
        ):
            for name, entry in ast.literal_eval(node.value).items():
                yield {
                    "framework": "awsy",
                    "harness": "awsy",
                    "test": name,
                    "suite_aliases": [name],
                    "areas": split_names(entry.get("areas")),
                    "scenario": entry.get("scenario"),
                    "path": AWSY_INIT_PATH,
                }


def parse_script_metadata(text: str) -> dict[str, Any]:
    """``name``, ``areas`` and ``scenario`` of a mozperftest script.

    JavaScript and HTML tests declare them in ``perfMetadata`` (or
    ``module.exports``); shell tests in ``# name:`` style header comments.
    """
    found: dict[str, Any] = {}
    match = JS_METADATA_RE.search(text)
    if match is not None:
        body = text[match.end() :]
        for key, pattern in JS_FIELD_RE.items():
            field = pattern.search(body)
            if field is None:
                continue
            value = field.group(1)
            if key == "areas":
                if value.startswith("["):
                    value = [v.strip().strip("\"'") for v in value[1:-1].split(",")]
                else:
                    value = value.strip("\"'")
                found[key] = split_names(value)
            else:
                found[key] = value
        return found
    for key, value in SHELL_FIELD_RE.findall(text):
        key = key.lower()
        found[key] = split_names(value) if key == "areas" else value.strip()
    return found


def perftest_task_name(name: str) -> str:
    """A perftest task name without its leading hardware tokens, as labels carry it."""
    tokens = name.split("-")
    while len(tokens) > 1 and PERFTEST_TASK_PLATFORM_TOKEN_RE.match(tokens[0]):
        tokens.pop(0)
    return "-".join(tokens)


def _perftest_task_tests(
    firefox_dir: str, known_paths: Iterable[str]
) -> dict[str, set[str]]:
    """Test paths run by each perftest task, from the task graph kind files.

    Commands name tests by their tree path, sometimes under a fetches
    directory, so a path counts when a known test path ends it.
    """
    known = sorted(known_paths, key=len, reverse=True)
    tasks: dict[str, set[str]] = {}
    root = pathlib.Path(firefox_dir, PERFTEST_KINDS_DIR)
    for kind in sorted(root.glob("*.yml")):
        if kind.name == "kind.yml":
            continue
        with open(kind, encoding="utf-8") as fh:
            content = yaml.safe_load(fh) or {}
        for name, definition in content.items():
            if not isinstance(definition, dict) or "defaults" in name:
                continue
            command = (definition.get("run") or {}).get("command", "")
            if isinstance(command, list):
                command = " ".join(str(part) for part in command)
            for candidate in TEST_PATH_RE.findall(str(command)):
                for path in known:
                    if candidate == path or candidate.endswith("/" + path):
                        tasks.setdefault(perftest_task_name(name), set()).add(path)
                        break
    return tasks


def iter_mozperftest_tests(firefox_dir: str) -> Iterator[dict[str, Any]]:
    """One record per mozperftest script: manifest entries plus the mobile shell tests."""
    root = pathlib.Path(firefox_dir)
    paths: set[pathlib.Path] = set()
    for manifest in root.rglob("perftest.toml"):
        if _skipped(manifest):
            continue
        with open(manifest, "rb") as fh:
            content = tomllib.load(fh)
        for name in content:
            if name != "DEFAULT":
                paths.add(manifest.parent / name)
    for scripts_dir in MOBILE_SCRIPTS_DIRS:
        for script in pathlib.Path(firefox_dir, scripts_dir).rglob("*.sh"):
            if not _skipped(script):
                paths.add(script)

    task_tests = _perftest_task_tests(
        firefox_dir, (_relative(path, firefox_dir) for path in paths)
    )
    tasks_of_path: dict[str, list[str]] = {}
    for task, task_paths in task_tests.items():
        for test_path in task_paths:
            tasks_of_path.setdefault(test_path, []).append(task)

    for path in sorted(paths):
        if not path.is_file():
            logger.warning("Missing mozperftest script %s", path)
            continue
        metadata = parse_script_metadata(
            path.read_text(encoding="utf-8", errors="replace")
        )
        if path.suffix == ".sh" and "name" not in metadata:
            continue
        relative = _relative(path, firefox_dir)
        name = metadata.get("name") or path.name
        aliases = {name, path.name, path.stem}
        yield {
            "framework": "mozperftest",
            "harness": "mozperftest",
            "test": name,
            "suite_aliases": sorted(aliases),
            "areas": metadata.get("areas", []),
            "scenario": metadata.get("scenario"),
            "path": relative,
            "perftest_tasks": sorted(tasks_of_path.get(relative, [])),
        }


def generate_test_categories(firefox_dir: str) -> Iterator[dict[str, Any]]:
    """Every test's record, merged by framework and test name.

    A Raptor test defined in several manifests (desktop and mobile) is one
    record whose areas are the union of its declarations.
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for record in (
        *iter_raptor_tests(firefox_dir),
        *iter_talos_tests(firefox_dir),
        *iter_awsy_tests(firefox_dir),
        *iter_mozperftest_tests(firefox_dir),
    ):
        key = (record["framework"], record["test"])
        existing = merged.get(key)
        if existing is None:
            record = dict(record)
            record["paths"] = [record.pop("path")]
            for field in ("apps", "suite_aliases", "talos_suites", "perftest_tasks"):
                if field in record:
                    record[field] = sorted(set(record[field]))
            record.pop("group", None)
            merged[key] = record
            continue
        if record["path"] not in existing["paths"]:
            existing["paths"].append(record["path"])
        if set(record["areas"]) != set(existing["areas"]):
            logger.warning(
                "%s/%s declares different areas in %s: %s vs %s",
                *key,
                record["path"],
                record["areas"],
                existing["areas"],
            )
        existing["areas"] = sorted(set(existing["areas"]) | set(record["areas"]))
        if existing.get("scenario") is None:
            existing["scenario"] = record.get("scenario")
        for field in ("apps", "suite_aliases"):
            if field in record:
                existing[field] = sorted(
                    set(existing.get(field, [])) | set(record[field])
                )
    for key in sorted(merged):
        yield merged[key]


def _normalize(name: str) -> str:
    return name.strip().lower()


class TestCatalog:
    """What the tests cover, looked up by Perfherder suite or by task label."""

    def __init__(self, records: Iterable[dict[str, Any]]) -> None:
        self.records = list(records)
        self.by_alias: dict[tuple[str, str], dict[str, Any]] = {}
        self.talos_suites: dict[str, list[dict[str, Any]]] = {}
        self.perftest_tasks: dict[str, list[dict[str, Any]]] = {}
        self.raptor_tests: dict[str, dict[str, Any]] = {}
        for record in self.records:
            framework = record["framework"]
            for alias in record.get("suite_aliases", [record["test"]]):
                self.by_alias.setdefault((framework, _normalize(alias)), record)
            if record["harness"] == "raptor":
                self.raptor_tests[_normalize(record["test"])] = record
            for suite in record.get("talos_suites", []):
                self.talos_suites.setdefault(suite, []).append(record)
            for task in record.get("perftest_tasks", []):
                self.perftest_tasks.setdefault(task, []).append(record)
        self._label_cache: dict[str, tuple[dict[str, Any], ...]] = {}

    def lookup(self, framework: str, suite: str) -> dict[str, Any] | None:
        """The record of a Perfherder ``(framework, suite)``, or None when unknown."""
        if framework == "awsy":
            test = (
                "base"
                if _normalize(suite).startswith(AWSY_BASE_SUITE_PREFIX)
                else "tp6"
            )
            return self.by_alias.get(("awsy", test))
        if framework == "devtools":
            framework_record = self.by_alias.get(("devtools", _normalize(suite)))
            return framework_record or self.by_alias.get(("talos", _normalize(suite)))
        record = self.by_alias.get((framework, _normalize(suite)))
        if record is None and framework == "browsertime" and "." in suite:
            # Navigation benchmarks report each page as ``test.page``.
            record = self.by_alias.get((framework, _normalize(suite.split(".")[0])))
        return record

    def _raptor_candidates(
        self, subtest: str | None, family: str
    ) -> dict[str, Any] | None:
        if subtest:
            name = _normalize(subtest)
            while True:
                if name in self.raptor_tests:
                    return self.raptor_tests[name]
                stripped = name
                for suffix in LABEL_VARIANT_SUFFIXES:
                    if stripped.endswith(suffix):
                        stripped = stripped[: -len(suffix)]
                        break
                if stripped == name:
                    break
                name = stripped
        # The family is ``<harness>-<kind entry>``; a test name may sit in the
        # kind entry (``benchmark-speedometer3-mobile``), never the harness.
        tokens = _normalize(family).split("-")[1:]
        for size in range(len(tokens), 0, -1):
            for start in range(0, len(tokens) - size + 1):
                candidate = "-".join(tokens[start : start + size])
                if candidate in self.raptor_tests:
                    return self.raptor_tests[candidate]
        return None

    def tests_of_label(self, label: str) -> tuple[dict[str, Any], ...]:
        """The tests a perf task label reports; empty when the label is unknown."""
        cached = self._label_cache.get(label)
        if cached is not None:
            return cached
        parts = split_task_label(label)
        found: tuple[dict[str, Any], ...] = ()
        if parts is not None:
            test_name = parts[2]
            grouping = split_test_name(test_name)
            harness = test_name.split("-")[0]
            if harness in ("browsertime", "raptor"):
                record = self._raptor_candidates(
                    grouping["subtest"], grouping["family"] or ""
                )
                found = (record,) if record else ()
            elif harness == "talos":
                suite = re.sub(
                    r"-(native-)?profiling$|-swr$|-nv$|-no-nv$",
                    "",
                    test_name[len("talos-") :],
                )
                suite = re.sub(r"-\d+$", "", suite)
                suite = TALOS_LABEL_SUITES.get(suite, suite)
                found = tuple(self.talos_suites.get(suite, ()))
            elif harness == "awsy":
                record = self.by_alias.get(("awsy", test_name[len("awsy-") :]))
                found = (record,) if record else ()
            elif harness == "perftest":
                found = self._perftest_tests(test_name)
        self._label_cache[label] = found
        return found

    def _perftest_tests(self, test_name: str) -> tuple[dict[str, Any], ...]:
        """The tests of a perftest task named by a label, tolerating renamed variants."""
        candidates = [test_name[len("perftest-") :], test_name]
        for candidate in list(candidates):
            candidates.extend(candidate + suffix for suffix in PERFTEST_TASK_SUFFIXES)
        found: list[dict[str, Any]] = []
        for candidate in candidates:
            for record in self.perftest_tasks.get(candidate, ()):
                if record not in found:
                    found.append(record)
            if found and candidate in candidates[:2]:
                break
        return tuple(found)

    def areas_of_label(self, label: str) -> frozenset[str] | None:
        """The union of the areas of the tests a label runs; None when unknown."""
        tests = self.tests_of_label(label)
        if not tests:
            return None
        return frozenset(area for test in tests for area in test["areas"])

    def scenarios_of_label(self, label: str) -> frozenset[str]:
        return frozenset(
            test["scenario"]
            for test in self.tests_of_label(label)
            if test.get("scenario")
        )
