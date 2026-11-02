# -*- coding: utf-8 -*-
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

"""The code areas a commit touches, through Bugzilla components.

Every performance test declares the code areas it stresses, and each area
lists in ``testing/performance/categories.yml`` the Bugzilla components whose
changes are most likely to move its tests (see ``test_categories``). The
tree's ``moz.build`` files assign every file a component, published as
``components.json`` by the ``source-bugzilla-info`` task. A commit's areas are
the areas of the components of the files it touched, which is what lets a
change be matched to the tests it can regress.
"""

import collections
import json
import logging
from typing import Any, Iterable, Mapping

from bugbug import utils

logger = logging.getLogger(__name__)

COMPONENT_MAPPING_URL = (
    "https://firefox-ci-tc.services.mozilla.com/api/index/v1/task/"
    "gecko.v2.mozilla-central.latest.source.source-bugzilla-info/artifacts/public/components.json"
)
COMPONENT_MAPPING_PATH = "data/perf_component_mapping.json"


class ComponentMapping:
    """File path to Bugzilla component, with a directory fallback.

    The published mapping lists the files of one revision of the tree. A
    commit from another revision can touch files it does not list (added,
    removed or renamed since), which fall back to the most common component
    of the files under the nearest listed directory.
    """

    def __init__(self, files: Mapping[str, str]) -> None:
        self.files = dict(files)
        counts: dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter
        )
        for path, component in self.files.items():
            parts = path.split("/")
            for depth in range(1, len(parts)):
                counts["/".join(parts[:depth])][component] += 1
        self.directories = {
            directory: counter.most_common(1)[0][0]
            for directory, counter in counts.items()
        }

    @classmethod
    def from_json(cls, path: str) -> "ComponentMapping":
        """Load a ``components.json`` file: path to ``[product, component]``."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls({path: "::".join(component) for path, component in data.items()})

    @classmethod
    def download(cls, path: str = COMPONENT_MAPPING_PATH) -> "ComponentMapping":
        utils.download_check_etag(COMPONENT_MAPPING_URL, path)
        return cls.from_json(path)

    def component_of(self, path: str) -> str | None:
        found = self.files.get(path)
        if found is not None:
            return found
        parts = path.split("/")
        for depth in range(len(parts) - 1, 0, -1):
            found = self.directories.get("/".join(parts[:depth]))
            if found is not None:
                return found
        return None


class AreaIndex:
    """Bugzilla component to the areas whose tests it is likely to move.

    ``areas`` maps each area to its component entries from ``categories.yml``:
    a component listed by name belongs to the areas that name it, and an
    entry ending in ``*`` matches every component starting with the text
    before it, but only for components no area names.
    """

    def __init__(self, areas: Mapping[str, Iterable[str]]) -> None:
        self.exact: dict[str, set[str]] = collections.defaultdict(set)
        self.prefixes: list[tuple[str, str]] = []
        for area, entries in areas.items():
            for entry in entries:
                if entry.endswith("*"):
                    self.prefixes.append((entry[:-1], area))
                else:
                    self.exact[entry].add(area)

    def areas_of(self, component: str) -> frozenset[str]:
        if component in self.exact:
            return frozenset(self.exact[component])
        return frozenset(
            area for prefix, area in self.prefixes if component.startswith(prefix)
        )


def describe_commit(
    files: Iterable[str], mapping: ComponentMapping, index: AreaIndex
) -> dict[str, Any]:
    """The components and areas of a commit from the files it touched.

    ``unmapped_components`` are the components no area claims, such as the
    build system; a commit whose components are all unmapped has no areas,
    and the caller decides whether that means it can regress nothing or
    anything.
    """
    components: set[str] = set()
    unknown_files = 0
    for path in files:
        component = mapping.component_of(path)
        if component is None:
            unknown_files += 1
        else:
            components.add(component)
    areas: set[str] = set()
    unmapped: set[str] = set()
    for component in components:
        found = index.areas_of(component)
        if found:
            areas.update(found)
        else:
            unmapped.add(component)
    return {
        "components": sorted(components),
        "areas": sorted(areas),
        "unmapped_components": sorted(unmapped),
        "unknown_files": unknown_files,
    }


def read_commit_files(path: str) -> Iterable[tuple[str, list[str]]]:
    r"""Yield ``(node, files)`` from a tab-separated ``node<TAB>file|file`` dump.

    Such a dump comes from ``hg log --template '{node}\\t{join(files,"|")}\\n'``.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            node, _, joined = line.rstrip("\n").partition("\t")
            if node:
                yield node, [f for f in joined.split("|") if f]


def load_commit_areas(records: Iterable[dict[str, Any]]) -> dict[str, frozenset[str]]:
    """Index ``perf_commit_areas.json`` records by node."""
    return {record["node"]: frozenset(record["areas"]) for record in records}
