# -*- coding: utf-8 -*-
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

r"""Describe every commit of the perf pushes snapshot by the code areas it touched.

Writes ``perf_commit_areas.json``: one record per commit with the Bugzilla
components of the files it touched, the areas those components belong to
according to ``perf_test_areas.json``, and the components no area claims.
The files come from ``hg log`` on a local clone, or from a dump of
``hg log --template '{node}\\t{join(files,"|")}\\n'``; the file to component
mapping is the ``components.json`` the ``source-bugzilla-info`` task
publishes. See ``bugbug.perf.commit_areas``.
"""

import argparse
import json
import logging
import os
import subprocess
import tempfile
from typing import Iterable

from bugbug import perf
from bugbug.perf import commit_areas
from bugbug.utils import parse_timestamp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HG_TEMPLATE = r"{node}\t{join(files,'|')}\n"


def dump_commit_files(repo_dir: str, since: str, path: str) -> None:
    """Write the files of every commit of ``repo_dir`` since ``since`` to ``path``."""
    with open(path, "w", encoding="utf-8") as fh:
        subprocess.run(
            [
                "hg",
                "-R",
                repo_dir,
                "log",
                "-r",
                f"date('>{since}')",
                "--template",
                HG_TEMPLATE,
            ],
            stdout=fh,
            check=True,
        )


def load_areas(path: str) -> dict[str, Iterable[str]]:
    with open(path, encoding="utf-8") as fh:
        return {
            record["name"]: record["components"]
            for record in (json.loads(line) for line in fh if line.strip())
            if record["kind"] == "area"
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--repo-dir", help="Local Mercurial clone holding the pushes' commits."
    )
    source.add_argument(
        "--commit-files",
        metavar="PATH",
        help="Dump of hg log with the template the module documents, instead of a clone.",
    )
    parser.add_argument(
        "--components",
        metavar="PATH",
        help="components.json mapping tree paths to Bugzilla components. Default: download the published one.",
    )
    parser.add_argument(
        "--areas",
        default="data/perf_test_areas.json",
        help="perf_test_areas.json written by bugbug-data-perf-test-categories. Default: %(default)s.",
    )
    parser.add_argument(
        "--output",
        default="data/perf_commit_areas.json",
        help="Where to write the records. Default: %(default)s.",
    )
    args = parser.parse_args()

    pushes = list(perf.get_perf_pushes())
    nodes = {commit["revision"] for push in pushes for commit in push["commits"]}
    logger.info("%d commits in %d pushes", len(nodes), len(pushes))

    if args.components:
        mapping = commit_areas.ComponentMapping.from_json(args.components)
    else:
        mapping = commit_areas.ComponentMapping.download()
    index = commit_areas.AreaIndex(load_areas(args.areas))

    if args.commit_files:
        files_path = args.commit_files
    else:
        first = min(
            time
            for push in pushes
            if (time := parse_timestamp(push["time"])) is not None
        )
        files_path = os.path.join(tempfile.mkdtemp(), "commit_files.tsv")
        logger.info("Listing files of commits since %s from %s", first, args.repo_dir)
        dump_commit_files(args.repo_dir, first.strftime("%Y-%m-%d"), files_path)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    written = with_areas = 0
    seen = set()
    with open(args.output, "w", encoding="utf-8") as out:
        for node, files in commit_areas.read_commit_files(files_path):
            if node not in nodes or node in seen:
                continue
            seen.add(node)
            record = commit_areas.describe_commit(files, mapping, index)
            out.write(json.dumps({"node": node, "files": len(files), **record}) + "\n")
            written += 1
            with_areas += bool(record["areas"])
    missing = len(nodes) - written
    if missing:
        logger.warning("%d commits of the pushes are not in the clone", missing)
    logger.info(
        "%d commits written to %s, %d with at least one area",
        written,
        args.output,
        with_areas,
    )


if __name__ == "__main__":
    main()
