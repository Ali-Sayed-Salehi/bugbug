# -*- coding: utf-8 -*-
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

"""Read what every performance test covers from a Firefox checkout.

Writes ``perf_test_categories.json``, one record per test keyed by Perfherder
framework and suite with the areas and scenario the test declares in the
tree, and ``perf_test_areas.json``, the vocabulary of
``testing/performance/categories.yml`` with the Bugzilla components of each
area. Until Perfherder stores the categories on the signature, this is how
the test selection experiments get them; see ``bugbug.perf.test_categories``.
"""

import argparse
import collections
import json
import logging
import os

from bugbug.perf import test_categories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def write_records(path: str, records) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--firefox-dir", required=True, help="Firefox checkout to read the tests from."
    )
    parser.add_argument(
        "--output",
        default="data/perf_test_categories.json",
        help="Where to write the test records. Default: %(default)s.",
    )
    parser.add_argument(
        "--areas-output",
        default="data/perf_test_areas.json",
        help="Where to write the areas and scenarios vocabulary. Default: %(default)s.",
    )
    args = parser.parse_args()

    vocabulary = test_categories.load_vocabulary(args.firefox_dir)
    records = list(test_categories.generate_test_categories(args.firefox_dir))
    unknown = sorted(
        {area for record in records for area in record["areas"]}
        - set(vocabulary["areas"])
    )
    if unknown:
        logger.warning("Areas not in categories.yml: %s", ", ".join(unknown))
    undeclared = [
        f"{record['framework']}/{record['test']}"
        for record in records
        if not record["areas"] or not record.get("scenario")
    ]
    if undeclared:
        logger.warning(
            "%d tests declare no areas or scenario: %s",
            len(undeclared),
            ", ".join(undeclared[:10]),
        )
    write_records(args.output, records)
    write_records(args.areas_output, test_categories.vocabulary_records(vocabulary))
    by_framework = collections.Counter(record["framework"] for record in records)
    logger.info(
        "%d tests written to %s (%s); vocabulary of %d areas and %d scenarios to %s",
        len(records),
        args.output,
        ", ".join(f"{k} {v}" for k, v in sorted(by_framework.items())),
        len(vocabulary["areas"]),
        len(vocabulary["scenarios"]),
        args.areas_output,
    )


if __name__ == "__main__":
    main()
