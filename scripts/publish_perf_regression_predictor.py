# -*- coding: utf-8 -*-
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

"""Publish an externally-trained Perf Regression Predictor checkpoint.

The checkpoint for this model is trained outside bugbug and published as a
public ``.tar.zst`` archive holding a single ``perfregressionpredictormodel``
directory. That is already the shape ``utils.download_model`` expects, so this
script downloads the archive under the artifact name and Taskcluster indexes it
as-is; the bytes are never repacked or inspected.

Nothing here trains anything. Once training moves into bugbug the model can
implement ``train()``, set ``training_supported = True`` and be published by
``bugbug-train`` instead; the artifact this script publishes is deliberately
identical to what such a task would emit, so that switch requires no changes
downstream.
"""

import argparse
import json
import logging
import os
import sys
from urllib.parse import urlparse

from bugbug.utils import get_session, get_user_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Must match bugbug.models.perf_regression_predictor.MODEL_IDENTIFIER. It is
# duplicated rather than imported so that publishing does not depend on the
# extras that the model module needs.
MODEL_IDENTIFIER = "perfregressionpredictor"
MODEL_DIRECTORY = f"{MODEL_IDENTIFIER}model"

# utils.download_model() fetches exactly this name and asserts that unpacking it
# leaves a MODEL_DIRECTORY directory behind, so the published archive is saved
# under this name and indexed verbatim.
ARTIFACT = f"{MODEL_DIRECTORY}.tar.zst"

DOWNLOAD_CHUNK_SIZE = 1024 * 1024

METRICS_FILE = "metrics.json"


def download_archive(url: str, destination: str) -> None:
    """Stream the published archive to disk under the artifact name."""
    logger.info("Downloading %s", url)

    response = get_session(urlparse(url).netloc).get(
        url,
        stream=True,
        allow_redirects=True,
        headers={"User-Agent": get_user_agent()},
    )
    response.raise_for_status()

    with open(destination, "wb") as output:
        for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
            output.write(chunk)

    logger.info("Downloaded %.1f MiB", os.path.getsize(destination) / 1024 / 1024)


def write_metrics(url: str) -> None:
    """Emit metrics.json alongside the artifact, matching the train tasks.

    The checkpoint is trained outside bugbug, so there are no training metrics
    to report. Publish its provenance instead rather than leaving the artifact
    the pipeline expects missing.
    """
    with open(METRICS_FILE, "w", encoding="utf-8") as output:
        json.dump(
            {"trained_by_bugbug": False, "source_url": url},
            output,
            indent=2,
            sort_keys=True,
        )


def publish(url: str) -> str:
    """Download the checkpoint archive into the cwd under the artifact name."""
    download_archive(url, ARTIFACT)
    write_metrics(url)

    logger.info(
        "Published %s (%.1f MiB) from %s",
        ARTIFACT,
        os.path.getsize(ARTIFACT) / 1024 / 1024,
        url,
    )

    return ARTIFACT


def parse_args(args):
    parser = argparse.ArgumentParser(
        description=(
            "Publish an externally-trained Perf Regression Predictor checkpoint "
            "as a bugbug model artifact"
        )
    )
    parser.add_argument(
        "--url",
        required=True,
        help=(
            "Public URL of the checkpoint archive, e.g. "
            "https://storage.googleapis.com/<bucket>/<object>.tar.zst. It must "
            f"be a .tar.zst holding a single {MODEL_DIRECTORY}/ directory"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write the artifact to (default: the current directory)",
    )

    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])

    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)

    publish(args.url)


if __name__ == "__main__":
    main()
