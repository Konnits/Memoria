import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.irregular_timeseries_generator import (
    DatasetBundle,
    GeneratorConfig,
    SamplingConfig,
    SplitConfig,
    SyntheticCollection,
    _generate_gaps,
    benchmark_preset,
)


def test_long_gap_preset_places_a_gap_inside_every_chronological_split() -> None:
    config = benchmark_preset("long_gaps", seed=7)
    gaps = _generate_gaps(
        config.sampling,
        np.random.default_rng(7),
        split_config=config.split,
    )

    boundaries = ((0.0, 0.7), (0.7, 0.85), (0.85, 1.0))
    assert len(gaps) == 3
    for (start, end), (split_start, split_end) in zip(gaps, boundaries):
        assert split_start <= start < end <= split_end


def test_random_gap_generation_remains_backwards_compatible() -> None:
    config = SamplingConfig(n_gaps=2, total_gap_fraction=0.1)
    gaps = _generate_gaps(config, np.random.default_rng(3))

    assert len(gaps) == 2
    assert gaps[0][1] <= gaps[1][0]


def test_future_collection_saves_preserve_generator_seed_provenance(
    tmp_path: Path,
) -> None:
    seed = 3031
    bundle = DatasetBundle(
        observations=pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "value": [0.0, 0.5, 1.0],
                "split": ["train", "validation", "test"],
            }
        ),
        truth=None,
        metadata={"dataset_id": "probe_0000"},
    )
    collection = SyntheticCollection(
        datasets=[bundle],
        kind="univariate",
        config=GeneratorConfig(seed=seed),
    )

    collection.save(tmp_path)

    collection_metadata = json.loads(
        (tmp_path / f"collection_metadata_gseed{seed}.json").read_text(
            encoding="utf-8"
        )
    )
    dataset_metadata = json.loads(
        (tmp_path / "probe_0000" / "metadata.json").read_text(encoding="utf-8")
    )
    assert collection_metadata["generator_seed"] == seed
    assert dataset_metadata["generator_seed"] == seed
