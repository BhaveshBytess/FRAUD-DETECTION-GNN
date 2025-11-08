"""Utility helpers for grouping Elliptic++ feature columns by provenance.

The dataset exposes three disjoint feature families:

1. Local transaction descriptors (`Local_feature_*`), which cover intrinsic
   transaction statistics such as amount distributions and wallet metadata.
2. Aggregate neighbor descriptors (`Aggregate_feature_*`), which are the suspected
   neighbor-aggregation fields we target in the M7 causality experiment.
3. Structural/manual graph statistics (degree counts, BTC totals, address counts)
   that behave like hand-crafted topological features.

These helpers keep the grouping logic in one place so notebooks/scripts can
request specific feature subsets without re-implementing string filters.
"""
from __future__ import annotations

from typing import Dict, List

LOCAL_FEATURES: List[str] = [f"Local_feature_{i}" for i in range(1, 94)]

# Elliptic++ publishes aggregate columns up to Aggregate_feature_72. Earlier
# documentation referred to AF94–AF182; in this corrected dataset the aggregate
# block is 72 features wide and already contains neighbor-aggregated signals.
AGGREGATE_FEATURES: List[str] = [f"Aggregate_feature_{i}" for i in range(1, 73)]

# Transaction-level structural descriptors (degrees, BTC totals, address counts).
STRUCTURAL_FEATURES: List[str] = [
    "in_txs_degree",
    "out_txs_degree",
    "total_BTC",
    "fees",
    "size",
    "num_input_addresses",
    "num_output_addresses",
    "in_BTC_min",
    "in_BTC_max",
    "in_BTC_mean",
    "in_BTC_median",
    "in_BTC_total",
    "out_BTC_min",
    "out_BTC_max",
    "out_BTC_mean",
    "out_BTC_median",
    "out_BTC_total",
]


def all_features() -> List[str]:
    """Return the union of all known feature columns (without tx/time/class)."""
    return LOCAL_FEATURES + AGGREGATE_FEATURES + STRUCTURAL_FEATURES


def feature_groups() -> Dict[str, List[str]]:
    """Return a mapping of named feature groups for downstream selection."""
    return {
        "local": LOCAL_FEATURES.copy(),
        "aggregate": AGGREGATE_FEATURES.copy(),
        "structural": STRUCTURAL_FEATURES.copy(),
        "local_plus_structural": LOCAL_FEATURES + STRUCTURAL_FEATURES,
        "full": all_features(),
    }


def resolve_group(name: str) -> List[str]:
    """Resolve a friendly group name to concrete feature columns."""
    groups = feature_groups()
    key = name.lower()
    if key not in groups:
        raise KeyError(
            f"Unknown feature group '{name}'. "
            f"Available groups: {', '.join(sorted(groups))}"
        )
    return groups[key]
