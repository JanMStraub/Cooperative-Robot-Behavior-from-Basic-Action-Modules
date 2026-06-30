"""
BenchmarkFeatureFlags - per-sequence feature flag overrides for ablation benchmarks.

Serialised as a compact JSON blob appended to SEQUENCE_QUERY messages.
None values are omitted (meaning: use server default).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class BenchmarkFeatureFlags:
    """
    Optional per-sequence feature overrides sent from benchmark runner to server.

    None = no override (server keeps its current value).
    True/False = explicit override applied for this sequence only.
    """

    use_rag: Optional[bool] = None
    use_vgn: Optional[bool] = None
    use_ros: Optional[bool] = None
    use_negotiation: Optional[bool] = None
    use_reflection: Optional[bool] = None

    def to_json(self) -> str:
        """
        Serialise to compact JSON string; returns empty string when all fields are None.
        """
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        return json.dumps(d, separators=(",", ":")) if d else ""

    @classmethod
    def from_json(cls, text: str) -> "BenchmarkFeatureFlags":
        """
        Deserialise from JSON string produced by to_json().
        """
        if not text:
            return cls()
        d: Dict[str, Any] = json.loads(text)
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_config_snapshot(cls, snapshot: Dict[str, Any]) -> "BenchmarkFeatureFlags":
        """
        Build flags from a BenchmarkConfig.config_snapshot dict.

        Translates BenchmarkConfig field names to BenchmarkFeatureFlags field names.
        Only includes fields that are explicitly present in the snapshot.
        """
        mapping = {
            "use_rag": "use_rag",
            "use_vgn": "use_vgn",
            "use_ros_movement": "use_ros",
            "reflection_enabled": "use_reflection",
        }
        kwargs: Dict[str, Any] = {}
        for snap_key, flag_key in mapping.items():
            if snap_key in snapshot:
                kwargs[flag_key] = snapshot[snap_key]
        if "use_negotiation" in snapshot:
            kwargs["use_negotiation"] = snapshot["use_negotiation"]
        return cls(**kwargs)
