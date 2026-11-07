"""Pre-etiquetado asistido por panel de jueces LLM (Actividad A2)."""

from src.labeling.io import load_instances, write_instances_by_split
from src.labeling.panel import aggregate_votes, apply_decision
from src.labeling.prompts import PROMPT_VERSION, PROMPT_VERSION_V2, build_judge_messages
from src.labeling.schemas import (
    AggregationConfig,
    JudgeConfig,
    JudgeVote,
    LabelingConfig,
    PanelDecision,
    PanelVote,
)

__all__ = [
    "AggregationConfig",
    "JudgeConfig",
    "JudgeVote",
    "LabelingConfig",
    "PanelDecision",
    "PanelVote",
    "PROMPT_VERSION",
    "PROMPT_VERSION_V2",
    "aggregate_votes",
    "apply_decision",
    "build_judge_messages",
    "load_instances",
    "write_instances_by_split",
]
