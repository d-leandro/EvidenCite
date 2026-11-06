"""Taxonomía canónica de 9 clases de intención de citación (Lyu et al., Scientometrics 2021)."""

from typing import Dict, List, Literal

CanonicalLabel = Literal[
    "Background",
    "Gap",
    "Basis",
    "Comparison",
    "Application",
    "Improvement",
    "Evidence",
    "Originator",
    "Further Reading",
]

CANONICAL_9_CLASSES: List[str] = [
    "Background",
    "Gap",
    "Basis",
    "Comparison",
    "Application",
    "Improvement",
    "Evidence",
    "Originator",
    "Further Reading",
]

CANONICAL_CLASS_DEFINITIONS: Dict[str, str] = {
    "Background": (
        "The cited work provides general context, theoretical background, domain definitions, "
        "or broad overview of the research problem."
    ),
    "Gap": (
        "The cited work motivates the research by highlighting an unsolved problem, limitation, "
        "weakness, or open future research direction."
    ),
    "Basis": (
        "The cited work serves as the direct intellectual, theoretical, or methodological "
        "foundation upon which the citing authors build their approach."
    ),
    "Comparison": (
        "The citing authors compare their method, results, or findings against the cited work "
        "(highlighting similarities, differences, or benchmark performance)."
    ),
    "Application": (
        "The citing authors directly use, adopt, or execute a tool, dataset, algorithm, metric, "
        "or hyperparameter setting from the cited work."
    ),
    "Improvement": (
        "The citing authors modify, extend, optimize, or refine a method or system originally "
        "introduced in the cited work."
    ),
    "Evidence": (
        "The cited work provides empirical facts, quantitative observations, or theorem proofs "
        "that substantiate a specific claim in the citing sentence."
    ),
    "Originator": (
        "The citation credits the seminal author(s) who coined a term, introduced a canonical "
        "concept, or originally defined the task."
    ),
    "Further Reading": (
        "The citation directs the reader to external surveys, appendices, or supplementary "
        "materials for additional details outside the scope of the current paper."
    ),
}
