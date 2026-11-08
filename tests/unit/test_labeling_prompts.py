"""Pruebas unitarias del cegado de prompts de los jueces respecto al artículo citado (src.labeling.prompts)."""

from src.data.schemas import (
    AnnotationMetadata,
    CitedPaperInfo,
    CitingPaperInfo,
    DatasetInstance,
    RetrievedChunk,
)
from src.data.taxonomy import CANONICAL_9_CLASSES, CANONICAL_CLASS_DEFINITIONS
import pytest

from src.labeling.prompts import PROMPT_VERSION, PROMPT_VERSION_V2, build_judge_messages, build_system_prompt

CITED_TITLE = "UNIQUE_CITED_TITLE_MARKER"
CITED_ABSTRACT = "UNIQUE_CITED_ABSTRACT_MARKER"
CHUNK_TEXT = "UNIQUE_CHUNK_TEXT_MARKER"
CHUNK_SECTION = "UNIQUE_CHUNK_SECTION_MARKER"
CITED_ID = "UNIQUE_CITED_ID_MARKER"


def make_instance(section="Related Work") -> DatasetInstance:
    return DatasetInstance(
        instance_id="inst_00042",
        label="Background",
        annotation_metadata=AnnotationMetadata(source_dataset="unarxive_mined", label_source="weak_nli", is_gold=False, split="train"),
        citing_paper=CitingPaperInfo(id="citing_42", rhetorical_section=section),
        citation_context="Prior work <cite>Doe (2020)</cite> studied this problem.",
        cited_paper=CitedPaperInfo(
            id=CITED_ID,
            title=CITED_TITLE,
            abstract=CITED_ABSTRACT,
            retrieved_chunks=[
                RetrievedChunk(chunk_id="x_s0_c0", section_heading=CHUNK_SECTION, text=CHUNK_TEXT, similarity_score=0.8, rank=1)
            ],
        ),
    )


@pytest.mark.parametrize("version", [PROMPT_VERSION, PROMPT_VERSION_V2])
def test_messages_contain_only_citing_information(version):
    messages = build_judge_messages(make_instance(), version)
    full_text = "\n".join(m["content"] for m in messages)
    for marker in (CITED_TITLE, CITED_ABSTRACT, CHUNK_TEXT, CHUNK_SECTION, CITED_ID):
        assert marker not in full_text


def test_user_prompt_includes_context_and_section():
    messages = build_judge_messages(make_instance())
    assert [m["role"] for m in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert "Rhetorical Section: Related Work" in user
    assert "<cite>Doe (2020)</cite>" in user


def test_missing_section_falls_back_to_unknown():
    user = build_judge_messages(make_instance(section=None))[1]["content"]
    assert "Rhetorical Section: Unknown Section" in user


def test_system_prompt_lists_all_canonical_definitions_without_examples():
    system = build_system_prompt()
    for label in CANONICAL_9_CLASSES:
        assert f"- {label}: {CANONICAL_CLASS_DEFINITIONS[label]}" in system
    assert "Example" not in system


def test_prompt_version_is_declared():
    assert PROMPT_VERSION == "v1_zeroshot_definitions_only"


def test_v2_prompt_uses_only_the_citing_context():
    messages = build_judge_messages(make_instance(), PROMPT_VERSION_V2)
    assert messages[1]["content"] == "Citing Context:\nPrior work <cite>Doe (2020)</cite> studied this problem."
    full_text = "\n".join(m["content"] for m in messages)
    assert "Related Work" not in full_text
    assert "rhetorical section" not in full_text.lower()


def test_v2_system_prompt_keeps_all_definitions():
    system = build_system_prompt(PROMPT_VERSION_V2)
    for label in CANONICAL_9_CLASSES:
        assert f"- {label}: {CANONICAL_CLASS_DEFINITIONS[label]}" in system


def test_unknown_prompt_version_is_rejected():
    with pytest.raises(ValueError):
        build_judge_messages(make_instance(), "v9")
