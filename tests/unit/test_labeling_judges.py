"""Pruebas unitarias de jueces con cliente simulado, caché, escritura por partición y orquestador (src.labeling)."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest
import yaml

from src.data.schemas import AnnotationMetadata, CitedPaperInfo, CitingPaperInfo, DatasetInstance
from src.labeling import judge_panel
from src.labeling.io import load_instances, write_instances_by_split
from src.labeling.judges import Judge, parse_vote_text
from src.labeling.schemas import JudgeConfig, JudgeVote


class FakeCompletions:
    """Simula client.chat.completions con parse() y create() configurables."""

    def __init__(self, label="Application", parse_mode="ok", text=None):
        self.label = label
        self.parse_mode = parse_mode
        self.text = text
        self.calls = 0

    def _completion(self, message):
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    def parse(self, **kwargs):
        self.calls += 1
        if self.parse_mode == "error":
            raise RuntimeError("structured outputs no soportado")
        if self.parse_mode == "none":
            return self._completion(SimpleNamespace(parsed=None, content=self.text))
        vote = JudgeVote(rationale="uses a tool from the cited work", label=self.label)
        return self._completion(SimpleNamespace(parsed=vote, content=vote.model_dump_json()))

    def create(self, **kwargs):
        self.calls += 1
        if self.text is None:
            raise RuntimeError("endpoint caído")
        return self._completion(SimpleNamespace(content=self.text))


def fake_client(**kwargs):
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(**kwargs)))


def judge_config(judge_id="j1"):
    return JudgeConfig(id=judge_id, provider="test", model_name=f"model-{judge_id}", base_url="http://localhost:0/v1", max_concurrency=2)


def make_instance(idx: int, split: str, label: str = "Background", is_gold: bool = False) -> DatasetInstance:
    return DatasetInstance(
        instance_id=f"inst_{idx:05d}",
        label=label,
        annotation_metadata=AnnotationMetadata(
            source_dataset="multicite" if is_gold else "unarxive_mined",
            label_source="human_gold" if is_gold else "weak_nli",
            is_gold=is_gold,
            split=split,
        ),
        citing_paper=CitingPaperInfo(id=f"citing_{idx}", rhetorical_section="Method"),
        citation_context=f"We use the tool of <cite>Author {idx}</cite>.",
        cited_paper=CitedPaperInfo(id=f"cited_{idx}", title="t", abstract="a"),
    )


def test_parse_vote_text_tolerates_markdown_and_label_case():
    vote = parse_vote_text('Sure!\n```json\n{"rationale": "compares results", "label": "comparison"}\n```')
    assert vote.label == "Comparison"
    assert parse_vote_text('{"rationale": "x", "label": "further_reading"}').label == "Further Reading"


def test_parse_vote_text_rejects_unknown_label():
    with pytest.raises(ValueError):
        parse_vote_text('{"rationale": "x", "label": "Motivation"}')


def test_judge_structured_vote_and_cache_idempotence(tmp_path: Path):
    client = fake_client(label="Application")
    judge = Judge(judge_config(), tmp_path, client=client)
    inst = make_instance(1, "train")

    vote = judge.vote(inst)
    assert vote.label == "Application"
    assert vote.prompt_tokens == 100 and vote.completion_tokens == 20
    assert judge.vote(inst).label == "Application"
    assert client.chat.completions.calls == 1

    reloaded_client = fake_client(label="Gap")
    reloaded = Judge(judge_config(), tmp_path, client=reloaded_client)
    assert reloaded.vote(inst).label == "Application"
    assert reloaded_client.chat.completions.calls == 0

    forced = Judge(judge_config(), tmp_path, client=fake_client(label="Gap"), force=True)
    assert forced.vote(inst).label == "Gap"


def test_judge_falls_back_to_plain_text_when_parse_returns_none(tmp_path: Path):
    client = fake_client(parse_mode="none", text='{"rationale": "builds on it", "label": "Basis"}')
    assert Judge(judge_config(), tmp_path, client=client).vote(make_instance(2, "dev")).label == "Basis"


def test_judge_falls_back_to_create_when_parse_raises(tmp_path: Path):
    client = fake_client(parse_mode="error", text='{"rationale": "supports claim", "label": "Evidence"}')
    assert Judge(judge_config(), tmp_path, client=client).vote(make_instance(3, "dev")).label == "Evidence"


def test_judge_abstains_without_aborting(tmp_path: Path):
    client = fake_client(parse_mode="error", text=None)
    vote = Judge(judge_config(), tmp_path, client=client).vote(make_instance(4, "test"))
    assert vote.label is None
    assert vote.error


def test_vote_all_preserves_every_instance(tmp_path: Path):
    instances = [make_instance(i, "train") for i in range(25)]
    votes = Judge(judge_config(), tmp_path, client=fake_client()).vote_all(instances)
    assert list(votes) == [inst.instance_id for inst in instances]


def test_write_instances_by_split_preserves_order_and_splits(tmp_path: Path):
    instances = [make_instance(i, split) for i, split in enumerate(["train", "dev", "test", "train", "test"])]
    write_instances_by_split(instances, tmp_path)
    assert [i.instance_id for i in load_instances(tmp_path / "instances.jsonl")] == [i.instance_id for i in instances]
    for split, expected in {"train": 2, "dev": 1, "test": 2}.items():
        rows = load_instances(tmp_path / f"{split}.jsonl")
        assert len(rows) == expected
        assert all(r.annotation_metadata.split == split for r in rows)


def test_orchestrator_end_to_end(tmp_path: Path, monkeypatch):
    instances: List[DatasetInstance] = [
        make_instance(0, "train", label="Background"),
        make_instance(1, "dev", label="Application"),
        make_instance(2, "test", label="Comparison", is_gold=True),
    ]
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    with open(dataset_dir / "instances.jsonl", "w", encoding="utf-8") as f:
        for inst in instances:
            f.write(inst.model_dump_json() + "\n")

    labels = {"j1": "Application", "j2": "Application", "j3": "Gap"}
    config = {
        "input_path": str(dataset_dir / "instances.jsonl"),
        "output_dir": str(tmp_path / "labeled"),
        "log_path": str(tmp_path / "logs" / "labeling.log"),
        "judges": [judge_config(j).model_dump() for j in labels],
    }
    config_path = tmp_path / "labeling.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def fake_build_judges(configs, cache_dir, force=False):
        return [Judge(cfg, cache_dir, client=fake_client(label=labels[cfg.id]), force=force) for cfg in configs]

    monkeypatch.setattr(judge_panel, "build_judges", fake_build_judges)
    judge_panel.main(["--config", str(config_path)])

    out = tmp_path / "labeled"
    after = {i.instance_id: i for i in load_instances(out / "instances.jsonl")}
    assert len(after) == 3
    assert after["inst_00000"].annotation_metadata.label_source == "needs_review"
    assert after["inst_00000"].label == "Background"
    assert after["inst_00001"].annotation_metadata.label_source == "llm_panel"
    assert after["inst_00002"].label == "Comparison"
    assert after["inst_00002"].annotation_metadata.label_source == "human_gold"
    assert len(load_instances(out / "test.jsonl")) == 1

    metrics = json.loads((out / "labeling_metrics.json").read_text(encoding="utf-8"))
    assert metrics["decision_cases"] == {"B": 1, "C": 1, "F": 1}
    assert metrics["gold_calibration"]["evaluated_instances"] == 1
    assert set(metrics["judge_usage"]) == {"j1", "j2", "j3"}
    assert "NaN" not in (out / "labeling_metrics.json").read_text(encoding="utf-8")
    assert (tmp_path / "logs" / "labeling.log").exists()
    for judge_id in labels:
        assert (out / "votes" / f"{judge_id}.jsonl").exists()
