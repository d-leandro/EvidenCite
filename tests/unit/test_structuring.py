"""Pruebas unitarias para el módulo de estructuración de unarXive (src.data.structuring)."""

import json
from pathlib import Path
import shutil

from src.data.schemas import StructuredPaper
from src.data.structuring import (
    StructuringConfig,
    UnarxiveStructurer,
    extract_title_from_raw,
    is_computer_science_paper,
    parse_bib_entry_raw,
    resolve_ref_entries,
    split_sentences,
    structure_single_paper,
)


def test_filter_computer_science():
    """Valida el filtrado dual por macro-disciplina y subcategorías arXiv cs.*."""
    # Caso 1: Macro-disciplina Computer Science directa
    paper_cs_direct = {"discipline": "Computer Science", "metadata": {"categories": "cs.LG"}}
    assert is_computer_science_paper(paper_cs_direct) is True

    # Caso 2: Otra disciplina pero con categoría informática cruzada (ej. cs.LO en Mathematics)
    paper_cs_cross = {"discipline": "Mathematics", "metadata": {"categories": "math.LO cs.LO"}}
    assert is_computer_science_paper(paper_cs_cross) is True

    # Caso 3: Bioinformática cruzada con cs.AI
    paper_cs_bio = {"discipline": "Quantitative Biology", "metadata": {"categories": "physics.chem-ph cs.AI q-bio.BM"}}
    assert is_computer_science_paper(paper_cs_bio) is True

    # Caso 4: Matemática pura (debe ser descartado)
    paper_math = {"discipline": "Mathematics", "metadata": {"categories": "math.PR math.CA"}}
    assert is_computer_science_paper(paper_math) is False

    # Caso 5: Física pura (debe ser descartado, cuidando no hacer falso positivo con 'physics')
    paper_physics = {"discipline": "Physics", "metadata": {"categories": "physics.gen-ph quant-ph"}}
    assert is_computer_science_paper(paper_physics) is False


def test_heading_preservation():
    """Valida que los títulos de sección se conserven exactamente tal como fueron redactados."""
    record = {
        "paper_id": "2212.00001",
        "discipline": "Computer Science",
        "metadata": {"title": "Test Paper", "categories": "cs.CL"},
        "abstract": {"text": "Abstract text here."},
        "body_text": [
            {"section": "3.1 Adversarial Perturbations", "text": "This is paragraph 1."},
            {"section": "3.1 Adversarial Perturbations", "text": "This is paragraph 2."},
            {"section": "4. Ablation Study", "text": "This is paragraph 3."},
        ],
    }
    paper = structure_single_paper(record)
    assert paper is not None
    assert len(paper.sections) == 2
    assert paper.sections[0].heading == "3.1 Adversarial Perturbations"
    assert paper.sections[1].heading == "4. Ablation Study"


def test_formula_replacement_and_float_removal():
    """Valida que las fórmulas se reemplacen por su LaTeX y se remuevan tags flotantes de tablas/figuras."""
    ref_entries = {
        "uuid-form-1": {"latex": r"\mathcal{L}_{adv}", "type": "formula"},
        "uuid-form-empty": {"latex": "", "type": "formula"},
        "uuid-tab-1": {"caption": "Table 1", "type": "table"},
    }
    text = (
        "We optimize the loss {{formula:uuid-form-1}} as shown in Table REF. "
        "{{table:uuid-tab-1}} Also {{formula:uuid-form-empty}} done."
    )
    cleaned = resolve_ref_entries(text, ref_entries)
    assert cleaned == r"We optimize the loss $\mathcal{L}_{adv}$ as shown in Table REF. Also done."


def test_sentence_splitting_with_abbreviations():
    """Valida que el segmentador no se fracture ante abreviaturas comunes o números decimales."""
    text1 = "As shown by Smith et al. (2020), this achieves 0.95 accuracy. We adopt their model."
    sents1 = split_sentences(text1)
    assert len(sents1) == 2
    assert sents1[0] == "As shown by Smith et al. (2020), this achieves 0.95 accuracy."
    assert sents1[1] == "We adopt their model."

    text2 = "In Fig. 1 and Eq. (2), the loss decreases. This confirms Theorem 1.1."
    sents2 = split_sentences(text2)
    assert len(sents2) == 2
    assert sents2[0] == "In Fig. 1 and Eq. (2), the loss decreases."
    assert sents2[1] == "This confirms Theorem 1.1."


def test_cite_spans_anchoring():
    """Valida que los marcadores de citas mantengan offsets exactos y ref_keys válidas en la oración."""
    record = {
        "paper_id": "2212.00002",
        "discipline": "Computer Science",
        "metadata": {"title": "Title", "categories": "cs.IR"},
        "abstract": {"text": "Abstract"},
        "body_text": [
            {
                "section": "Introduction",
                "text": "First sentence has no cite. Second sentence follows {{cite:abc12345}} closely.",
            }
        ],
        "bib_entries": {
            "abc12345": {
                "bib_entry_raw": "Author Name. Title of the Cited Paper. Venue, 2021.",
                "ids": {"arxiv_id": "2105.05862"},
            }
        },
    }
    paper = structure_single_paper(record)
    assert paper is not None
    assert len(paper.sections) == 1
    sentences = paper.sections[0].sentences
    assert len(sentences) == 2

    # Primera oración sin citas
    assert len(sentences[0].citations) == 0

    # Segunda oración con 1 cita
    assert len(sentences[1].citations) == 1
    citation = sentences[1].citations[0]
    assert citation.marker_text == "{{cite:abc12345}}"
    assert citation.ref_key == "abc12345"
    assert sentences[1].text[citation.char_start : citation.char_end] == "{{cite:abc12345}}"


def test_bibliography_title_extraction():
    """Valida la extracción heurística del título desde bib_entry_raw."""
    raw1 = (
        "Alex M Lamb, Anirudh Goyal, Aaron C Courville, and Yoshua Bengio. "
        "Professor forcing: A new algorithm for training recurrent networks. "
        "Advances in neural information processing systems, 29, 2016."
    )
    title1 = extract_title_from_raw(raw1)
    assert title1 == "Professor forcing: A new algorithm for training recurrent networks"

    raw2 = "Hongyan Jing. Using hidden markov modeling to decompose human-written summaries. Computational linguistics, 2002."
    title2 = extract_title_from_raw(raw2)
    assert title2 == "Using hidden markov modeling to decompose human-written summaries"


def test_nearest_enclosing_scope_inheritance():
    """Valida que párrafos subsecuentes con section: null hereden el encabezado de su ámbito más cercano."""
    record = {
        "paper_id": "2212.00003",
        "discipline": "Computer Science",
        "metadata": {"title": "Hierarchical Paper", "categories": "cs.CV"},
        "abstract": {"text": "Abstract"},
        "body_text": [
            {"section": "5. Methodology", "text": "Methodology overview sentence."},
            {"section": "5.1 Feature Extraction", "text": "First sentence of subsection."},
            {"section": None, "text": "Second sentence inheriting subsection."},
            {"section": "", "text": "Third sentence inheriting subsection."},
            {"section": "6. Experiments", "text": "First sentence of experiments."},
        ],
    }
    paper = structure_single_paper(record)
    assert paper is not None
    # Debe consolidarse en exactamente 3 secciones discursivas (no 5 ni con Untitled Section)
    assert len(paper.sections) == 3
    assert paper.sections[0].heading == "5. Methodology"
    assert len(paper.sections[0].sentences) == 1

    assert paper.sections[1].heading == "5.1 Feature Extraction"
    assert len(paper.sections[1].sentences) == 3
    assert paper.sections[1].sentences[0].text == "First sentence of subsection."
    assert paper.sections[1].sentences[1].text == "Second sentence inheriting subsection."
    assert paper.sections[1].sentences[2].text == "Third sentence inheriting subsection."

    assert paper.sections[2].heading == "6. Experiments"
    assert len(paper.sections[2].sentences) == 1


def test_bibliography_parsing_authors_and_year():
    """Valida la extracción desacoplada de autores, año y título en diferentes formatos bibliográficos."""
    # Formato 1: ACM / APA con año entre puntos
    raw_acm = (
        "Aly S Abdelrahim, Moumen T El-Melegy, and Aly A Farag. 2012. "
        "Realistic 3d reconstruction of the human teeth using shape from shading with shape priors. "
        "In 2012 IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops. IEEE, 64–69."
    )
    p_acm = parse_bib_entry_raw(raw_acm)
    assert p_acm["authors"] == "Aly S Abdelrahim, Moumen T El-Melegy, and Aly A Farag"
    assert p_acm["year"] == 2012
    assert p_acm["title"] == "Realistic 3d reconstruction of the human teeth using shape from shading with shape priors"

    # Formato 2: IEEE con título entre comillas tipográficas
    raw_ieee = (
        "A. Calimera, E. Macii, and M. Poncino, “The human brain project and neuromorphic computing,” "
        "International Journal of Neural Systems, 2013."
    )
    p_ieee = parse_bib_entry_raw(raw_ieee)
    assert p_ieee["authors"] == "A. Calimera, E. Macii, and M. Poncino"
    assert p_ieee["year"] == 2013
    assert p_ieee["title"] == "The human brain project and neuromorphic computing"


def test_end_to_end_structuring_sample(tmp_path):
    """Valida la estructuración completa de un shard real de unarXive filtrando a 16 papers de CS."""
    sample_shard = Path("data/raw/unarxive/arXiv_src_2212_086.jsonl")
    if not sample_shard.exists():
        return

    # Usar únicamente el shard real como input creando un directorio aislado
    isolated_input = tmp_path / "shards"
    isolated_input.mkdir()
    shutil.copy(sample_shard, isolated_input / sample_shard.name)

    output_file = tmp_path / "papers.jsonl"
    config = StructuringConfig(
        input_dir=isolated_input,
        output_file=output_file,
        force=True,
    )
    structurer = UnarxiveStructurer(config)
    result = structurer.run()

    assert result["status"] == "success"
    assert result["structured_papers"] == 16
    assert output_file.exists()

    # Validar que cada línea en el archivo generado sea un StructuredPaper válido
    with open(output_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 16

        total_bib_entries = 0
        for line in lines:
            paper_data = json.loads(line)
            paper = StructuredPaper(**paper_data)
            assert paper.discipline == "Computer Science"
            assert len(paper.paper_id) > 0
            assert len(paper.title) > 0
            assert len(paper.sections) > 0
            assert isinstance(paper.bibliography, dict)
            total_bib_entries += len(paper.bibliography)
        assert total_bib_entries > 0


def test_idempotency_and_append(tmp_path):
    """Valida que si papers.jsonl existe se omita sin --force, y que --append agregue registros."""
    sample_shard = Path("data/raw/unarxive/arXiv_src_2212_086.jsonl")
    isolated_input = tmp_path / "shards"
    isolated_input.mkdir(exist_ok=True)
    if sample_shard.exists():
        shutil.copy(sample_shard, isolated_input / sample_shard.name)

    output_file = tmp_path / "papers.jsonl"
    output_file.write_text('{"dummy": 1}\n')

    # Sin force ni append -> omitir
    config = StructuringConfig(
        input_dir=isolated_input,
        output_file=output_file,
        force=False,
        append=False,
    )
    res_skip = UnarxiveStructurer(config).run()
    assert res_skip["status"] == "skipped"

    # Con append y limit=1 -> anexa 1 paper
    config_append = StructuringConfig(
        input_dir=isolated_input,
        output_file=output_file,
        append=True,
        limit=1,
    )
    res_append = UnarxiveStructurer(config_append).run()
    assert res_append["status"] == "limit_reached"
    assert res_append["structured_papers"] == 1

    # El archivo ahora debe tener 2 líneas (dummy + 1 paper)
    with open(output_file, "r", encoding="utf-8") as f:
        assert len(f.readlines()) == 2
