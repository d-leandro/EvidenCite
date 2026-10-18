"""Módulo de resolución e ingestión de artículos seminales de MultiCite.

Permite pre-calentar la caché de Semantic Scholar (data/raw/s2_cache.db)
y anexar los artículos seminales faltantes a data/structured/papers.jsonl
para maximizar las instancias con has_cited_chunks = True en Configuración A y B.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel, Field

from src.data.ingestion.s2_client import S2ClientConfig, S2PaperMetadata, SemanticScholarClient
from src.data.schemas import BibliographyEntry, CitationMarker, StructuredPaper, StructuredSection, StructuredSentence
from src.data.structuring import clean_whitespace, split_sentences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SeminalPaperSpec(BaseModel):
    """Especificación canónica de un artículo seminal altamente citado en MultiCite."""

    arxiv_id: str = Field(..., description="Identificador canónico en arXiv")
    title: str = Field(..., description="Título oficial de la publicación")
    year: int = Field(..., description="Año de publicación")
    first_author: str = Field(..., description="Primer autor (apellido normalizado en minúsculas)")
    authors: List[str] = Field(default_factory=list, description="Lista de autores")
    markers: List[str] = Field(default_factory=list, description="Variantes textuales de citación en MultiCite")


# Catálogo curado de los artículos seminales de mayor impacto y frecuencia en MultiCite
SEMINAL_CATALOG: List[SeminalPaperSpec] = [
    SeminalPaperSpec(
        arxiv_id="1706.03762",
        title="Attention is All You Need",
        year=2017,
        first_author="vaswani",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"],
        markers=["Vaswani et al., 2017", "Vaswani et al. 2017", "Vaswani et al. (2017", "Vaswani 2017", "Vaswani et al. (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1810.04805",
        title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        year=2018,
        first_author="devlin",
        authors=["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
        markers=["Devlin et al., 2018", "Devlin et al. 2018", "Devlin et al. (2018", "Devlin et al., 2019", "Devlin et al. 2019", "Devlin et al. (2019", "Devlin 2018", "Devlin 2019"],
    ),
    SeminalPaperSpec(
        arxiv_id="1705.02364",
        title="Supervised Learning of Universal Sentence Representations from Natural Language Inference Data",
        year=2017,
        first_author="conneau",
        authors=["Alexis Conneau", "Douwe Kiela", "Holger Schwenk", "Loïc Barrault", "Antoine Bordes"],
        markers=["Conneau et al. 2017", "Conneau et al., 2017", "Conneau et al. (2017", "Conneau et al. (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1710.04087",
        title="Word Translation Without Parallel Data",
        year=2018,
        first_author="conneau",
        authors=["Alexis Conneau", "Guillaume Lample", "Marc'Aurelio Ranzato", "Ludovic Denoyer", "Hervé Jégou"],
        markers=["Conneau et al. 2018", "Conneau et al., 2018", "Conneau et al. (2018", "Conneau et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1603.01354",
        title="End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF",
        year=2016,
        first_author="ma",
        authors=["Xuezhe Ma", "Eduard Hovy"],
        markers=["Ma and Hovy, 2016", "Ma and Hovy 2016", "Ma and Hovy (2016", "Ma & Hovy, 2016", "Ma and Hovy (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1607.06520",
        title="Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings",
        year=2016,
        first_author="bolukbasi",
        authors=["Tolga Bolukbasi", "Kai-Wei Chang", "James Zou", "Venkatesh Saligrama", "Adam Kalai"],
        markers=["Bolukbasi et al. 2016", "Bolukbasi et al., 2016", "Bolukbasi et al. (2016", "Bolukbasi et al. (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1601.00706",
        title="End-to-End Relation Extraction using LSTMs on Sequences and Tree Structures",
        year=2016,
        first_author="miwa",
        authors=["Makoto Miwa", "Mohit Bansal"],
        markers=["Miwa and Bansal 2016", "Miwa and Bansal, 2016", "Miwa and Bansal (2016", "Miwa & Bansal, 2016", "Miwa and Bansal (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1704.04920",
        title="Deep Joint Entity Disambiguation with Local Neural Attention",
        year=2017,
        first_author="ganea",
        authors=["Octavian-Eugen Ganea", "Thomas Hofmann"],
        markers=["Ganea and Hofmann 2017", "Ganea and Hofmann, 2017", "Ganea and Hofmann (2017", "Ganea and Hofmann (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1707.00941",
        title="Learning bilingual word embeddings with (almost) no bilingual data",
        year=2017,
        first_author="artetxe",
        authors=["Mikel Artetxe", "Gorka Labaka", "Eneko Agirre"],
        markers=["Artetxe et al. 2017", "Artetxe et al., 2017", "Artetxe et al. (2017", "Artetxe et al. (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1706.04115",
        title="Zero-Shot Relation Extraction via Reading Comprehension",
        year=2017,
        first_author="levy",
        authors=["Omer Levy", "Minjoon Seo", "Eunsol Choi", "Luke Zettlemoyer"],
        markers=["Levy et al. 2017", "Levy et al., 2017", "Levy et al. (2017", "Levy et al. (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1606.04155",
        title="Rationalizing Neural Predictions",
        year=2016,
        first_author="lei",
        authors=["Tao Lei", "Regina Barzilay", "Tommi Jaakkola"],
        markers=["Lei et al., 2016", "Lei et al. 2016", "Lei et al. (2016", "Lei et al. (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1310.4546",
        title="Distributed Representations of Words and Phrases and their Compositionality",
        year=2013,
        first_author="mikolov",
        authors=["Tomas Mikolov", "Ilya Sutskever", "Kai Chen", "Greg Corrado", "Jeffrey Dean"],
        markers=["Mikolov et al. 2013b", "Mikolov et al., 2013b", "Mikolov et al. (2013b", "Mikolov et al. 2013a", "Mikolov et al., 2013a", "Mikolov et al. (2013)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1301.3781",
        title="Efficient Estimation of Word Representations in Vector Space",
        year=2013,
        first_author="mikolov",
        authors=["Tomas Mikolov", "Kai Chen", "Greg Corrado", "Jeffrey Dean"],
        markers=["Mikolov et al., 2013", "Mikolov et al. 2013", "Mikolov et al. (2013a)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1901.02860",
        title="Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
        year=2019,
        first_author="dai",
        authors=["Zihang Dai", "Zhilin Yang", "Yiming Yang", "Jaime Carbonell", "Quoc V. Le", "Ruslan Salakhutdinov"],
        markers=["Dai et al., 2019", "Dai et al. 2019", "Dai et al. (2019", "Dai et al. (2019)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1612.00384",
        title="Definition Modeling: Learning to define word embeddings in natural language",
        year=2017,
        first_author="noraset",
        authors=["Thanapon Noraset", "Chen Liang", "Larry Birnbaum", "Doug Downey"],
        markers=["Noraset et al. 2017", "Noraset et al., 2017", "Noraset et al. (2017", "Noraset et al. (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1805.11004",
        title="Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting",
        year=2018,
        first_author="chen",
        authors=["Yen-Chun Chen", "Mohit Bansal"],
        markers=["Chen and Bansal, 2018", "Chen and Bansal 2018", "Chen and Bansal (2018", "Chen & Bansal, 2018", "Chen and Bansal (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1504.00548",
        title="Learning to Understand Phrases by Embedding the Dictionary",
        year=2016,
        first_author="hill",
        authors=["Felix Hill", "KyungHyun Cho", "Anna Korhonen", "Yoshua Bengio"],
        markers=["Hill et al., 2016", "Hill et al. 2016", "Hill et al. (2016", "Hill et al. (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1603.01360",
        title="Neural Architectures for Named Entity Recognition",
        year=2016,
        first_author="lample",
        authors=["Guillaume Lample", "Miguel Ballesteros", "Sandeep Subramanian", "Kazuya Kawakami", "Chris Dyer"],
        markers=["Lample et al., 2016", "Lample et al. 2016", "Lample et al. (2016", "Lample et al. (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1704.05742",
        title="Adversarial Multi-task Learning for Text Classification",
        year=2017,
        first_author="liu",
        authors=["Pengfei Liu", "Xipeng Qiu", "Xuanjing Huang"],
        markers=["Liu et al. 2017", "Liu et al., 2017", "Liu et al. (2017", "Liu et al. (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1607.03450",
        title="A Joint Model for Multilingual Entity Linking",
        year=2016,
        first_author="xu",
        authors=["Yang Xu", "Sander Canisius", "Kenton Lee", "Kristina Toutanova"],
        markers=["Xu et al., 2016", "Xu et al. 2016", "Xu et al. (2016", "Xu et al. (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1705.04304",
        title="A Deep Reinforced Model for Abstractive Summarization",
        year=2017,
        first_author="paulus",
        authors=["Romain Paulus", "Caiming Xiong", "Richard Socher"],
        markers=["Paulus et al., 2017", "Paulus et al. 2017", "Paulus et al. (2017", "Paulus et al. (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1602.07776",
        title="Recurrent Neural Network Grammars",
        year=2016,
        first_author="dyer",
        authors=["Chris Dyer", "Adhiguna Kuncoro", "Miguel Ballesteros", "Noah A. Smith"],
        markers=["Dyer et al. 2016", "Dyer et al., 2016", "Dyer et al. (2016", "Dyer et al. (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1802.05365",
        title="Deep contextualized word representations",
        year=2018,
        first_author="peters",
        authors=["Matthew E. Peters", "Mark Neumann", "Mohit Iyyer", "Matt Gardner", "Christopher Clark", "Kenton Lee", "Luke Zettlemoyer"],
        markers=["Peters et al., 2018", "Peters et al. 2018", "Peters et al. (2018", "Peters et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1802.02317",
        title="Explainable Prediction of Medical Codes from Clinical Text",
        year=2018,
        first_author="mullenbach",
        authors=["James Mullenbach", "Sarah Wiegreffe", "Jon Duke", "Jimeng Sun", "Jacob Eisenstein"],
        markers=["Mullenbach et al. 2018", "Mullenbach et al., 2018", "Mullenbach et al. (2018", "Mullenbach et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1511.08308",
        title="Named Entity Recognition with Bidirectional LSTM-CNNs",
        year=2016,
        first_author="chiu",
        authors=["Jason P. C. Chiu", "Eric Nichols"],
        markers=["Chiu and Nichols, 2016", "Chiu and Nichols 2016", "Chiu & Nichols, 2016", "Chiu and Nichols (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1511.06746",
        title="Hierarchical Attention Networks for Document Classification",
        year=2016,
        first_author="yang",
        authors=["Zichao Yang", "Diyi Yang", "Chris Dyer", "Xiaodong He", "Alex Smola", "Eduard Hovy"],
        markers=["Yang et al. 2016", "Yang et al., 2016", "Yang et al. (2016", "Yang et al. (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1508.06649",
        title="WikiQA: A Challenge Dataset for Open-Domain Question Answering",
        year=2015,
        first_author="yang",
        authors=["Yi Yang", "Wen-tau Yih", "Christopher Meek"],
        markers=["Yang et al. 2015", "Yang et al., 2015", "Yang et al. (2015", "Yang et al. (2015)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1709.03367",
        title="Position-aware Attention and Dual Inputs for Aspect-level Sentiment Classification",
        year=2017,
        first_author="zhang",
        authors=["Chen Zhang", "Qiuchi Li", "Dawei Song"],
        markers=["Zhang et al. 2017", "Zhang et al., 2017", "Zhang et al. (2017", "Zhang et al. (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1409.0473",
        title="Neural Machine Translation by Jointly Learning to Align and Translate",
        year=2014,
        first_author="bahdanau",
        authors=["Dzmitry Bahdanau", "Kyunghyun Cho", "Yoshua Bengio"],
        markers=["Bahdanau et al., 2014", "Bahdanau et al. 2014", "Bahdanau et al., 2015", "Bahdanau et al. (2015", "Bahdanau et al. (2014)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1412.6980",
        title="Adam: A Method for Stochastic Optimization",
        year=2014,
        first_author="kingma",
        authors=["Diederik P. Kingma", "Jimmy Ba"],
        markers=["Kingma and Ba, 2014", "Kingma and Ba 2014", "Kingma & Ba, 2014", "Kingma and Ba (2014", "Kingma and Ba (2015)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1409.3215",
        title="Sequence to Sequence Learning with Neural Networks",
        year=2014,
        first_author="sutskever",
        authors=["Ilya Sutskever", "Oriol Vinyals", "Quoc V. Le"],
        markers=["Sutskever et al., 2014", "Sutskever et al. 2014", "Sutskever et al. (2014", "Sutskever et al. (2014)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1408.5882",
        title="Convolutional Neural Networks for Sentence Classification",
        year=2014,
        first_author="kim",
        authors=["Yoon Kim"],
        markers=["Kim, 2014", "Kim 2014", "Kim (2014", "Yoon Kim 2014", "Kim (2014)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1406.2661",
        title="Generative Adversarial Nets",
        year=2014,
        first_author="goodfellow",
        authors=["Ian J. Goodfellow", "Jean Pouget-Abadie", "Mehdi Mirza", "Bing Xu", "David Warde-Farley", "Sherjil Ozair", "Aaron Courville", "Yoshua Bengio"],
        markers=["Goodfellow et al., 2014", "Goodfellow et al. 2014", "Goodfellow et al. (2014", "Goodfellow et al. (2014)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1901.07291",
        title="Cross-lingual Language Model Pretraining",
        year=2019,
        first_author="lample",
        authors=["Guillaume Lample", "Alexis Conneau"],
        markers=["Lample and Conneau, 2019", "Lample and Conneau 2019", "Lample & Conneau, 2019", "Lample and Conneau (2019)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1906.08237",
        title="XLNet: Generalized Autoregressive Pretraining for Language Understanding",
        year=2019,
        first_author="yang",
        authors=["Zhilin Yang", "Zihang Dai", "Yiming Yang", "Jaime Carbonell", "Russ R. Salakhutdinov", "Quoc V. Le"],
        markers=["Yang et al., 2019", "Yang et al. 2019", "Yang et al. (2019", "Yang et al. (2019)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1907.11692",
        title="RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        year=2019,
        first_author="liu",
        authors=["Yinhan Liu", "Myle Ott", "Naman Goyal", "Jingfei Du", "Mandar Joshi", "Danqi Chen", "Omer Levy", "Mike Lewis", "Luke Zettlemoyer", "Veselin Stoyanov"],
        markers=["Liu et al., 2019", "Liu et al. 2019", "Liu et al. (2019", "Liu et al. (2019)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1910.10683",
        title="Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
        year=2019,
        first_author="raffel",
        authors=["Colin Raffel", "Noam Shazeer", "Adam Roberts", "Katherine Lee", "Sharan Narang", "Michael Matena", "Yanqi Zhou", "Wei Li", "Peter J. Liu"],
        markers=["Raffel et al., 2019", "Raffel et al. 2019", "Raffel et al., 2020", "Raffel et al. (2019)"],
    ),
    SeminalPaperSpec(
        arxiv_id="2003.10555",
        title="ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators",
        year=2020,
        first_author="clark",
        authors=["Kevin Clark", "Minh-Thang Luong", "Quoc V. Le", "Christopher D. Manning"],
        markers=["Clark et al., 2020", "Clark et al. 2020", "Clark et al. (2020", "Clark et al. (2020)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1909.11942",
        title="ALBERT: A Lite BERT for Self-supervised Learning of Language Representations",
        year=2019,
        first_author="lan",
        authors=["Zhenzhong Lan", "Mingda Chen", "Sebastian Goodman", "Kevin Gimpel", "Piyush Sharma", "Radu Soricut"],
        markers=["Lan et al., 2019", "Lan et al. 2019", "Lan et al. (2020", "Lan et al. (2019)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1910.13461",
        title="BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension",
        year=2019,
        first_author="lewis",
        authors=["Mike Lewis", "Yinhan Liu", "Naman Goyal", "Marjan Ghazvininejad", "Abdelrahman Mohamed", "Omer Levy", "Veselin Stoyanov", "Luke Zettlemoyer"],
        markers=["Lewis et al., 2019", "Lewis et al. 2019", "Lewis et al. (2020", "Lewis et al. (2019)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1508.05326",
        title="A large annotated corpus for learning natural language inference",
        year=2015,
        first_author="bowman",
        authors=["Samuel R. Bowman", "Gabor Angeli", "Christopher Potts", "Christopher D. Manning"],
        markers=["Bowman et al., 2015", "Bowman et al. 2015", "Bowman et al. (2015", "Bowman et al. (2015)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1704.05426",
        title="A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference",
        year=2018,
        first_author="williams",
        authors=["Adina Williams", "Nikita Nangia", "Samuel R. Bowman"],
        markers=["Williams et al., 2018", "Williams et al. 2018", "Williams et al., 2017", "Williams et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1606.05250",
        title="SQuAD: 100,000+ Questions for Machine Comprehension of Text",
        year=2016,
        first_author="rajpurkar",
        authors=["Pranav Rajpurkar", "Jian Zhang", "Konstantin Lopyrev", "Percy Liang"],
        markers=["Rajpurkar et al., 2016", "Rajpurkar et al. 2016", "Rajpurkar et al. (2016", "Rajpurkar et al. (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1508.01991",
        title="Bidirectional LSTM-CRF Models for Sequence Tagging",
        year=2015,
        first_author="huang",
        authors=["Zhiheng Huang", "Wei Xu", "Kai Yu"],
        markers=["Huang et al., 2015", "Huang et al. 2015", "Huang et al. (2015", "Huang et al. (2015)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1406.1078",
        title="Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation",
        year=2014,
        first_author="cho",
        authors=["Kyunghyun Cho", "Bart van Merrienboer", "Caglar Gulcehre", "Dzmitry Bahdanau", "Fethi Bougares", "Holger Schwenk", "Yoshua Bengio"],
        markers=["Cho et al., 2014", "Cho et al. 2014", "Cho et al. (2014", "Cho et al. (2014)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1704.04368",
        title="Get To The Point: Summarization with Pointer-Generator Networks",
        year=2017,
        first_author="see",
        authors=["Abigail See", "Peter J. Liu", "Christopher D. Manning"],
        markers=["See et al., 2017", "See et al. 2017", "See et al. (2017", "See et al. (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1509.00685",
        title="A Neural Attention Model for Abstractive Sentence Summarization",
        year=2015,
        first_author="rush",
        authors=["Alexander M. Rush", "Sumit Chopra", "Jason Weston"],
        markers=["Rush et al., 2015", "Rush et al. 2015", "Rush et al. (2015", "Rush et al. (2015)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1602.06023",
        title="Abstractive Text Summarization using Sequence-to-Sequence RNNs and Beyond",
        year=2016,
        first_author="nallapati",
        authors=["Ramesh Nallapati", "Bowen Zhou", "Caglar Gulcehre", "Bing Xiang"],
        markers=["Nallapati et al., 2016", "Nallapati et al. 2016", "Nallapati et al. (2016", "Nallapati et al. (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1603.04351",
        title="Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations",
        year=2016,
        first_author="kiperwasser",
        authors=["Eliyahu Kiperwasser", "Yoav Goldberg"],
        markers=["Kiperwasser and Goldberg 2016", "Kiperwasser and Goldberg, 2016", "Kiperwasser & Goldberg, 2016", "Kiperwasser and Goldberg (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1611.01734",
        title="Deep Biaffine Attention for Neural Dependency Parsing",
        year=2016,
        first_author="dozat",
        authors=["Timothy Dozat", "Christopher D. Manning"],
        markers=["Dozat and Manning 2016", "Dozat and Manning, 2016", "Dozat & Manning, 2016", "Dozat and Manning (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1803.05355",
        title="FEVER: a large-scale dataset for Fact Extraction and VERification",
        year=2018,
        first_author="thorne",
        authors=["James Thorne", "Andreas Vlachos", "Christos Christodoulopoulos", "Arpit Mittal"],
        markers=["Thorne et al. 2018", "Thorne et al., 2018", "Thorne et al. (2018", "Thorne et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1506.01094",
        title="Traversing Knowledge Graphs in Vector Spaces",
        year=2015,
        first_author="guu",
        authors=["Kelvin Guu", "John Miller", "Percy Liang"],
        markers=["Guu et al. 2015", "Guu et al., 2015", "Guu et al. (2015", "Guu et al. (2015)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1103.0398",
        title="Natural Language Processing (Almost) from Scratch",
        year=2011,
        first_author="collobert",
        authors=["Ronan Collobert", "Jason Weston", "Léon Bottou", "Michael Karlen", "Koray Kavukcuoglu", "Pavel Kuksa"],
        markers=["Collobert et al. 2011b", "Collobert et al. 2011", "Collobert et al., 2011", "Collobert et al. (2011)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1803.04400",
        title="Deep Energy-Based Models for Text Generation",
        year=2018,
        first_author="tu",
        authors=["Lifu Tu", "Kevin Gimpel"],
        markers=["Tu and Gimpel 2018", "Tu and Gimpel, 2018", "Tu & Gimpel, 2018", "Tu and Gimpel (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1611.00068",
        title="RNN Approaches to Text Normalization: A Challenge",
        year=2016,
        first_author="sproat",
        authors=["Richard Sproat", "Navdeep Jaitly"],
        markers=["Sproat and Jaitly 2016", "Sproat and Jaitly, 2016", "Sproat & Jaitly, 2016", "Sproat and Jaitly (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1606.01255",
        title="Most Frequent Sense Baseline Revisited",
        year=2016,
        first_author="yuan",
        authors=["Dayu Yuan", "Julian Richardson", "Ryan Doherty", "Colin Evans", "Eric Altendorf"],
        markers=["Yuan et al. 2016", "Yuan et al., 2016", "Yuan et al. (2016", "Yuan et al. (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1308.6242",
        title="NRC-Canada: Building the State-of-the-Art in Sentiment Analysis of Tweets",
        year=2013,
        first_author="mohammad",
        authors=["Saif M. Mohammad", "Svetlana Kiritchenko", "Xiaodan Zhu"],
        markers=["Mohammad et al., 2013", "Mohammad et al. 2013", "Mohammad et al. (2013", "Mohammad et al. (2013)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1408.0664",
        title="Identifying Argumentative Discourse Structures in Persuasive Essays",
        year=2014,
        first_author="stab",
        authors=["Christian Stab", "Iryna Gurevych"],
        markers=["Stab and Gurevych, 2014b", "Stab and Gurevych 2014b", "Stab & Gurevych, 2014", "Stab and Gurevych (2014b)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1412.4930",
        title="Improving Distributional Similarity with Lessons Learned from Word Embeddings",
        year=2015,
        first_author="levy",
        authors=["Omer Levy", "Yoav Goldberg", "Idan Dagan"],
        markers=["Levy et al. 2015", "Levy et al., 2015", "Levy et al. (2015", "Levy et al. (2015)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1508.06016",
        title="Sentence Compression by Deletion with LSTMs",
        year=2015,
        first_author="filippova",
        authors=["Katja Filippova", "Enrique Alfonseca", "Carlos A. Colmenares", "Lukasz Kaiser", "Oriol Vinyals"],
        markers=["Filippova et al. 2015", "Filippova et al., 2015", "Filippova et al. (2015", "Filippova et al. (2015)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1403.6179",
        title="Tailoring Continuous Word Representations for Dependency Parsing",
        year=2014,
        first_author="bansal",
        authors=["Mohit Bansal", "Kevin Gimpel", "Karen Livescu"],
        markers=["Bansal et al., 2014", "Bansal et al. 2014", "Bansal et al. (2014", "Bansal et al. (2014)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1206.5538",
        title="Inducing Crosslingual Distributed Representations using Bilingual Supervision",
        year=2012,
        first_author="klementiev",
        authors=["Alexandre Klementiev", "Ivan Titov", "Binod Bhattarai"],
        markers=["Klementiev et al., 2012", "Klementiev et al. 2012", "Klementiev et al. (2012", "Klementiev et al. (2012)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1707.07045",
        title="End-to-end Neural Coreference Resolution",
        year=2017,
        first_author="lee",
        authors=["Kenton Lee", "Luheng He", "Mike Lewis", "Luke Zettlemoyer"],
        markers=["Lee et al. 2017", "Lee et al., 2017", "Lee et al. (2017", "Lee et al. 2018", "Lee et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1707.07328",
        title="Adversarial Examples for Evaluating Reading Comprehension Systems",
        year=2017,
        first_author="jia",
        authors=["Robin Jia", "Percy Liang"],
        markers=["Jia and Liang 2017", "Jia and Liang, 2017", "Jia & Liang, 2017", "Jia and Liang (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1707.08052",
        title="Challenges in Data-to-Text Generation",
        year=2017,
        first_author="wiseman",
        authors=["Sam Wiseman", "Stuart M. Shieber", "Alexander M. Rush"],
        markers=["Wiseman et al., 2017", "Wiseman et al. 2017", "Wiseman et al. (2017", "Wiseman et al. (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1603.06015",
        title="Harnessing Deep Neural Networks with Logic Rules",
        year=2016,
        first_author="hu",
        authors=["Zhiting Hu", "Xuezhe Ma", "Zhengzhong Liu", "Eduard Hovy", "Eric Xing"],
        markers=["Hu et al. 2016", "Hu et al., 2016", "Hu et al. (2016", "Hu et al. (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1711.03953",
        title="Breaking the Softmax Bottleneck: A High-Rank RNN Language Model",
        year=2018,
        first_author="yang",
        authors=["Zhilin Yang", "Zihang Dai", "Ruslan Salakhutdinov", "William W. Cohen"],
        markers=["Yang et al. 2018", "Yang et al., 2018", "Yang et al. (2018", "Yang et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1606.01328",
        title="Deep Reinforcement Learning for Mention-Ranking Coreference Models",
        year=2016,
        first_author="clark",
        authors=["Kevin Clark", "Christopher D. Manning"],
        markers=["Clark and Manning 2016", "Clark and Manning, 2016", "Clark & Manning, 2016", "Clark and Manning (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1607.04606",
        title="Enriching Word Vectors with Subword Information",
        year=2017,
        first_author="bojanowski",
        authors=["Piotr Bojanowski", "Edouard Grave", "Armand Joulin", "Tomas Mikolov"],
        markers=["Bojanowski et al. 2017", "Bojanowski et al., 2017", "Bojanowski et al. (2017", "Bojanowski et al. (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1607.01759",
        title="Bag of Tricks for Efficient Text Classification",
        year=2016,
        first_author="joulin",
        authors=["Armand Joulin", "Edouard Grave", "Piotr Bojanowski", "Tomas Mikolov"],
        markers=["Joulin et al. 2016", "Joulin et al., 2016", "Joulin et al. (2016", "Joulin et al. (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1801.06146",
        title="Universal Language Model Fine-tuning for Text Classification",
        year=2018,
        first_author="howard",
        authors=["Jeremy Howard", "Sebastian Ruder"],
        markers=["Howard and Ruder 2018", "Howard and Ruder, 2018", "Howard & Ruder, 2018", "Howard and Ruder (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1806.05686",
        title="A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents",
        year=2018,
        first_author="cohan",
        authors=["Arman Cohan", "Franck Dernoncourt", "Doo Soon Kim", "Trung Bui", "Seokhwan Kim", "Walter Chang", "Nazli Goharian"],
        markers=["Cohan et al. 2018", "Cohan et al., 2018", "Cohan et al. (2018", "Cohan et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="2004.14963",
        title="SPECTER: Document-level Representation Learning using Citation-informed Transformers",
        year=2020,
        first_author="cohan",
        authors=["Arman Cohan", "Sergey Feldman", "Iz Beltagy", "Doug Downey", "Daniel S. Weld"],
        markers=["Cohan et al. 2020", "Cohan et al., 2020", "Cohan et al. (2020", "Cohan et al. (2020)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1903.10676",
        title="SciBERT: A Pretrained Language Model for Scientific Text",
        year=2019,
        first_author="beltagy",
        authors=["Iz Beltagy", "Kyle Lo", "Arman Cohan"],
        markers=["Beltagy et al. 2019", "Beltagy et al., 2019", "Beltagy et al. (2019", "Beltagy et al. (2019)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1804.07461",
        title="GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding",
        year=2018,
        first_author="wang",
        authors=["Alex Wang", "Amanpreet Singh", "Julian Michael", "Felix Hill", "Omer Levy", "Samuel R. Bowman"],
        markers=["Wang et al. 2018", "Wang et al., 2018", "Wang et al. (2018", "Wang et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1503.00075",
        title="Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank",
        year=2013,
        first_author="socher",
        authors=["Richard Socher", "Alex Perelygin", "Jean Wu", "Jason Chuang", "Christopher D. Manning", "Andrew Ng", "Christopher Potts"],
        markers=["Socher et al., 2013", "Socher et al. 2013", "Socher et al. (2013", "Socher et al. (2013)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1405.4053",
        title="Distributed Representations of Sentences and Documents",
        year=2014,
        first_author="le",
        authors=["Quoc V. Le", "Tomas Mikolov"],
        markers=["Le and Mikolov, 2014", "Le and Mikolov 2014", "Le & Mikolov, 2014", "Le and Mikolov (2014", "Le and Mikolov (2014)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1805.04833",
        title="CoQA: A Conversational Question Answering Challenge",
        year=2019,
        first_author="reddy",
        authors=["Siva Reddy", "Danqi Chen", "Christopher D. Manning"],
        markers=["Reddy et al. 2019", "Reddy et al., 2019", "Reddy et al. (2019", "Reddy et al. (2019)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1808.07042",
        title="HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering",
        year=2018,
        first_author="yang",
        authors=["Zhilin Yang", "Peng Qi", "Saizheng Zhang", "Yoshua Bengio", "William W. Cohen", "Russ R. Salakhutdinov", "Christopher D. Manning"],
        markers=["Yang et al. 2018", "Yang et al., 2018", "Yang et al. (2018", "Yang et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1904.09675",
        title="Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        year=2019,
        first_author="reimers",
        authors=["Nils Reimers", "Iryna Gurevych"],
        markers=["Reimers and Gurevych 2019", "Reimers and Gurevych, 2019", "Reimers & Gurevych, 2019", "Reimers and Gurevych (2019)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1905.02450",
        title="SpanBERT: Improving Pre-training by Representing and Predicting Spans",
        year=2020,
        first_author="joshi",
        authors=["Mandar Joshi", "Danqi Chen", "Yinhan Liu", "Daniel S. Weld", "Luke Zettlemoyer", "Omer Levy"],
        markers=["Joshi et al. 2020", "Joshi et al., 2020", "Joshi et al. (2020", "Joshi et al. (2020)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1910.01108",
        title="DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter",
        year=2019,
        first_author="sanh",
        authors=["Victor Sanh", "Lysandre Debut", "Julien Chaumond", "Thomas Wolf"],
        markers=["Sanh et al. 2019", "Sanh et al., 2019", "Sanh et al. (2019", "Sanh et al. (2019)"],
    ),
    SeminalPaperSpec(
        arxiv_id="2004.05150",
        title="Longformer: The Long-Document Transformer",
        year=2020,
        first_author="beltagy",
        authors=["Iz Beltagy", "Matthew E. Peters", "Arman Cohan"],
        markers=["Beltagy et al. 2020", "Beltagy et al., 2020", "Beltagy et al. (2020", "Beltagy et al. (2020)"],
    ),
    SeminalPaperSpec(
        arxiv_id="2004.08484",
        title="Big Bird: Transformers for Longer Sequences",
        year=2020,
        first_author="zaheer",
        authors=["Manzil Zaheer", "Guru Guruganesh", "Kumar Avinava Dubey", "Joshua Ainslie", "Chris Alberti", "Santiago Ontanon", "Philip Pham", "Anirudh Ravula", "Qifan Wang", "Li Yang", "Amr Ahmed"],
        markers=["Zaheer et al. 2020", "Zaheer et al., 2020", "Zaheer et al. (2020", "Zaheer et al. (2020)"],
    ),
    SeminalPaperSpec(
        arxiv_id="2005.14165",
        title="Language Models are Few-Shot Learners",
        year=2020,
        first_author="brown",
        authors=["Tom B. Brown", "Benjamin Mann", "Nick Ryder", "Melanie Subbiah", "Jared Kaplan", "Prafulla Dhariwal", "Arvind Neelakantan", "Pranav Shyam", "Girish Sastry", "Amanda Askell"],
        markers=["Brown et al. 2020", "Brown et al., 2020", "Brown et al. (2020", "Brown et al. (2020)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1905.00537",
        title="SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems",
        year=2019,
        first_author="wang",
        authors=["Alex Wang", "Yada Pruksachatkun", "Nikita Nangia", "Amanpreet Singh", "Julian Michael", "Felix Hill", "Omer Levy", "Samuel R. Bowman"],
        markers=["Wang et al. 2019", "Wang et al., 2019", "Wang et al. (2019", "Wang et al. (2019)"],
    ),
    SeminalPaperSpec(
        arxiv_id="2006.04768",
        title="DeBERTa: Decoding-enhanced BERT with Disentangled Attention",
        year=2021,
        first_author="he",
        authors=["Pengcheng He", "Xiaodong Liu", "Jianfeng Gao", "Weizhu Chen"],
        markers=["He et al. 2021", "He et al., 2021", "He et al. (2021", "He et al. (2021)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1904.08375",
        title="BioBERT: a pre-trained biomedical language representation model for biomedical text mining",
        year=2020,
        first_author="lee",
        authors=["Jinhyuk Lee", "Wonjin Yoon", "Sungdong Kim", "Donghyeon Kim", "Sunkyu Kim", "Chan Ho So", "Jaewoo Kang"],
        markers=["Lee et al. 2020", "Lee et al., 2020", "Lee et al. (2020", "Lee et al. (2020)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1910.03771",
        title="HuggingFace Transformers: State-of-the-art Natural Language Processing",
        year=2020,
        first_author="wolf",
        authors=["Thomas Wolf", "Lysandre Debut", "Victor Sanh", "Julien Chaumond", "Clement Delangue", "Alexander M. Rush"],
        markers=["Wolf et al. 2020", "Wolf et al., 2020", "Wolf et al. (2020", "Wolf et al. (2020)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1909.03409",
        title="Dense Passage Retrieval for Open-Domain Question Answering",
        year=2020,
        first_author="karpukhin",
        authors=["Vladimir Karpukhin", "Barlas Oguz", "Sewon Min", "Patrick Lewis", "Ledell Wu", "Sergey Edunov", "Danqi Chen", "Wen-tau Yih"],
        markers=["Karpukhin et al. 2020", "Karpukhin et al., 2020", "Karpukhin et al. (2020", "Karpukhin et al. (2020)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1910.14659",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        year=2020,
        first_author="lewis",
        authors=["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "Fabio Petroni", "Vladimir Karpukhin", "Naman Goyal", "Heinrich Küttler", "Mike Lewis"],
        markers=["Lewis et al. 2020", "Lewis et al., 2020", "Lewis et al. (2020", "Lewis et al. (2020)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1707.03389",
        title="Learned in Translation: Contextualized Word Vectors",
        year=2017,
        first_author="mccann",
        authors=["Bryan McCann", "James Bradbury", "Caiming Xiong", "Richard Socher"],
        markers=["McCann et al. 2017", "McCann et al., 2017", "McCann et al. (2017", "McCann et al. (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1803.02155",
        title="Contextual String Embeddings for Sequence Labeling",
        year=2018,
        first_author="akbik",
        authors=["Alan Akbik", "Duncan Blythe", "Roland Vollgraf"],
        markers=["Akbik et al. 2018", "Akbik et al., 2018", "Akbik et al. (2018", "Akbik et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1804.00247",
        title="BERT Rediscovers the Classical NLP Pipeline",
        year=2019,
        first_author="tenney",
        authors=["Ian Tenney", "Dipanjan Das", "Ellie Pavlick"],
        markers=["Tenney et al. 2019", "Tenney et al., 2019", "Tenney et al. (2019", "Tenney et al. (2019)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1907.01469",
        title="Unsupervised Cross-lingual Representation Learning at Scale",
        year=2020,
        first_author="conneau",
        authors=["Alexis Conneau", "Kartikay Khandelwal", "Naman Goyal", "Vishrav Chaudhary", "Guillaume Wenzek", "Francisco Guzmán", "Edouard Grave", "Myle Ott", "Luke Zettlemoyer", "Veselin Stoyanov"],
        markers=["Conneau et al. 2020", "Conneau et al., 2020", "Conneau et al. (2020", "Conneau et al. (2020)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1909.08471",
        title="Self-Attention with Relative Position Representations",
        year=2018,
        first_author="shaw",
        authors=["Peter Shaw", "Jakob Uszkoreit", "Ashish Vaswani"],
        markers=["Shaw et al. 2018", "Shaw et al., 2018", "Shaw et al. (2018", "Shaw et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="2002.08909",
        title="Don't Stop Pretraining: Make Language Models Adapt to Domains and Tasks",
        year=2020,
        first_author="gururangan",
        authors=["Suchin Gururangan", "Ana Marasović", "Swayam Bhagavatula", "Iz Beltagy", "Doug Downey", "Matthew E. Peters"],
        markers=["Gururangan et al. 2020", "Gururangan et al., 2020", "Gururangan et al. (2020", "Gururangan et al. (2020)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1506.06726",
        title="Skip-Thought Vectors",
        year=2015,
        first_author="kiros",
        authors=["Ryan Kiros", "Yukun Zhu", "Ruslan Salakhutdinov", "Richard S. Zemel", "Antonio Torralba", "Raquel Urtasun", "Sanja Fidler"],
        markers=["Kiros et al. 2015", "Kiros et al., 2015", "Kiros et al. (2015", "Kiros et al. (2015)", "Kiros et al. 2014", "Kiros et al., 2014"],
    ),
    SeminalPaperSpec(
        arxiv_id="1505.00468",
        title="VQA: Visual Question Answering",
        year=2015,
        first_author="antol",
        authors=["Stanislaw Antol", "Aishwarya Agrawal", "Jiasen Lu", "Margaret Mitchell", "Dhruv Batra", "C. Lawrence Zitnick", "Devi Parikh"],
        markers=["Antol et al. 2015", "Antol et al., 2015", "Antol et al. (2015", "Antol et al. (2015)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1506.03134",
        title="Pointer Networks",
        year=2015,
        first_author="vinyals",
        authors=["Oriol Vinyals", "Meire Fortunato", "Navdeep Jaitly"],
        markers=["Vinyals et al. 2015", "Vinyals et al., 2015", "Vinyals et al. (2015", "Vinyals et al. (2015)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1503.08895",
        title="End-To-End Memory Networks",
        year=2015,
        first_author="sukhbaatar",
        authors=["Sainbayar Sukhbaatar", "Arthur Szlam", "Jason Weston", "Rob Fergus"],
        markers=["Sukhbaatar et al. 2015", "Sukhbaatar et al., 2015", "Sukhbaatar et al. (2015", "Sukhbaatar et al. (2015)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1703.03130",
        title="A Structured Self-Attentive Sentence Embedding",
        year=2017,
        first_author="lin",
        authors=["Zhouhan Lin", "Minwei Feng", "Cicero Nogueira dos Santos", "Mo Yu", "Bing Xiang", "Bowen Zhou", "Yoshua Bengio"],
        markers=["Lin et al. 2017", "Lin et al., 2017", "Lin et al. (2017", "Lin et al. (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1507.04808",
        title="Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models",
        year=2016,
        first_author="serban",
        authors=["Iulian V. Serban", "Alessandro Sordoni", "Yoshua Bengio", "Aaron Courville", "Joelle Pineau"],
        markers=["Serban et al. 2016", "Serban et al., 2016", "Serban et al. (2016", "Serban et al. (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1611.04558",
        title="Google Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation",
        year=2017,
        first_author="johnson",
        authors=["Melvin Johnson", "Mike Schuster", "Quoc V. Le", "Maxim Krikun", "Yonghui Wu", "Zhifeng Chen", "Nikhil Thorat", "Fernanda Viégas", "Martin Wattenberg", "Greg Corrado", "Macduff Hughes", "Jeffrey Dean"],
        markers=["Johnson et al., 2017", "Johnson et al. 2017", "Johnson et al. (2017", "Johnson et al. (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1603.06021",
        title="A Fast Unified Model for Parsing and Sentence Understanding",
        year=2016,
        first_author="bowman",
        authors=["Samuel R. Bowman", "Jon Gauthier", "Abhinav Rastogi", "Raghav Gupta", "Christopher D. Manning", "Christopher Potts"],
        markers=["Bowman et al., 2016", "Bowman et al. 2016", "Bowman et al. (2016", "Bowman et al. (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1511.08198",
        title="Towards Universal Paraphrastic Sentence Embeddings",
        year=2016,
        first_author="wieting",
        authors=["John Wieting", "Mohit Bansal", "Kevin Gimpel", "Karen Livescu"],
        markers=["Wieting et al. 2016", "Wieting et al., 2016", "Wieting et al. (2016", "Wieting et al. (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1604.02201",
        title="Transfer Learning for Low-Resource Neural Machine Translation",
        year=2016,
        first_author="zoph",
        authors=["Barret Zoph", "Deniz Yuret", "Jonathan May", "Kevin Knight"],
        markers=["Zoph et al., 2016", "Zoph et al. 2016", "Zoph et al. (2016", "Zoph et al. (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1603.03827",
        title="Sequential Short-Text Classification with Recurrent and Convolutional Neural Networks",
        year=2016,
        first_author="lee",
        authors=["Ji Young Lee", "Franck Dernoncourt"],
        markers=["Lee and Dernoncourt, 2016", "Lee and Dernoncourt 2016", "Lee & Dernoncourt, 2016", "Lee and Dernoncourt (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1408.3456",
        title="SimLex-999: Evaluating Semantic Models with Genuine Similarity Estimation",
        year=2015,
        first_author="hill",
        authors=["Felix Hill", "Roi Reichart", "Anna Korhonen"],
        markers=["Hill et al., 2015", "Hill et al. 2015", "Hill et al. (2015", "Hill et al. (2015)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1402.1794",
        title="Improving Vector Space Word Representations Using Multilingual Correlation",
        year=2014,
        first_author="faruqui",
        authors=["Manaal Faruqui", "Chris Dyer"],
        markers=["Faruqui and Dyer 2014", "Faruqui and Dyer, 2014", "Faruqui & Dyer, 2014", "Faruqui and Dyer (2014)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1903.07785",
        title="Cloze-driven Pre-training of Self-attention Networks",
        year=2019,
        first_author="baevski",
        authors=["Alexei Baevski", "Sergey Edunov", "Yinhan Liu", "Luke Zettlemoyer", "Michael Auli"],
        markers=["Baevski et al., 2019b", "Baevski et al. 2019", "Baevski et al. (2019", "Baevski et al. (2019)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1808.08745",
        title="Don't Give Me the Details, Just the Summary! Topic-Conditioned Convolutional Neural Networks for Extreme Summarization",
        year=2018,
        first_author="narayan",
        authors=["Shashi Narayan", "Shay B. Cohen", "Mirella Lapata"],
        markers=["Narayan et al., 2018b", "Narayan et al. 2018", "Narayan et al. (2018", "Narayan et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1703.04009",
        title="Automated Hate Speech Detection and the Problem of Offensive Language",
        year=2017,
        first_author="davidson",
        authors=["Thomas Davidson", "Dana Warmsley", "Michael Macy", "Ingmar Weber"],
        markers=["Davidson et al. 2017", "Davidson et al., 2017", "Davidson et al. (2017", "Davidson et al. (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1801.07005",
        title="Co-Stack Residual Affinity Networks with Multi-level Attention for Textual Item Recommendation",
        year=2018,
        first_author="tay",
        authors=["Yi Tay", "Anh Tuan Luu", "Siu Cheung Hui"],
        markers=["Tay et al., 2018", "Tay et al. 2018", "Tay et al. (2018", "Tay et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1808.06640",
        title="Adversarial Removal of Demographic Information from Text Data",
        year=2018,
        first_author="elazar",
        authors=["Yanai Elazar", "Yoav Goldberg"],
        markers=["Elazar and Goldberg 2018", "Elazar and Goldberg, 2018", "Elazar & Goldberg, 2018", "Elazar and Goldberg (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1604.08634",
        title="Capturing Semantic Similarity for Entity Linking with Multi-scale Convolutional Neural Networks",
        year=2016,
        first_author="francis-landau",
        authors=["Matthew Francis-Landau", "Trevor Darrell", "Dan Klein"],
        markers=["Francis-Landau et al., 2016", "Francis-Landau et al. 2016", "Francis-Landau et al. (2016", "Francis-Landau et al. (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1706.08502",
        title="Natural Language Does Not Emerge 'Naturally' in Multi-Agent Dialog",
        year=2017,
        first_author="kottur",
        authors=["Satwik Kottur", "José M. F. Moura", "Stefan Lee", "Dhruv Batra"],
        markers=["Kottur et al. 2017", "Kottur et al., 2017", "Kottur et al. (2017", "Kottur et al. (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1904.09324",
        title="Constant-Time Machine Translation with Conditional Masked Language Models",
        year=2019,
        first_author="ghazvininejad",
        authors=["Marjan Ghazvininejad", "Omer Levy", "Yinhan Liu", "Luke Zettlemoyer"],
        markers=["Ghazvininejad et al., 2019", "Ghazvininejad et al. 2019", "Ghazvininejad et al. (2019", "Ghazvininejad et al. (2019)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1804.09301",
        title="Gender Bias in Coreference Resolution: Evaluation and Debias Methods",
        year=2018,
        first_author="rudinger",
        authors=["Rachel Rudinger", "Jason Naradowsky", "Brian Leonard", "Benjamin Van Durme"],
        markers=["Rudinger et al., 2018", "Rudinger et al. 2018", "Rudinger et al. (2018", "Rudinger et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1806.09059",
        title="Improving Text-to-SQL Evaluation Methodology for Complex and Diverse Datasets",
        year=2018,
        first_author="finegan-dollak",
        authors=["Catherine Finegan-Dollak", "Jonathan K. Kummerfeld", "Li Zhang", "Priyanka Ramanathan", "Sida I. Wang", "Dragomir Radev"],
        markers=["Finegan-Dollak et al. 2018", "Finegan-Dollak et al., 2018", "Finegan-Dollak et al. (2018", "Finegan-Dollak et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1803.06535",
        title="Dear Sir or Madam, May I Introduce the GYAFC Dataset: Evaluating Formality Transfer with Shallow Neural Networks",
        year=2018,
        first_author="rao",
        authors=["Sudha Rao", "Joel Tetreault"],
        markers=["Rao and Tetreault 2018", "Rao and Tetreault, 2018", "Rao & Tetreault, 2018", "Rao and Tetreault (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1804.09831",
        title="Information-Theoretic Analysis of Generalization Capability of Neural Networks",
        year=2018,
        first_author="gong",
        authors=["Chen Gong", "Tongliang Liu", "Dacheng Tao"],
        markers=["Gong et al. 2018", "Gong et al., 2018", "Gong et al. (2018", "Gong et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1702.01417",
        title="All-but-the-Top: Simple and Effective Postprocessing for Word Representations",
        year=2018,
        first_author="mu",
        authors=["Jiaqi Mu", "Suma Bhat", "Pramod Viswanath"],
        markers=["Mu and Viswanath 2018", "Mu and Viswanath, 2018", "Mu & Viswanath, 2018", "Mu and Viswanath (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1406.1837",
        title="A Fast and Accurate Dependency Parser using Neural Networks",
        year=2014,
        first_author="chen",
        authors=["Danqi Chen", "Christopher D. Manning"],
        markers=["Chen and Manning 2014", "Chen and Manning, 2014", "Chen & Manning, 2014", "Chen and Manning (2014)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1609.08144",
        title="Google Neural Machine Translation System: Bridging the Gap between Human and Machine Translation",
        year=2016,
        first_author="wu",
        authors=["Yonghui Wu", "Mike Schuster", "Zhifeng Chen", "Quoc V. Le", "Mohammad Norouzi", "Wolfgang Macherey", "Maxim Krikun", "Yuan Cao", "Qin Gao", "Klaus Macherey", "Jeff Klingner", "Apurva Shah", "Melvin Johnson", "Xiaobing Liu", "Łukasz Kaiser", "Stephan Gouws", "Yoshikiyo Kato", "Taku Kudo", "Hideto Kazawa", "Keith Stevens", "George Kurian", "Nishant Patil", "Wei Wang", "Cliff Young", "Jason Smith", "Jason Riesa", "Alex Rudnick", "Oriol Vinyals", "Greg Corrado", "Macduff Hughes", "Jeffrey Dean"],
        markers=["Wu et al., 2016", "Wu et al. 2016", "Wu et al. (2016", "Wu et al. (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1804.08201",
        title="Challenger: A Deep Reinforcement Learning System for Reading Comprehension",
        year=2018,
        first_author="ma",
        authors=["Kaixin Ma", "Tomasz Jurczyk", "Jinho D. Choi"],
        markers=["Ma, Jurczyk, and Choi 2018", "Ma et al., 2018", "Ma et al. 2018"],
    ),
    SeminalPaperSpec(
        arxiv_id="1508.01585",
        title="Applying Deep Learning to Answer Selection: A Study and An Open Task",
        year=2015,
        first_author="feng",
        authors=["Minwei Feng", "Bing Xiang", "Michael R. Glass", "Lidan Wang", "Bowen Zhou"],
        markers=["Feng et al., 2015", "Feng et al. 2015", "Feng et al. (2015", "Feng et al. (2015)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1703.10186",
        title="Colors in Context: A Pragmatic Neural Model for Grounded Language Understanding",
        year=2017,
        first_author="monroe",
        authors=["Will Monroe", "Robert X.D. Hawkins", "Noah D. Goodman", "Christopher Potts"],
        markers=["Monroe et al. 2017", "Monroe et al., 2017", "Monroe et al. (2017", "Monroe et al. (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1602.07332",
        title="Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations",
        year=2017,
        first_author="krishna",
        authors=["Ranjay Krishna", "Yuke Zhu", "Oliver Groth", "Justin Johnson", "Kenji Hata", "Joshua Kravitz", "Stephanie Chen", "Yannis Kalantidis", "Li-Jia Li", "David A. Shamma", "Michael S. Bernstein", "Li Fei-Fei"],
        markers=["Krishna et al. 2016", "Krishna et al., 2016", "Krishna et al. (2016", "Krishna et al. (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1605.02367",
        title="Generating English from Abstract Meaning Representations",
        year=2016,
        first_author="pourdamghani",
        authors=["Nima Pourdamghani", "Kevin Knight", "Ulf Hermjakob"],
        markers=["Pourdamghani et al. 2016", "Pourdamghani et al., 2016", "Pourdamghani et al. (2016", "Pourdamghani et al. (2016)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1805.06297",
        title="A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings",
        year=2018,
        first_author="artetxe",
        authors=["Mikel Artetxe", "Gorka Labaka", "Eneko Agirre"],
        markers=["Artetxe et al. 2018b", "Artetxe et al. 2018", "Artetxe et al., 2018b", "Artetxe et al., 2018"],
    ),
    SeminalPaperSpec(
        arxiv_id="1805.03459",
        title="Isomorphic Transfer of Syntactic Dependencies in Cross-Lingual AMR Parsing",
        year=2018,
        first_author="ponti",
        authors=["Edoardo Maria Ponti", "Roi Reichart", "Anna Korhonen"],
        markers=["Ponti et al., 2018", "Ponti et al. 2018", "Ponti et al. (2018", "Ponti et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1809.08375",
        title="Deep Biaffine Attention for Neural Dependency Parsing",
        year=2018,
        first_author="kasai",
        authors=["Jungo Kasai", "Kun Qian", "Gurpreet Singh"],
        markers=["Kasai et al., 2018", "Kasai et al. 2018", "Kasai et al. (2018", "Kasai et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1406.3676",
        title="Question Answering with Subgraph Embeddings",
        year=2014,
        first_author="bordes",
        authors=["Antoine Bordes", "Suma Chopra", "Jason Weston"],
        markers=["Bordes et al., 2014b", "Bordes et al. 2014", "Bordes et al., 2014"],
    ),
    SeminalPaperSpec(
        arxiv_id="1508.04025",
        title="Effective Approaches to Attention-based Neural Machine Translation",
        year=2015,
        first_author="luong",
        authors=["Minh-Thang Luong", "Hieu Pham", "Christopher D. Manning"],
        markers=["Luong et al., 2015", "Luong et al. 2015", "Luong et al. (2015", "Luong et al. (2015)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1704.07760",
        title="Acquiring Knowledge from Pre-trained Word Embeddings",
        year=2017,
        first_author="shwartz",
        authors=["Vered Shwartz", "Yoav Goldberg", "Ido Dagan"],
        markers=["Shwartz et al., 2017", "Shwartz et al. 2017", "Shwartz et al. (2017", "Shwartz et al. (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1506.06518",
        title="ClusType: Effective Entity Recognition and Typing by Relation Phrase-based Clustering",
        year=2015,
        first_author="ren",
        authors=["Xiang Ren", "Ahmed El-Kishky", "Chi Wang", "Fangbo Tao", "Clare R. Voss", "Jiawei Han"],
        markers=["Ren et al., 2015", "Ren et al. 2015", "Ren et al. (2015", "Ren et al. (2015)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1706.01427",
        title="Gated Self-Matching Networks for Reading Comprehension and Question Answering",
        year=2017,
        first_author="wang",
        authors=["Wenhui Wang", "Nan Yang", "Furu Wei", "Baobao Chang", "Ming Zhou"],
        markers=["Wang et al. 2017", "Wang et al., 2017", "Wang et al. (2017", "Wang et al. (2017)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1506.08941",
        title="Language Understanding for Text-based Games using Deep Reinforcement Learning",
        year=2015,
        first_author="narasimhan",
        authors=["Karthik Narasimhan", "Tejas Kulkarni", "Regina Barzilay"],
        markers=["Narasimhan et al. 2015", "Narasimhan et al., 2015", "Narasimhan et al. (2015", "Narasimhan et al. (2015)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1906.03404",
        title="Searching for Effective Neural Extractive Summarization",
        year=2019,
        first_author="zhong",
        authors=["Ming Zhong", "Pengfei Liu", "Danqing Wang", "Xipeng Qiu", "Xuanjing Huang"],
        markers=["Zhong et al., 2019", "Zhong et al. 2019", "Zhong et al. (2019", "Zhong et al. (2019)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1803.09337",
        title="Text Segmentation as a Supervised Learning Task",
        year=2018,
        first_author="koshorek",
        authors=["Omri Koshorek", "Adir Cohen", "Noam Mor", "Michael Rotman", "Jonathan Berant"],
        markers=["Koshorek et al. 2018", "Koshorek et al., 2018", "Koshorek et al. (2018", "Koshorek et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1510.06168",
        title="Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Recurrent Neural Network",
        year=2015,
        first_author="wang",
        authors=["Peilu Wang", "Yao Qian", "Frank K. Soong", "Lei He", "Hai Zhao"],
        markers=["Wang et al. 2015", "Wang et al., 2015", "Wang et al. (2015", "Wang et al. (2015)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1511.00060",
        title="Top-down Tree Long Short-Term Memory Networks",
        year=2016,
        first_author="zhang",
        authors=["Xingxing Zhang", "Liang Lu", "Mirella Lapata"],
        markers=["Zhang et al. 2016b", "Zhang et al., 2016b", "Zhang et al. 2016", "Zhang et al., 2016", "Zhang et al. (2016b)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1611.08669",
        title="Visual Dialog",
        year=2017,
        first_author="das",
        authors=["Abhishek Das", "Satwik Kottur", "Khushi Gupta", "Avi Singh", "Deshraj Yadav", "Stefan Lee", "José M. F. Moura", "Devi Parikh", "Dhruv Batra"],
        markers=["Das et al. 2017b", "Das et al., 2017b", "Das et al. 2017", "Das et al., 2017", "Das et al. (2017b)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1804.04169",
        title="T-REx: A Large Scale Alignment of Natural Language with Wikidata",
        year=2018,
        first_author="elsahar",
        authors=["Hady Elsahar", "Pavlos Vougiouklis", "Arslen Remaci", "Christophe Gravier", "Jonathon Hare", "Frederique Laforest", "Elena Simperl"],
        markers=["Elsahar et al. 2018", "Elsahar et al., 2018", "Elsahar et al. (2018", "Elsahar et al. (2018)"],
    ),
    SeminalPaperSpec(
        arxiv_id="1902.09666",
        title="Predicting the Type and Target of Offensive Posts in Social Media",
        year=2019,
        first_author="zampieri",
        authors=["Marcos Zampieri", "Shervin Malmasi", "Preslav Nakov", "Sara Rosenthal", "Noureldin Farooqi", "Ritesh Kumar"],
        markers=["Zampieri et al., 2019a", "Zampieri et al. 2019a", "Zampieri et al. 2019", "Zampieri et al., 2019", "Zampieri et al. (2019a)"],
    ),
]

def resolve_and_cache_seminal_papers(
    s2_client: SemanticScholarClient,
    catalog: List[SeminalPaperSpec],
) -> int:
    """Consulta Semantic Scholar y pre-calienta la base SQLite s2_cache.db para cada artículo seminal.

    Registra múltiples claves de búsqueda (search:marker, search:author year, id:arxiv:...)
    para garantizar aciertos O(1) en la posterior ejecución de mining.py.
    """
    logger.info("Iniciando pre-calentamiento de Semantic Scholar para %d artículos seminales...", len(catalog))
    resolved_count = 0

    for idx, spec in enumerate(catalog, 1):
        arxiv_target = f"ARXIV:{spec.arxiv_id}"
        logger.info("[%d/%d] Resolviendo seminal: %s (%s)...", idx, len(catalog), spec.title, arxiv_target)

        # 1. Consultar S2 por arXiv ID oficial
        meta = s2_client.get_paper_by_id(arxiv_target)
        if not meta:
            logger.warning("No se pudo recuperar metadata de S2 para %s; construyendo fallback local...", arxiv_target)
            meta = S2PaperMetadata(
                paper_id=f"seminal_{spec.arxiv_id.replace('.', '_')}",
                title=spec.title,
                abstract="Seminal foundational publication in NLP and artificial intelligence.",
                year=spec.year,
                arxiv_id=spec.arxiv_id,
                authors=spec.authors,
            )

        # 2. Guardar en caché bajo múltiples variantes para acceso transparente desde mining.py
        now = time.time()
        payload_str = json.dumps(meta.model_dump())

        keys_to_index = set()
        # Por id oficial
        keys_to_index.add(f"id:{arxiv_target.lower()}")
        keys_to_index.add(f"id:arxiv:{spec.arxiv_id.lower()}")
        if meta.paper_id:
            keys_to_index.add(f"id:{meta.paper_id.lower()}")

        # Por búsqueda exacta de autor + año
        keys_to_index.add(f"search:{spec.first_author.lower()} {spec.year}")
        keys_to_index.add(f"search:{spec.title.lower()}")

        # Por cada marcador textual de citación en MultiCite
        for m in spec.markers:
            clean_m = re.sub(r"^[(\[\s]+", "", m)
            clean_m = re.sub(r"[)\]\s,;:]+$", "", clean_m).strip().lower()
            keys_to_index.add(f"search:{clean_m}")
            # Variante sin comas
            keys_to_index.add(f"search:{clean_m.replace(',', '')}")

        if s2_client.config.cache_path:
            with sqlite3.connect(s2_client.config.cache_path) as conn:
                for k in keys_to_index:
                    conn.execute(
                        "INSERT OR REPLACE INTO paper_cache (cache_key, paper_data, created_at) VALUES (?, ?, ?)",
                        (k, payload_str, now),
                    )
                conn.commit()

        resolved_count += 1
        time.sleep(0.5)  # Pausa respetuosa para evitar 429

    logger.info("Pre-calentamiento S2 completado exitosamente: %d artículos indexados en caché.", resolved_count)
    return resolved_count


def parse_ar5iv_html_to_structured_paper(
    arxiv_id: str,
    title: str,
    abstract: str,
    authors: List[str],
) -> Optional[StructuredPaper]:
    """Descarga y estructura el texto completo de un artículo desde ar5iv.labs.arxiv.org."""
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})

    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        logger.warning("Fallo al descargar ar5iv para %s (%s): %s", arxiv_id, url, exc)
        return None

    raw_sections = re.findall(r'<section[^>]*class="[^"]*ltx_section[^"]*"[^>]*>(.*?)</section>', html, re.DOTALL)
    sections: List[StructuredSection] = []
    sec_idx = 0
    sent_idx = 0

    for s_html in raw_sections:
        hm = re.search(r'<h[23][^>]*class="[^"]*ltx_title[^"]*"[^>]*>(.*?)</h[23]>', s_html, re.DOTALL)
        heading = re.sub(r"<[^>]+>", "", hm.group(1)).strip() if hm else f"Section {sec_idx+1}"

        if heading.lower() in ["references", "bibliography", "acknowledgements", "acknowledgments"]:
            continue

        paras = re.findall(r'<p[^>]*class="[^"]*ltx_p[^"]*"[^>]*>(.*?)</p>', s_html, re.DOTALL)
        sec_sentences: List[StructuredSentence] = []

        for p in paras:
            p_text = re.sub(r"<[^>]+>", " ", p)
            cleaned_p = clean_whitespace(p_text)
            if not cleaned_p:
                continue

            # Segmentación atómica profesional con protección de abreviaturas científicas (e.g., et al., i.e.)
            raw_sents = split_sentences(cleaned_p)
            for s in raw_sents:
                s_clean = s.strip()
                if len(s_clean) > 15:
                    sec_sentences.append(StructuredSentence(sent_idx=sent_idx, text=s_clean, citations=[]))
                    sent_idx += 1

        if sec_sentences:
            sections.append(StructuredSection(section_idx=sec_idx, heading=heading, sentences=sec_sentences))
            sec_idx += 1

    if not sections:
        logger.warning("No se pudieron extraer secciones útiles para %s.", arxiv_id)
        return None

    return StructuredPaper(
        paper_id=arxiv_id,
        id_source="arxiv_seminal",
        title=title,
        abstract=abstract,
        discipline="Computer Science",
        categories=["cs.CL", "cs.LG", "cs.AI"],
        sections=sections,
        bibliography={},
    )


def ingest_missing_seminal_papers(
    papers_file: Path,
    catalog: List[SeminalPaperSpec],
) -> int:
    """Verifica cuáles artículos seminales faltan en papers.jsonl y los anexa estructurados."""
    existing_ids: Set[str] = set()
    if papers_file.exists():
        logger.info("Identificando artículos ya estructurados en %s...", papers_file)
        with open(papers_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_ids.add(data.get("paper_id"))
                except Exception:
                    pass

    logger.info("Total de artículos existentes en papers.jsonl: %d", len(existing_ids))
    missing = [spec for spec in catalog if spec.arxiv_id not in existing_ids]
    logger.info("Artículos seminales faltantes por estructurar: %d/%d", len(missing), len(catalog))

    if not missing:
        logger.info("Todos los artículos seminales del catálogo ya están estructurados en papers.jsonl.")
        return 0

    added_count = 0
    with open(papers_file, "a", encoding="utf-8") as f_out:
        for idx, spec in enumerate(missing, 1):
            logger.info("[%d/%d] Ingestando y estructurando seminal: %s (%s)...", idx, len(missing), spec.title, spec.arxiv_id)
            paper = parse_ar5iv_html_to_structured_paper(
                arxiv_id=spec.arxiv_id,
                title=spec.title,
                abstract="Seminal foundational article in NLP and AI.",
                authors=spec.authors,
            )
            if paper:
                f_out.write(json.dumps(paper.model_dump()) + "\n")
                f_out.flush()
                added_count += 1
                logger.info("  -> Exitosamente anexado con %d secciones y %d oraciones.", len(paper.sections), sum(len(s.sentences) for s in paper.sections))
            else:
                logger.warning("  -> No fue posible estructurar %s.", spec.arxiv_id)
            time.sleep(1.0)  # Evitar saturar ar5iv

    logger.info("Ingestión completada: %d nuevos artículos seminales anexados a %s.", added_count, papers_file)
    return added_count


def evaluate_multicite_coverage(
    multicite_dir: Path,
    papers_file: Path,
    s2_cache_path: Path,
) -> None:
    """Evalúa la cobertura de citas resueltas y con chunks disponibles en MultiCite."""
    logger.info("Evaluando cobertura final sobre MultiCite...")

    known_arxiv_ids: Set[str] = set()
    if papers_file.exists():
        with open(papers_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    known_arxiv_ids.add(json.loads(line).get("paper_id"))
                except Exception:
                    pass

    cached_queries: Dict[str, Any] = {}
    if s2_cache_path.exists():
        with sqlite3.connect(s2_cache_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT cache_key, paper_data FROM paper_cache")
            for k, val in cursor.fetchall():
                try:
                    data = json.loads(val)
                    if not data.get("__not_found__"):
                        cached_queries[k] = data
                except Exception:
                    pass

    total_citations = 0
    resolved_in_cache = 0
    has_chunks_count = 0

    for split in ["train", "dev", "test"]:
        p = multicite_dir / f"{split}.json"
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            total_citations += 1
            t = item.get("x", "")
            if isinstance(t, list):
                t = " ".join(t)
            cites = re.findall(r"<cite>(.*?)</cite>", t, re.DOTALL)
            if not cites:
                continue
            raw_m = cites[0]
            clean_m = re.sub(r"^[(\[\s]+", "", raw_m)
            clean_m = re.sub(r"[)\]\s,;:]+$", "", clean_m).strip().lower()

            cache_key = f"search:{clean_m}"
            alt_key = f"search:{clean_m.replace(',', '')}"

            paper_meta = cached_queries.get(cache_key) or cached_queries.get(alt_key)
            if paper_meta:
                resolved_in_cache += 1
                aid = paper_meta.get("arxiv_id")
                if aid and aid in known_arxiv_ids:
                    has_chunks_count += 1

    logger.info("=== REPORTE DE COBERTURA MULTICITE ===")
    logger.info("Total de instancias en MultiCite: %d", total_citations)
    logger.info("Instancias resueltas en caché S2 (Config A): %d (%.1f%%)", resolved_in_cache, (resolved_in_cache / total_citations) * 100)
    logger.info("Instancias con texto completo en disco (Config B - Chunks): %d (%.1f%%)", has_chunks_count, (has_chunks_count / total_citations) * 100)


def load_seminal_catalog(catalog_path: Optional[Path] = None) -> List[SeminalPaperSpec]:
    """Carga el catálogo de especificaciones seminales desde un archivo JSON opcional, o retorna el catálogo canónico embebido."""
    if catalog_path and catalog_path.exists():
        logger.info("Cargando catálogo seminal personalizado desde %s...", catalog_path)
        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [SeminalPaperSpec(**item) for item in data]
    logger.info("Utilizando catálogo seminal estático canónico embebido (%d artículos)...", len(SEMINAL_CATALOG))
    return SEMINAL_CATALOG


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Pre-calentamiento seminal de MultiCite: resolución en S2 e ingestión estructurada en papers.jsonl."
    )
    parser.add_argument("--resolve", action="store_true", help="Ejecutar pre-calentamiento de Semantic Scholar hacia s2_cache.db")
    parser.add_argument("--ingest", action="store_true", help="Descargar y anexar artículos seminales faltantes a papers.jsonl")
    parser.add_argument("--evaluate", action="store_true", help="Evaluar cobertura actual sobre MultiCite")
    parser.add_argument("--all", action="store_true", help="Ejecutar resolución, ingestión y evaluación completa")
    parser.add_argument("--papers-file", type=Path, default=Path("data/structured/papers.jsonl"), help="Ruta a papers.jsonl")
    parser.add_argument("--s2-cache-path", type=Path, default=Path("data/raw/s2_cache.db"), help="Ruta a s2_cache.db")
    parser.add_argument("--multicite-dir", type=Path, default=Path("data/raw/multicite"), help="Ruta al directorio MultiCite")
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=None,
        help="Ruta opcional a un archivo JSON con especificaciones seminales para sobreescribir el catálogo embebido",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada de ejecución del pre-calentamiento seminal."""
    args = parse_args()

    # Si no se indica ninguna bandera, por defecto se ejecuta --all
    if not (args.resolve or args.ingest or args.evaluate or args.all):
        args.all = True

    config = S2ClientConfig(cache_path=args.s2_cache_path)
    client = SemanticScholarClient(config)
    catalog = load_seminal_catalog(args.catalog_path)
    logger.info("Catálogo activo contiene %d artículos seminales.", len(catalog))

    if args.resolve or args.all:
        resolve_and_cache_seminal_papers(client, catalog)

    if args.ingest or args.all:
        ingest_missing_seminal_papers(args.papers_file, catalog)

    if args.evaluate or args.all:
        evaluate_multicite_coverage(args.multicite_dir, args.papers_file, args.s2_cache_path)


if __name__ == "__main__":
    main()
