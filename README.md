# LLMaAA
> The official repository for paper "LLMaAA: Making Large Language Models as Active Annotators".

## Key Dependencies

- openai                   **0.27.4**

- numpy                    1.24.2

- torch                    1.10.0+cu111

- tokenizers               0.10.3

- transformers             4.12.4

- sentence-transformers    2.2.2

- kmeans-pytorch           0.3

- func_timeout

- ujson

- tqdm

## Brief Introduction to usage

*Disclaimer: Since I (the first author) didn't have access to Azure OpenAI service after internship at Langboat, I haven't test the code recently. So unfortunately I cannot guarantee that the code can be ran bug-free without any modification and please take this repository as a reference implementation.* 

1. Setup openai config @ `~/openai_config.json`. We use the Azure GPT API in our experiments, so in default you need to provide the key and base for OpenAI service.

2. Download data to ``~/data/``. See ``~/data/README.md`` in the directory for details.

3. For active annotation (LLMaAA),

    - First retrieve demonstations (random/knn) for train/test data with `~/src/demo_retrieval.py`.
    - Run active annotation with ``~/src/active_annotate.py``. Since the demo indices are static, so the previous annotation results will be stored in an auto-generated cache file.

4. For data generation (ZeroGen/FewGen),
    - Run with ``~/src/data_gen.py``.

5. For testing prompting (Prompt) performance directly,
    - Run inference and evaluation with ``~/src/llm_test.py``. May experience timeout/ratelimit/etc.

## Adapt to new dataset

1. Prepare the data in NER/RE format.
2. Setup ``meta.json`` in data directory and ``configs/{dataset}.json`` for annotator/generator. The ``configs`` folder can be found in ``~/src/data_synth/`` (for generator) and ``~/src/llm_annotator`` (for annotator).
3. If use demonstration in prompt engineering, you need to build a map from train/test to demo data, with ``~/src/demo_retrieval.py``.

## Citation
```bibtex
@inproceedings{zhang-etal-2023-llmaaa,
    title = "{LLM}a{AA}: Making Large Language Models as Active Annotators",
    author = "Zhang, Ruoyu  and Li, Yanzeng  and Ma, Yongliang  and Zhou, Ming  and Zou, Lei",
    editor = "Bouamor, Houda  and Pino, Juan  and Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.872",
    doi = "10.18653/v1/2023.findings-emnlp.872",
    pages = "13088--13103",
}
```
