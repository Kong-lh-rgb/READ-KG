# READ-KG


## Repository structure

- `code/retrieve/`: structured question analysis, retriever training, inference, and evaluation.
- `code/reason/`: evidence-view construction, reader inference, answer scoring, answer-set selection, and evaluation.
- `configs/`: retriever configurations for ComplexWebQuestions and WebQSP.
- `scripts/`: staged experiment and ablation runners.
- `prompts/`: prompt templates used by the pipeline.
- `environment/`: recorded Python package versions for the reader environments.

## External resources

Benchmark data, Freebase resources, model weights, embeddings, retrieval tensors, checkpoints, generated predictions, and experiment results are not included. Set local model and data paths through the environment variables documented in the runner scripts.

Typical entry points are:

```bash
bash scripts/run_readkg_stage_b.sh preflight
bash scripts/run_readkg_webqsp_stage_b.sh preflight
```

See the comments and environment-variable defaults in each runner for the corresponding dataset and model paths.
