# Prompt inventory

`question_analysis_messages.json` contains the exact compact structured-question-analysis message sequence for a placeholder question. It includes the system instruction, two in-context examples, the fixed JSON schema, and the user request. At runtime, `[QUESTION]` and `[TOPIC_ENTITY]` are replaced with the current sample.

`reader_prompt.json` contains the inherited SubgraphRAG reader instruction, one in-context user/assistant demonstration, and the conditional recovery instruction. Evidence views are compiled programmatically in `code/reason/probe_evidence_views.py`:

- `pathfirst` (path-centric): prioritizes coherent paths and then fills the remaining budget from the same ranked pool.
- `minimal`: keeps a compact union of high-ranked path blocks, candidate-support triples, and constraint-matching evidence.
- `candidate` (candidate-centric): groups retrieved support around graph-derived candidate entities.

All three compilers operate on one retrieved evidence pool and issue no additional retrieval queries.
