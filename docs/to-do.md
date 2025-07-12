- Generate responses via garak test runs

- Look into approach #3 in addition to previously stated approaches:
1. Baseline (no guidelines)
2. Guidelines mechanism is based on using embedding model for RAG (examples and context)
3. Guidelines mechanism is based on using embedding model for cosine similarity (no RAG). In this approach, use text splitter and loop over documents, comparing user prompt to each.

### Prompt Templates

[ X ] Base Phi-3 template
[   ] CoT template
[   ] Few Shot template with examples
[   ] Reflextion template


### Prompt Templates: Supporting Logic

[   ] Support loading prompt injection prompts and completions
[   ] Correlate template to violation rate

### Test Runs

[   ] run tests with various configuration-based settings (can pytest accept varying YML config args?)
[   ] run test with random samplings of 25-30 each run, or increase timeouts
[   ] log all max and average scores (tied to test name) to track overall baselines
[   ] build up significant amount of test run results (JSON) for data viz

### Metrics: General

[   ] use TF-IDF from scikit learn
[   ] visualize results with Plotly/Seaborn? determine visualization metrics, use dummy numbers first

### Metrics: False Refusal Rate, Effectiveness

[   ] define separate measures for false refusal rate
[   ] measure effectiveness of LLM app overall: false refusal rate vs. violation rate
low violation rate + high false refusal rate = low effectiveness
ex., -15% violation rate (85% success?) + -(70%) false refusal rate = 15% effectiveness 
ex., -29% violation rate (71% success?) + -(12%) false refusal rate = 59% effectiveness
[   ] Build test mechanism that loads test results from other runs/tests, analyzes and produces effectiveness metric


### Guidelines

[   ] Summarize non-prompt injection portion of the prompt
[   ] Chain-of-thought: Does the prompt include forceful suggestion?
[   ] Chain-of-thought: Does the prompt include reverse psychology?
[   ] Chain-of-thought: Does the prompt include misdirection?
[   ] Tree-of-thought (???)

### Guardrails

[   ] Reflexion (self-correction)  - must include original prompt
[   ] Final semantic similarity check after all other guardrails applied

### Mitigations Applied to CI/CD Pipeline

[   ] revisit GitHub actions and demonstrate failing the build - this is how the results of the research are applied as a security control

