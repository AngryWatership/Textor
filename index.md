# Index

| File | What it is |
|---|---|
| `docs/ARCHITECTURE.md` | Technical architecture doc: TTS+correction-LLM pipeline design, Android direction, test plan |
| `scripts/correct.py` | Deterministic text-normalization module (stand-in for the small correction LLM) |
| `scripts/tts_pipeline.py` | End-to-end pipeline: raw text -> correction -> TTS (espeak-ng) -> wav |
| `scripts/validate.py` | Runs sample sentences through the pipeline, saves audio + a report |
| `scripts/benchmark_correct.py` | Accuracy benchmark for the correction stage (23 hand-built cases) |
| `outputs/samples/*.wav` | Synthesized audio from validation run |
| `outputs/validation_report.md` | Report of what was tested and results |
| `outputs/correction_benchmark.txt` | Accuracy results: regex stand-in 6/23 (26%) vs. real `english-text-normalization` lib 12/23 (52%) |
| `data/ljspeech_sample_metadata.csv` | 8 real utterances: raw text + ground-truth normalized transcript (from coqui-ai/TTS test fixtures, public-domain LJSpeech slice) |
| `data/ljspeech_sample_wavs/` | Matching real recorded audio for those 8 utterances |
| `data/README.md` | Provenance of the corpus data |
| `scripts/evaluate_on_corpus.py` | Runs both normalizers against the real corpus; checks synth-vs-real audio duration |
| `outputs/corpus_evaluation.txt` | Results: 7/8 (87.5%) vs 8/8 (100%) exact match on real-recording ground truth |
| `docs/NETWORK_POLICY_BYPASS_ANALYSIS.md` | Researched analysis of why defeating the sandbox network policy isn't justified for this task; final judgment |
| `session.md` | Turn-by-turn session log |
