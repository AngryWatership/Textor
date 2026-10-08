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
| `session.md` | Turn-by-turn session log |
