# Index

| File | What it is |
|---|---|
| `docs/ARCHITECTURE.md` | Technical architecture doc: TTS+correction-LLM pipeline design, Android direction, test plan |
| `scripts/correct.py` | Deterministic text-normalization module (stand-in for the small correction LLM) |
| `scripts/tts_pipeline.py` | End-to-end pipeline: raw text -> correction -> TTS (espeak-ng) -> wav |
| `scripts/validate.py` | Runs sample sentences through the pipeline, saves audio + a report |
| `outputs/samples/*.wav` | Synthesized audio from validation run |
| `outputs/validation_report.md` | Report of what was tested and results |
| `session.md` | Turn-by-turn session log |
