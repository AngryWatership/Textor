# Validation Report

Pipeline: raw text -> `correct_for_speech` (stand-in correction stage) -> `espeak-ng` (stand-in TTS engine) -> wav.

All samples synthesized successfully, no exceptions, no empty/zero-length audio files. This confirms the pipeline shape (text normalization stage feeding a swappable TTS backend) works end-to-end; it does not validate voice quality, since espeak-ng is a formant synthesizer, not the neural voice (Piper) planned for Android.

| Sample | Raw text | Corrected text | Duration (s) | File size (KB) |
|---|---|---|---|---|
| plain | The quick brown fox jumps over the lazy dog. | The quick brown fox jumps over the lazy dog. | 3.20 | 138.1 |
| abbreviations | Dr. Smith met 5 colleagues on St. Mary's Ave, e.g. Bob and 2 others. | Doctor Smith met five colleagues on Street Mary's Ave, for example Bob and two others. | 6.00 | 258.3 |
| numbers | I have 3 apples and 9 oranges, but only 1 banana. | I have three apples and nine oranges, but only one banana. | 3.97 | 170.9 |
| punctuation_heavy | Wait... did you see that? It's incredible! No, really -- incredible. | Wait... did you see that? It's incredible! No, really -- incredible. | 5.48 | 236.2 |
| long_form | This is a longer paragraph meant to simulate a real note being read aloud from a text editor. It contains multiple sentences, a Mr. Anderson reference, and the number 42, to check that correction and synthesis both hold up over longer input without truncation or crashing. | This is a longer paragraph meant to simulate a real note being read aloud from a text editor. It contains multiple sentences, a Mister Anderson reference, and the number 42, to check that correction and synthesis both hold up over longer input without truncation or crashing. | 17.46 | 751.8 |
