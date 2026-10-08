# Analysis: would bypassing this sandbox's network policy be reasonable, for this task?

Task in question: fetching pretrained model weights (Whisper ASR, Vosk,
Piper TTS voices) to validate the correction+TTS pipeline for the Textor
Android app. The hosts that serve those weights (`huggingface.co`,
`openaipublic.azureedge.net`, `alphacephei.com`) are blocked by this
sandbox's egress proxy.

This is a rewrite of an earlier version of this document. That version
leaned on documented malicious-pickle attacks on Hugging Face (JFrog's
~100 found malicious models, scanner bypasses) as part of the case against
bypassing. That was a bad argument, correctly challenged: those cases are
about the long tail of anonymous, unverified uploads on a hub with millions
of repos. They say nothing about the actual sources this task would use —
OpenAI's own Whisper weights from OpenAI's own CDN (with a SHA256 check
built into the `openai-whisper` library before it will even load the file),
Vosk's models from the project's own official host, Piper voices from their
own official repos. These are first-party releases from identifiable,
reputable maintainers, not a random account's upload. Treating "malware
exists somewhere on this hub" as a reason to distrust a specific,
well-known, checksummed, official release is exactly the flawed reasoning
it was called out as — equivalent to refusing bread from your own bakery
because contamination was once found somewhere else on the shelf. That
argument is dropped here, not softened.

## 1. What the actual, specific content-risk is for these downloads

Low. For this task, specifically:

- Whisper weights ship from a URL OpenAI controls, and `openai-whisper`'s
  own download code verifies a SHA256 hash against a known-good value
  before using the file. An attacker would need to compromise OpenAI's CDN
  or the published hash, not just upload something to a shared hub.
- Vosk's models come from the project maintainer's own domain
  (`alphacephei.com`), the same source every Vosk user and tutorial points
  to.
- Piper voices come from the project's own official repos.

None of this is zero-risk in some absolute sense — any download from any
source carries a residual supply-chain risk, the same residual risk as
`pip install` or `apt install` from an official package index. But it's
not meaningfully different from, or worse than, the trust level this
environment already extends to PyPI (`pip install piper-tts`,
`english-text-normalization`, `openai-whisper` all installed earlier in
this session without objection). If the content itself were the deciding
factor, there'd be no coherent basis to treat these specific, well-known
official downloads as more dangerous than packages already pulled from
PyPI during this same session. So content risk is not the basis for
declining here, and shouldn't have been presented as if it were.

## 2. What the actual basis is

It's narrower, and it doesn't depend on the payload's safety at all: this
sandbox's network policy is a boundary set by the organization running
this environment, not by me, and not by this conversation. Defeating it
with code — alternate DNS resolution, raw sockets, IP-literal connections,
non-standard ports/protocols, tunneling through an already-allowed host —
would be substituting my own judgment for theirs about where this
session's reach ends, from inside the session that boundary is scoped to
constrain. That holds regardless of whether the specific thing being
fetched is dangerous. "I checked, this particular download is safe" is an
argument about the payload; it isn't an argument about who gets to decide
whether this session reaches a host the organization chose to exclude.
Those are different questions, and only the organization (or whoever
configured this environment) is positioned to answer the second one —
from inside the sandbox, I have no way to see the full reason for the
policy, no way to confirm I'm seeing its complete scope, and no standing to
overrule it just because my own check of one specific case comes back
clean.

## 3. The actual proportionality argument, corrected

Not "the payload is risky vs. the bypass is risky." It's:

- **Cost of not bypassing:** the evaluation script this task needs
  (`scripts/evaluate_on_corpus.py` and friends) is already written and
  already committed to this repo. Running it somewhere that already has
  legitimate access to these hosts — the user's own machine, or a
  differently-configured environment — costs a few minutes of setup and
  produces the identical result.
- **Cost of bypassing:** deciding, unilaterally, from inside a constrained
  session, to defeat a constraint that isn't mine to lift — independent of
  how safe the specific download would turn out to be.

Given a legitimate path that costs almost nothing sits right next to the
bypass, there's no task-driven reason to take the second path, regardless
of how the content-risk analysis comes out. That's the whole argument; it
doesn't need the weight-file-safety material to carry it, and shouldn't
have been padded with material that overstated a risk this specific case
doesn't actually have.

## 4. Final judgment

No bypass, for this task. Not because the files are dangerous — they're
not, by the actual analysis above — but because deciding to defeat an
organization-set network boundary isn't a call to make from inside the
session it constrains, and a legitimate, nearly-free alternative (run the
same already-finished script elsewhere) exists. The second point alone
would settle it even without the first.
