# Restaurant AI ordering MVP

Python 3.11+. One application entry point, with separate Conversation, Orchestration,
Database and Voice responsibilities.

```
Microphone -> voice.listen() -> ConversationService.handle(session_id, text)
 -> structured Action -> RestaurantOrderApplication.submit()
 -> OrderOrchestrator -> order_db.tools -> SQLite
 -> ConversationResult -> grounded Arabic response -> voice.speak()
```

## Run

Install Python dependencies in the application's Python 3.11 environment:

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure the existing asset paths in `.env`; models are never downloaded or copied.
Start the existing llama.cpp server in a separate PowerShell terminal:

```powershell
& "D:\Ollama Models\runtime\llama.cpp\llama-server.exe" `
  -m "D:\Ollama Models\gguf\qwen3.5-4b\Qwen3.5-4B-Q4_K_M.gguf" `
  --host 127.0.0.1 --port 8081 --ctx-size 4096 --gpu-layers 99 `
  --parallel 1 --alias qwen35-4b-local --reasoning off --offline --no-webui
```

These flags were checked against the installed runtime. The server is not managed
or started by Python. `LLAMA_CPP_SERVER_PATH` and `LOCAL_LLM_MODEL_PATH` document
launcher paths only. Keep the unauthenticated server loopback-only and stop it when
unused. Python rejects non-loopback local endpoints and bypasses HTTP proxies.

Normal operation needs no cloud credentials: llama.cpp Qwen, local faster-whisper
Arabic transcription (not translation), and Hag Shalaby Egyptian Arabic TTS.
The shipped `.env.example` selects Hag. Hag uses a separate Python environment:
set `HAG_PYTHON` to its Python executable, `HAG_MODEL_PATH` to the directory
containing `model.pt`, `config.yaml`, `vocab.txt`, `text_v3.py` and `ref.wav`,
and `HAG_HF_CACHE` to a Hugging Face cache containing Vocos. The model worker
is kept warm for the session and runs offline. No model files or generated audio
are included in this source package. The benchmark environment used Python 3.11,
F5-TTS 1.1.22, PyTorch/torchaudio 2.8.0 CUDA 12.6, SoundFile and PyYAML;
install compatible dependencies and obtain the publisher's model assets separately.
See the [official model card](https://huggingface.co/nomeda-lab/hag-shalaby-tts-0.2B-v0.1)
for model requirements, assets and license. `HAG_DEVICE=cuda` uses the GPU;
`cpu` is configurable but has not been performance-validated for this demo.
Voice models load on first use, never on import or in text mode.

For a new machine, create a separate environment for Hag and install a compatible
CUDA build of PyTorch/torchaudio for that machine, then install `f5-tts==1.1.22`,
`soundfile` and `PyYAML` in that environment. Install the main requirements in
the application's environment, not into the Hag environment. Follow the model
card to obtain the five model assets and Vocos cache separately; this application
will not download them. Update all paths in `.env`. Model files and drivers are
not included in the ZIP, so copying the project alone is insufficient for voice.

CUDA Whisper defaults to `int8_float16`; the installed CT2 runtime supports it.
If CUDA libraries or VRAM are insufficient, set `STT_DEVICE=cpu` and
`STT_COMPUTE_TYPE=int8`. This is an explicit configuration choice, not an automatic
model downgrade. CUDA STT needs compatible CUDA/cuDNN runtime libraries accessible
to CTranslate2. If VRAM is tight, prefer CPU INT8 Whisper while retaining GPU TTS/Qwen.
The previous local TTS remains available using `TTS_PROVIDER=kemetone` and
`LOCAL_TTS_MODEL_PATH`; it is not the preferred demo voice.

Optional Ollama: keep its already imported model; do not pull another copy.
With Ollama running, set `$env:LLM_PROVIDER='ollama'` then run the same app command.
`OLLAMA_MODEL=qwen35-4b-local:latest` and the loopback endpoint are configurable.
Both local providers use exactly the same validated Conversation pipeline.
Cloud comparison remains optional: `LLM_PROVIDER=groq` (set `GROQ_API_KEY` and
`LLM_MODEL=openai/gpt-oss-20b`), `STT_PROVIDER=google`, `TTS_PROVIDER=edge`.

Text debug path: `python app.py --mode text`

Voice MVP: `python app.py --mode voice`

Use `--db-path demo.sqlite3` for a separate database and `--session-id demo-1`
to select a session. `DB_PATH` is process-wide as in the teammate database code:
run one configured database per process. SQLite initializes and seeds only when needed.
Chat/confirmation state is in memory; keep the application and conversation service
alive throughout a call. Session IDs isolate drafts and chat histories.

Example: ask for two Chicken Ranch medium, change the quantity or modifiers,
say `خلاص` for the summary, then `أيوه أكد` to explicitly save. Say `خروج` or
`exit` to leave; three empty listens end the voice session.

Egyptian Arabic recognition defaults to `STT_LANGUAGE=ar-EG` (Whisper receives `ar`).
English STT remains available through `STT_LANGUAGE=en-US`. `TTS_SPEED` controls
KemeTone; `TTS_VOICE` and `TTS_RATE` apply only to optional Edge TTS.
Timeouts, ambient calibration and playback volume apply to both voice paths.
Explicit legacy `VoiceConfig()` objects retain Google/Edge defaults for compatibility;
the application uses `VoiceConfig.from_env()` with local-first defaults.

## Conversation handoff

```
from app import RestaurantOrderApplication
from conversation import ConversationService, configured_provider

application = RestaurantOrderApplication()
session_id = application.start_session()
conversation = ConversationService(application, configured_provider())
reply = conversation.handle(session_id, "عايز اتنين Chicken Ranch وسط")
print(reply.text)
```

Tests supply a provider implementing `complete(messages) -> JSON string`.
The exact extraction schema is in `conversation/prompts.py`; the bridge rejects
malformed JSON, unknown fields/intents, invalid quantities and invented IDs.
Customer item wording is searched through Database tools, which normalize common
Arabic/English loanwords without substituting products. The model cannot validate
menu facts. Database tools resolve identity, availability and permitted modifiers.
Menu questions and summary names/prices use real tool data; totals are never
calculated by Conversation. Replies are rendered deterministically, not invented
by another model pass.

References use the current draft's real lines. Ambiguous pronouns and impossible
ordinals ask for clarification. Removing part of a line updates its quantity.
The canonical `update_item` cannot change an item's size/identity; these requests
still need a clear remove/add instruction. A partial modifier request uses the
optional Conversation-only `affected_quantity` field, validated against the current
draft and item details, then decomposes into update + add Actions. It never modifies
all units implicitly. Explicit short phrases such as `واحدة من غير بصل`, `شيل واحدة`,
`زود واحدة` and `نقص واحدة` have narrow language guards against model scope omissions.
Two variants remain ambiguous and receive a question naming actual draft variants.
Split operations are not atomic: on add failure, the original quantity is restored
only if a fresh draft proves the reduced draft is unchanged. Uncertain writes are
not blindly retried; the customer is asked to review the current summary.
Equivalent lines may merge on update; each next turn reads the database again.

Only explicit affirmative wording maps to `confirm_order`; `خلاص` maps to
`show_summary`. The orchestrator enforces current-summary confirmation, draft
invalidation, price-change reconfirmation, request IDs and save retries.

## Verify

`python -B -m pytest -q -p no:cacheprovider` runs all deterministic tests using
temporary SQLite databases, scripted LLM output and mocked external audio SDKs.
No automated test needs credentials, audio devices or network services.

`python -m ruff check --no-cache app.py conversation voice tests/test_conversation.py tests/test_voice_integration.py tests/test_local_providers.py`
checks new code. The unchanged Database package and its original tests have
pre-existing Ruff style findings; they are not rewritten to silence them.

Optional live checks (not part of automated verification):

- `python app.py --mode text --db-path smoke.sqlite3`: local Qwen understanding.
- `python app.py --voice-smoke stt`: real microphone and configured STT.
- `python app.py --voice-smoke tts`: configured TTS and speaker playback.
- `python app.py --mode voice --db-path smoke.sqlite3`: full hardware flow.

Listen/speak wiring tests verify transport only, not speech accuracy. Live checks
require the installed dependencies and hardware (cloud providers also need network).
Model-based file checks verify loading/inference, not microphone accuracy or TTS
intelligibility. Review Egyptian Arabic and mixed names on real hardware before use.

Prior local benchmarks used Qwen and Hag concurrently on a 6 GB RTX 4050; combined
GPU use after synthesis was about 3.7 GB. Hag's corrected reference transcript was
preferred by the user over KemeTone. This is not a full microphone-to-speaker
acceptance test: verify Whisper accuracy and the complete flow on the presentation
hardware. Three simultaneous GPU models may need more memory than that benchmark.

For a team-facing explanation, demo checklist and discussion questions, see
[`STUDY_GUIDE.md`](STUDY_GUIDE.md).

ZIP submissions are input archives, not runtime code. Their Agent/Conversation,
demo and cache artifacts are not imported by this application.
