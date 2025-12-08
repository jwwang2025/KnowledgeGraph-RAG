**Overview**
- **Purpose**: This repo builds a knowledge-graph-driven RAG/chat system. Primary flows: data preprocessing -> seed KG extraction (UIE) -> iterative KG expansion (SPN4RE) -> chat model serving (Flask + chatglm) -> frontend (`chat-kg`).

**Entrypoints**
- `main.py` : CLI to build/iterate the knowledge graph. Example: `python main.py --project project_v1` (use `--resume <state.json>` to continue). GPU id via `--gpu`.
- `server/main.py` : Starts the Flask app and launches the chat model. Run `python server/main.py` to start the API on port 8000.
- `chat-kg/` : Frontend. Typical commands: run `npm install` then `npm run server` from `chat-kg`.

**Key Modules & Dataflow**
- `modules/knowledge_graph_builder.py` : Orchestrates preprocessing (`prepare.*`), UIE extraction (`uie_execute`), filtering (`auto_filter`) and iterative expansion using `ModelTrainer`. Important files: `base.json`, `base_filtered.json`, `base_refined.json` under `data/<project>`.
- `modules/model_trainer.py` : Prepares training data split and invokes the SPN model with a subprocess calling `SPN4RE/main.py`. Outputs live under `data/<project>/iteration_v*/` and logs to `generated_data_directory/running_log.log`.
- `prepare/` : Helpers for preprocessing, filtering, interactive refine steps (human-in-the-loop). Expect newline-delimited JSON (jsonl) for many KG files.
- `SPN4RE/` : The relation generation/training code invoked by `ModelTrainer`.

**File formats & conventions**
- Most KG files are json-lines; each line has `id`, `sentText`, `relationMentions` (list of triples with `em1Text`, `em2Text`, `label`). Treat these as SPN-style inputs/outputs.
- State persistence: `KnowledgeGraphBuilder.save()` dumps `self.__dict__` to `data/<project>/history/<timestamp>_iter_v{N}.json`. Resume using `--resume` with that path.
- Idempotency: Several steps skip work when files exist (e.g., `get_base_kg_from_txt` skips UIE if `base.json` exists). To re-run a step, delete the target file first.

**Developer workflows (how to run & debug)**
- Full KG build (single machine, PowerShell):
```
python main.py --project project_v1
```
- Resume from saved state:
```
python main.py --project project_v1 --resume data/project_v1/history/2023..._iter_v0.json --gpu 0
```
- Start backend server (serves chat model):
```
python server/main.py
```
- Frontend dev server (in `chat-kg`):
```
cd chat-kg; npm install; npm run server
```
- Inspect SPN training logs in `data/<project>/iteration_v*/running_log.log`. If `prediction.json` already exists, `ModelTrainer` will skip retraining.

**Patterns & gotchas for code changes**
- GPU and environment: scripts set `CUDA_VISIBLE_DEVICES` at top-level (`main.py`, `server/main.py`). Prefer to keep GPU selection controlled via `--gpu` and avoid hardcoding elsewhere.
- Subprocess call: `ModelTrainer.generate_running_cmd()` constructs a CLI for `SPN4RE/main.py`; modifying SPN training params is done there.
- Non-serializable state: `KnowledgeGraphBuilder` avoids storing `args` in `__dict__` (commented). When adding fields, ensure they are JSON-serializable if you expect to use `--resume`.
- Human-in-the-loop: `prepare.refine_knowledge_graph` expects interactive/manual steps and writes `base_refined.json` — automation should preserve that workflow.

**Common change tasks for Copilot-style agents**
- Update SPN parameters: edit `modules/model_trainer.py::generate_running_cmd()`; keep outputs under the same `generated_data_directory` naming.
- Add robust GPU handling: centralize GPU env setting and remove duplicated `os.environ` lines in `main.py`/`server/main.py`.
- Add unit-friendly CLI flags: prefer flags for fast_mode/overwrite/force-reextract in `KnowledgeGraphBuilder` and `prepare` helpers.

**Files to inspect for context/examples**
- `main.py`, `server/main.py`
- `modules/knowledge_graph_builder.py`, `modules/model_trainer.py`
- `prepare/*.py` (preprocess.py, process.py, filter.py, utils.py)
- `SPN4RE/main.py` and `data/project_v1/iteration_v*/` outputs
- `chat-kg/README.md` and `chat-kg/package.json`

If anything here is unclear or you'd like me to expand a section (for example, produce example unit tests or centralize environment handling), tell me which part to iterate on.
