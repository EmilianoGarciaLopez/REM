# REM: Reasoning over Embodied Multi‑Frame Trajectories

REM is a framework and benchmark for evaluating multimodal LLMs on long‑horizon embodied spatial reasoning. It uses controllable 3D scenes rendered in Blender to produce egocentric trajectories and automatically generates question–answer (QA) pairs probing:

- Counting and numerical comparison (object permanence and individuation across frames)
- Left/right relational reasoning (consistent spatial relations across views)
- Temporal ordering (first appearance across a trajectory)

Paper: “REM: Evaluating LLM Embodied Spatial Reasoning through Multi‑Frame Trajectories”  
Code: https://github.com/EmilianoGarciaLopez/REM

This README explains how to install, generate trajectories, create questions, run model inference, aggregate results, and reproduce plots. It also includes a script‑by‑script index of this repo.


## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Environment variables](#environment-variables)
- [Quickstart (plots only)](#quickstart-plots-only)
- [Full pipeline](#full-pipeline)
  - [1) Generate scenes](#1-generate-scenes)
  - [2) Render trajectories in Blender](#2-render-trajectories-in-blender)
  - [3) Generate questions](#3-generate-questions)
  - [4) Run model inference](#4-run-model-inference)
  - [5) Aggregate results](#5-aggregate-results)
  - [6) Analysis and figures](#6-analysis-and-figures)
- [Data formats](#data-formats)
- [Column naming and consistency](#column-naming-and-consistency)
- [Troubleshooting](#troubleshooting)
- [Script-by-script index](#script-by-script-index)
- [Citing](#citing)


## Requirements
- Blender 4.0+ (EEVEE Next is used; Cycles GPU acceleration supported; see notes below)
- Python 3.10+ (system Python for CLI tools; Blender uses its own bundled Python at runtime)
- GPU recommended for rendering (NVIDIA/OPTIX or AMD/Metal on macOS)
- API keys for any models you want to evaluate:
  - OpenAI (OPENAI_API_KEY)
  - Anthropic (ANTHROPIC_API_KEY)
  - Google Gemini (GOOGLE_API_KEY)
  - OpenRouter (OPENROUTER_API_KEY) for Llama‑3.2 vision via OpenRouter


## Installation
1) Clone the repo
```
git clone https://github.com/EmilianoGarciaLopez/REM
cd REM
```

2) Install Python packages (system environment)
```
pip install -r requirements.txt
```

Notes:
- The code uses two Gemini SDKs:
  - google.generativeai (classic Gemini 1.5 API)
  - google.genai (new “genai” client, used in text‑only inference)
- For rendering, Blender is called in background mode (bpy comes from Blender’s Python, no need to install it in your system Python).


## Environment variables
Create a .env at repo root (or export env vars) with any keys you will use:
```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...
```

Many scripts also automatically load .env when present.


## Quickstart (plots only)
If you just want to reproduce plots/analysis using the bundled aggregate CSV (no rendering/inference):

- Counting analysis:
```
python src/counting_vs_gt_mf.py
```

- Overall success tables:
```
python src/overall_success.py --csv src/all_questions_aggregated_with_text_new_prompt.csv
```

- Duplicate/objects scaling plots:
```
python src/epic_graph_objects.py
python src/epic_graph_trajectories.py
```

These default to using the included aggregate CSVs under src/.


## Full pipeline

### 1) Generate scenes
This step creates run folders under src/trajectories/ with a scene_config.json and a trajectory path plan.

- Batch generator with controlled settings:
```
python src/generate_dataset.py
```
This script:
- Creates many run folders: src/trajectories/0001-run, 0002-run, …
- Writes scene_config.json and trajectory_moves.txt in each.

You can also create a single run interactively:
```
python src/generate_scene.py --num-shapes 24 --trajectory-size 8 --same-size --homogenous-texture
```
This produces a new run folder (e.g., src/trajectories/0001-run/) with scene_config.json and trajectory_moves.txt.

Tips:
- num-shapes controls scene congestion.
- trajectory-size ∈ {2, 4, 8, 16, 32, 64} controls length (egocentric frames).
- times (duplicates) controls how many duplicated objects to inject (each t adds t+1 instances of a chosen base object).


### 2) Render trajectories in Blender
Render each run’s images and generate annotations.csv (per-frame visible objects in left‑to‑right order).

- Render all runs:
```
python src/run_all_trajectories.py
```
This calls Blender like:
```
blender --background --python src/run_simulation.py -- --config-file src/trajectories/0001-run
```

- Or render a single run manually:
```
blender --background --python src/run_simulation.py -- --config-file src/trajectories/0001-run
```

Outputs per run (in src/trajectories/####-run/images/):
- image_0000_*.png, image_0001_*.png, …
- annotations.csv (with image_filename, annotation, visible_objects)
- scene_plot.png (top‑down visualization)


### 3) Generate questions
These scripts parse annotations and write per‑run question CSVs inside each images/ folder:
- comparison_questions_in_frame.csv
- comparison_questions_out_of_frame.csv
- left_right_questions.csv
- number_questions.csv
- order_preserving_questions.csv

By default, the generators use dataset‑specific glob paths. If you render into src/trajectories/, update the glob to point there.

Edit the following lines in each script:

- src/question_answering/generate_comparison_in_frame_questions.py  
  Replace:
```
annotation_paths = glob.glob("src/duplicates_1_0/*-run/images/annotations.csv")
```
with:
```
annotation_paths = glob.glob("src/trajectories/*-run/images/annotations.csv")
```

- src/question_answering/generate_comparison_out_of_frame_questions.py  
  Replace:
```
annotation_paths = glob.glob("src/duplicates_1_0/*-run/images/annotations.csv")
```
with:
```
annotation_paths = glob.glob("src/trajectories/*-run/images/annotations.csv")
```

- src/question_answering/generate_directional_questions.py  
  Replace:
```
all_annotations = glob.glob("src/unique/*-run/images/annotations.csv")
```
with:
```
all_annotations = glob.glob("src/trajectories/*-run/images/annotations.csv")
```

- src/question_answering/generate_number_questions.py  
  Replace:
```
annotation_paths = glob.glob("src/duplicates_1_0/*-run/images/annotations.csv")
```
with:
```
annotation_paths = glob.glob("src/trajectories/*-run/images/annotations.csv")
```

- src/question_answering/generate_order_preserving_questions.py  
  Replace:
```
annotation_paths = glob.glob("src/duplicates_1_0/*-run/images/annotations.csv")
```
with:
```
annotation_paths = glob.glob("src/trajectories/*-run/images/annotations.csv")
```

Then run them:
```
python src/question_answering/generate_number_questions.py
python src/question_answering/generate_comparison_in_frame_questions.py
python src/question_answering/generate_comparison_out_of_frame_questions.py
python src/question_answering/generate_directional_questions.py
python src/question_answering/generate_order_preserving_questions.py
```

Note: These scripts call read_annotations(), which also writes images/data_dict.csv (counts and frame indices) used later by aggregation and analysis.


### 4) Run model inference
Fill model answer columns in each questions CSV. Choose the APIs you want.

- OpenAI GPT‑4o (batch API):
```
python src/batch_inference_openai.py --base_dir src/trajectories --logging true
```
Writes/updates a “gpt-4o” column per CSV, using OpenAI’s batch jobs.

- Anthropic Claude 3.5 Sonnet (batch API):
```
python src/batch_inference_anthropic.py --base_dir src/trajectories
```
Writes/updates a “claude-3.5-sonnet-latest” column per CSV (custom_id based matching).

- Gemini 1.5 Flash (images):
```
python src/inference_gemini.py --base_dir src/trajectories --logging true
```
Writes/updates a “gemini-1.5-flash-latest” column.

- Gemini 1.5 Flash (text‑only from annotations):
```
python src/inference_gemini_text.py --base_dir src/trajectories --logging true
```
Writes/updates a “gemini-1.5-flash-latest-TEXT-NEW-PROMPT” column. Uses only the “visible_objects” text (no images).

- Llama‑3.2‑90B Vision (via OpenRouter):
```
python src/inference_llama.py --base_dir src/trajectories --logging true
```
Writes/updates a “llama-3.2-90b-latest” column.

Interactive alternative (sends each question + all images to multiple APIs and writes answers back into CSVs):
```
python src/process_images.py src/trajectories
```

Cost notice:
- These scripts upload all trajectory images at 960×640 to the APIs. Keep an eye on usage and set smaller test sets first.


### 5) Aggregate results
Merge all question CSVs across all runs into a single analysis file, compute correctness (rule‑based), and append scene/trajectory metadata.

```
python src/aggregate_questions.py --base_dir src/trajectories
```

This writes:
```
src/trajectories/all_questions_aggregated_final.csv
```

It also computes:
- answered_correctly_<model> columns via semantic matching (yes/no/before/after/same time/numeric)
- average_occlusion, number_of_objects_it_sees (unique types seen), duplicates stats
- comp_1/comp_2 (counts used in comparison questions)
- For counting questions, max_number_in_single_frame (per question target).

If you used different model column names than the defaults, edit MODEL_COLUMNS in src/aggregate_questions.py (see “Column naming” below).


### 6) Analysis and figures
- Counting vs ground truth & max per frame:
```
python src/counting_vs_gt_mf.py
```

- Overall success by question type (tables + normalized to chance):
```
python src/overall_success.py --csv src/trajectories/all_questions_aggregated_final.csv
```

- Duplicate scaling by object bins:
```
python src/epic_graph_objects.py
```

- Trajectory/duplicates/object scaling:
```
python src/epic_graph_trajectories.py
```

- Additional quick views:
```
python src/csv_viewing.py --csv src/trajectories/all_questions_aggregated_final.csv
python src/num_objects.py --csv src/trajectories/all_questions_aggregated_final.csv
```


## Data formats

Per trajectory folder:
```
src/trajectories/0001-run/
  scene_config.json            # object list and attributes; duplication info
  trajectory_moves.txt         # discrete actions between frames
  images/
    image_0000_*.png
    image_0001_*.png
    ...
    annotations.csv            # per frame: image_filename, annotation, visible_objects
    data_dict.csv              # derived counts/frames from read_annotations()
    comparison_questions_in_frame.csv
    comparison_questions_out_of_frame.csv
    left_right_questions.csv
    number_questions.csv
    order_preserving_questions.csv
```

annotations.csv:
- image_filename: e.g., image_0001_move_forward_1m.png
- annotation: optional textual caption (includes camera x,y)
- visible_objects: CSV string “object_name (pct%), …” in left‑to‑right order; GroundPlane is ignored downstream.

Question CSVs:
- question, answer (ground truth label)
- One column per model (string answer)
- Aggregation appends answered_correctly_<model> columns to the aggregate CSV.


## Column naming and consistency
The aggregator looks for a fixed set of model columns:
```
MODEL_COLUMNS = [
  "gpt-4o",
  "gemini-1.5-pro-latest",
  "gemini-1.5-flash-latest",
  "nova-lite-v1",
  "llama-3.2-11b-vision-instruct",
  "gemini-1.5-flash-latest-TEXT-NEW-PROMPT",
]
```
If your inference scripts wrote different names (e.g., “llama-3.2-90b-latest” or “claude-3.5-sonnet-latest”), either:
- Change the inference script constant(s) to match the aggregator, or
- Edit MODEL_COLUMNS in src/aggregate_questions.py to include the exact column headers your CSVs contain.

Similarly, batch scripts define their own MODEL_COLUMN names. Keep them aligned to avoid missing answers in the aggregate output.


## Troubleshooting

- Blender not using GPU:
  - On macOS: METAL devices are enabled automatically in run_simulation.py.
  - On Windows/Linux: the script tries OPTIX first, then CUDA. If nothing is found, Blender exits. Ensure your Blender Preferences (Edit → Preferences → System) see your GPU; invoke once in UI if needed.

- Images missing or corrupted:
  - Inference code resizes images to 960×640; it skips truncated images and logs warnings.
  - Terminal display (process_images.py) uses term-image; it will skip truncated images rather than fail.

- Batch API limits:
  - OpenAI batches are chunked to ~180 MB JSONL by default in batch_inference_openai.py.
  - For Anthropic batch, the script polls until status is “completed.” Errors log per‑request.

- Question generators not finding your runs:
  - Update the glob in each generator script to src/trajectories/*-run/images/annotations.csv (see instructions above), or create symlinks to match the current globs.

- Overseer script:
  - src/overseer.py references automate_question_answering.py which is not included; treat it as deprecated unless you add your own runner.

- Column mismatches:
  - If aggregate_questions.py reports missing columns, edit MODEL_COLUMNS to match your outputs, or re‑run inference with matching column names.


## Script-by-script index

Top level:
- cancel.py — Cancels all active (non‑final) OpenAI batch jobs.
- monitor.py — Prints counts/status of active OpenAI batches.
- requirements.txt — Python package pins.
- dataset_generation.log — Log file written by generate_dataset.py.
- prediction_accuracy.png — Example figure (if present).
- README.md — This file.

Core scene generation and rendering (src/):
- generate_scene.py — Build a scene_config.json with shapes/colors/duplicates, plus a discrete trajectory; also draws a top‑down plot.
- generate_dataset.py — Batch generator wrapping generate_scene.py; creates many run folders.
- run_simulation.py — Blender script: loads scene_config.json, renders trajectory frames, writes images/annotations.csv (and optional scene plot). Uses EEVEE Next for utilities and Cycles (GPU) for main renders.
- run_all_trajectories.py — Calls Blender in background to render every run under src/trajectories/.
- reading.py — Utilities for parsing annotations; prototype for earliest appearance grouping.

Question generation (src/question_answering/):
- common_utils.py — Parses “visible_objects”, builds counts and frame sets; writes images/data_dict.csv.
- generate_number_questions.py — Creates “How many …?” questions (colors/shapes/color‑shape).
- generate_comparison_in_frame_questions.py — “Are there more X or Y?” (requires X,Y co‑occur in at least one frame).
- generate_comparison_out_of_frame_questions.py — “Are there more X or Y?” (requires X,Y never co‑occur in the same frame).
- generate_directional_questions.py — Left/right questions; ensures consistent left/right across all frames where both appear.
- generate_order_preserving_questions.py — “Did we see X before, after, or same time as Y?” using earliest appearance frames.

Model APIs and inference (src/):
- api_models.py — Thin wrappers for GPT‑4o (OpenAI), Gemini (1.5/2.x), Claude 3.5 (Anthropic), and OLlama (local); includes RobotVision class for interactive runs.
- process_images.py — CLI that loads all images and appends model answers (all known APIs) into the question CSVs.
- batch_inference_openai.py — OpenAI GPT‑4o batch pipeline across all runs and question CSVs.
- batch_inference_anthropic.py — Anthropic Claude 3.5 Sonnet batch pipeline.
- inference_gemini.py — Parallel Gemini 1.5 Flash inference using images.
- inference_gemini_text.py — Gemini 1.5 Flash (new “genai” client) using text‑only visible_objects (no images).
- inference_llama.py — Llama‑3.2‑90B Vision via OpenRouter.

Aggregation and analysis (src/):
- aggregate_questions.py — Walks runs, merges all question CSVs into one file, computes correctness, comp_1/comp_2, occlusion and duplicate stats; writes all_questions_aggregated_final.csv under base_dir.
- counting_vs_gt_mf.py — Counting analysis (model vs ground truth, ratio to max per frame) + plots.
- overall_success.py — Success tables per question type for each model (and normalized to chance).
- epic_graph_objects.py — Non‑number questions: fraction correct vs duplicate bins, stratified by objects seen.
- epic_graph_trajectories.py — Non‑number questions: fraction correct vs duplicates or objects, stratified by trajectory length.
- csv_viewing.py — Example plot (success vs #objects).
- num_objects.py — Histogram over duplicates column (quick exploratory chart).

Data artifacts (src/):
- all_questions_aggregated.csv / ..._with_text*.csv — Large precomputed aggregates used by analysis scripts.