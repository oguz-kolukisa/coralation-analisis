# Claude Code Instructions for Coralation

## Git Commits
- Do NOT add `Co-Authored-By: Claude` to commit messages.

## MANDATORY: Clean Code Enforcement (Every Code Change)

**After EVERY function you write or modify, you MUST verify it against ALL of these rules BEFORE moving on. This is NOT optional. If ANY rule is violated, fix it immediately — do NOT proceed to the next function.**

### Per-Function Gate (check each function individually):
1. **Single Responsibility** — Does this function do exactly ONE thing? If you can describe it with "and", split it.
2. **Line Count** — Is it <= 20 lines (excluding docstring)? If not, extract sub-functions.
3. **Arguments** — Does it have <= 3 parameters (excluding `self`)? If not, group into a dataclass.
4. **Nesting** — Is max nesting <= 2 levels? If not, extract the inner block into a named function.
5. **Abstraction Level** — Does it mix orchestration with detail? Orchestrators call phases; phases call helpers. Never mix.
6. **Name** — Does the name describe exactly what it does? No "and" names.
7. **Side Effects** — Does it either compute OR act, not both?

### Per-Task Gate (check after ALL code changes for a task are done):
1. **Run tests**: `uv run python -m pytest tests/ -v` — ALL must pass.
2. **New methods need tests**: Every new public/internal method MUST have unit tests in `tests/unit/`.
3. **Modified methods need test verification**: Run tests IMMEDIATELY after changing any method. Fix broken tests by updating assertions to match new behavior. NEVER delete tests.
4. **DRY check**: Scan for duplicated logic across the changed functions. Extract shared helpers.

**If you skip any of these gates, you are violating project rules.**

## MANDATORY: Unit Testing Protocol

### When writing new code:
- Write unit tests for every new method BEFORE considering the task done
- Tests go in `tests/unit/` following existing patterns (see `tests/conftest.py` for fixtures)
- Mock ModelManager and external dependencies — test logic in isolation
- Use `make_*` helpers from `conftest.py` for test data
- Each test tests ONE behavior — name it `test_<what_it_checks>`

### When modifying existing code:
- Run `uv run python -m pytest tests/ -v` IMMEDIATELY after changes
- Fix broken tests by updating assertions to match the new behavior
- Do NOT delete tests — update them to reflect the new contract
- If a method's contract changes, add new tests for the new behavior

### Test structure convention:
```
tests/unit/test_<module>.py     — tests for src/<module>.py
tests/unit/test_<module>_<area>.py — tests for a specific area of a module
```

## Critical: Documentation Sync Requirements

**When modifying the pipeline, you MUST update these files:**

### 1. Pipeline Changes (`src/pipeline.py`)
If you add, remove, or modify any pipeline step:
- [ ] Update `src/ARCHITECTURE.md` - Pipeline Flow section
- [ ] Update `src/reporter.py` - Methodology tab (Pipeline Steps section around line 770)
- [ ] Update config.py if new parameters are needed
- [ ] Update main.py CLI arguments if user-facing options change

### 2. Model Changes (classifier, VLM, editor)
If you change default models or add new model support:
- [ ] Update `src/ARCHITECTURE.md` - Configuration Options section
- [ ] Update `src/reporter.py` - Methodology tab (Configuration table)
- [ ] Update `src/config.py` - default values and comments
- [ ] Update `main.py` - CLI help text

### 3. Report Changes (`src/reporter.py`)
If you modify the report structure:
- [ ] Update both HTML and Markdown templates
- [ ] Ensure config is passed and displayed in Methodology tab
- [ ] Test report generation with existing data

### 4. New Features
When adding new features:
- [ ] Add docstrings explaining the feature
- [ ] Update `src/ARCHITECTURE.md` with new module/class info
- [ ] Add CLI flags if user-controllable
- [ ] Update report to show feature results if applicable

## File Purposes

| File | Purpose | Update When |
|------|---------|-------------|
| `src/ARCHITECTURE.md` | Technical documentation for developers | Pipeline, models, or config changes |
| `src/reporter.py` | Report generation with Methodology tab | Pipeline steps, config, or output changes |
| `src/config.py` | Configuration parameters | New options or default changes |
| `main.py` | CLI entry point | New user-facing options |
| `CLAUDE.md` | This file - Claude instructions | New documentation requirements |

## Pipeline Overview (Keep in Sync)

Current pipeline steps:
1. **Sample Collection** - `dataset.py` - Get positive/negative samples
2. **Baseline Classification** - `classifier.py` - Classify + attention maps
3. **Feature Discovery** - `vlm.py` - VLM analyzes features, generates hypotheses
4. **Counterfactual Editing** - `editor.py` - Apply edits (Qwen-Image-Edit)
5. **Edit Verification** - `vlm.py` - VLM confirms edit was applied
6. **Impact Measurement + Attention Diff** - `classifier.py` + `statistical_validator.py` + `attention_maps.py` - Measure delta, Grad-CAM on edited images, attention diff heatmaps
7. **Report Generation** - `reporter.py` - Generate HTML/MD/JSON reports

## Code Style Guidelines

### VLM Prompts
- Edit instructions must be SPECIFIC and DECISIVE
- Never use "or", "such as", "for example" in generated edits
- Each edit = ONE concrete action

### Error Handling
- Use try/except for model operations (OOM common)
- Log warnings for non-fatal issues
- Include `_handle_oom()` fallback for GPU memory

### VRAM Management
- Support both low_vram, batch, and high_vram modes
- Implement `offload()` and `load_to_gpu()` for large models
- All model wrappers use `self.loaded` boolean flag
- `ModelManager._ensure_only()` auto-offloads other models
- Clear cache with `torch.cuda.empty_cache()` after offload

## Verification Commands

```bash
# Run unit tests (MANDATORY after every code change)
uv run python -m pytest tests/ -v

# Syntax check
python -m py_compile src/pipeline.py src/vlm.py src/editor.py src/reporter.py

# Generate report with existing data
python -c "
from pathlib import Path
import json
from src.reporter import Reporter
from src.config import Config

output_dir = Path('output')
results = [json.load(open(f)) for f in output_dir.glob('*/analysis.json')]
Reporter(output_dir, Config().model_dump()).generate_all(results)
print('Report generated successfully')
"

# Quick test run (if models available)
python main.py --classes 1 --samples 1 --iterations 1 --generations 1
```
