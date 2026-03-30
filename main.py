"""
CLI entry point for the Coralation image classification analysis tool.

Usage:
    uv run python main.py                        # analyze 30 default classes
    uv run python main.py --classes 5            # analyze first 5 classes
    uv run python main.py --class-names "cat" "dog"   # specific classes
    uv run python main.py --all                  # analyze all 1000 ImageNet classes
    uv run python main.py --output-dir ./results --samples 3
"""
from __future__ import annotations

# Suppress progress bars and warnings BEFORE importing any ML libraries
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["SAFETENSORS_FAST_GPU"] = "1"
# Note: TQDM_DISABLE is NOT set - we want our own tqdm progress bars to work

import argparse
import dataclasses
import json
import logging
import sys
import warnings
from pathlib import Path

import tqdm

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from src.config import Config, load_hf_token, load_classes_from_file
from src.pipeline import AnalysisPipeline
from src.reporter import Reporter


class TqdmLoggingHandler(logging.Handler):
    """Routes log output through tqdm.write() to avoid corrupting progress bars."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_logging(level: str = "INFO", debug: bool = False, verbose: bool = False, log_file: str = "coralation.log"):
    """Configure logging with clean console output and detailed file logging.

    Args:
        level: Base log level for console (default: INFO)
        debug: If True, write DEBUG logs to file (always on if log_file specified)
        verbose: If True, also show DEBUG logs in console
        log_file: Path to log file for detailed logging
    """
    # Console level: DEBUG only if verbose, otherwise use specified level
    console_level = logging.DEBUG if verbose else getattr(logging, level.upper(), logging.INFO)

    # Console handler - routes through tqdm.write() to avoid breaking progress bars
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(console_level)
    if verbose:
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        ))
    else:
        console_handler.setFormatter(logging.Formatter("%(message)s"))

    # File handler - always detailed format, always DEBUG level
    file_handler = logging.FileHandler(log_file, mode="w")  # Overwrite each run
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Silence noisy loggers
    noisy_loggers = [
        "httpx", "httpcore", "urllib3", "huggingface_hub", "huggingface_hub.file_download",
        "datasets", "datasets.builder", "datasets.info", "datasets.arrow_dataset",
        "transformers", "transformers.modeling_utils", "transformers.configuration_utils",
        "transformers.tokenization_utils_base", "transformers.generation",
        "diffusers", "diffusers.pipelines", "diffusers.models",
        "PIL", "PIL.PngImagePlugin", "PIL.TiffImagePlugin",
        "torch", "torch.distributed", "torch.nn.parallel",
        "accelerate", "accelerate.utils", "filelock", "fsspec",
        "qwen_vl_utils", "timm", "safetensors",
    ]
    for noisy in noisy_loggers:
        logging.getLogger(noisy).setLevel(logging.ERROR)

    # Suppress transformers and diffusers internal logging
    import transformers
    import diffusers
    transformers.logging.set_verbosity_error()
    diffusers.logging.set_verbosity_error()

    # Disable datasets progress bars
    try:
        import datasets
        datasets.disable_progress_bar()
    except (ImportError, AttributeError):
        pass


def resolve_classes(args, pipeline, cfg):
    """Determine which classes to analyze based on CLI arguments."""
    if args.class_names:
        return args.class_names
    all_names = _load_all_class_names(pipeline, cfg)
    if args.all:
        return all_names
    return _select_n_classes(all_names, args.classes, cfg)


def _load_all_class_names(pipeline, cfg):
    """Load class names from JSON file or dataset based on config."""
    if cfg.class_source == "json":
        return load_classes_from_file(cfg.class_file)
    return pipeline.models.sampler().get_label_names()


def _select_n_classes(all_names, n, cfg):
    """Select N classes from the full list, optionally randomized."""
    import random
    if not cfg.random_classes:
        return all_names[:n]
    seed = cfg.random_seed if cfg.random_seed is not None else random.randrange(2**32)
    rng = random.Random(seed)
    return rng.sample(all_names, min(n, len(all_names)))


def main():
    parser = argparse.ArgumentParser(
        description="Coralation: Automated bias/shortcut discovery for image classifiers"
    )

    # Class selection
    class_group = parser.add_mutually_exclusive_group()
    class_group.add_argument(
        "--classes", type=int, default=100,
        help="Number of ImageNet-100 classes to analyze (default: 100)"
    )
    class_group.add_argument(
        "--class-names", nargs="+", metavar="CLASS",
        help="Specific class names to analyze (e.g. 'tabby cat' 'golden retriever')"
    )
    class_group.add_argument(
        "--all", action="store_true",
        help="Analyze all 100 ImageNet-100 classes"
    )

    # Class source
    parser.add_argument(
        "--class-source", choices=["json", "dataset"], default="json",
        help="Where to read class names: 'json' from file or 'dataset' from HF dataset (default: json)"
    )
    parser.add_argument(
        "--class-file", type=str, default=None,
        help="Path to JSON class list file (default: src/imagenet100_classes.json)"
    )

    # Config overrides
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument(
        "--samples", type=int, default=10,
        help="Number of positive samples per class (used for inspection AND editing, default: 10)"
    )
    parser.add_argument(
        "--top-negative-classes", type=int, default=5,
        help="Number of confusing classes to select (default: 5)"
    )
    parser.add_argument(
        "--negative-samples", type=int, default=5,
        help="Negative images to sample from each confusing class (default: 5)"
    )
    parser.add_argument(
        "--delta-threshold", type=float, default=0.15,
        help="Min confidence delta to confirm a hypothesis (default: 0.15)"
    )
    parser.add_argument(
        "--max-hypotheses", type=int, default=3,
        help="Max edit instructions per image from VLM (default: 3)"
    )
    parser.add_argument(
        "--classifier", default="resnet50",
        help="Single classifier model (default: resnet50). Ignored if --classifiers is set."
    )
    parser.add_argument(
        "--classifiers", nargs="+", default=["resnet50", "dinov2_vitb14_lc", "vit_l_16"],
        help="Classifier models to compare (default: all three: resnet50 dinov2_vitb14_lc vit_l_16)"
    )
    parser.add_argument(
        "--vlm", default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="VLM model name (default: Qwen/Qwen2.5-VL-7B-Instruct)"
    )
    parser.add_argument(
        "--editor", default="black-forest-labs/FLUX.2-klein-9b-kv",
        help="Image editor model (default: FLUX.2-klein, 4 steps)"
    )
    parser.add_argument(
        "--iterations", type=int, default=2,
        help="Number of VLM analysis iterations per class (default: 2)"
    )
    parser.add_argument(
        "--generations", type=int, default=1,
        help="Number of image generations per edit (default: 1)"
    )
    parser.add_argument(
        "--low-vram", action="store_true", default=True,
        help="Load/offload models one at a time to save VRAM (default: True)"
    )
    parser.add_argument(
        "--high-vram", action="store_true",
        help="Keep all models loaded (requires 40GB+ VRAM)"
    )
    parser.add_argument(
        "--attention", choices=["gradcam", "gradcam++", "scorecam"],
        default="scorecam",
        help="Attention map method: gradcam, gradcam++, or scorecam (default: scorecam)"
    )
    parser.add_argument(
        "--no-stats", action="store_true",
        help="Disable statistical validation (use simple threshold instead)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Enable VLM verification of edits (disabled by default, slower)"
    )
    parser.add_argument(
        "--random-classes", action="store_true",
        help="Randomly select classes from dataset instead of taking first N"
    )
    parser.add_argument("--log-level", default="INFO", help="Console logging level (default: INFO)")
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging to file (detailed logs saved to --log-file)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show debug logs in console as well (implies --debug)"
    )
    parser.add_argument(
        "--log-file", default="coralation.log",
        help="Log file path (default: coralation.log)"
    )
    parser.add_argument(
        "--hf-token", default=None,
        help="HuggingFace token (or set HF_TOKEN env var)"
    )

    args = parser.parse_args()
    # --verbose implies --debug
    if args.verbose:
        args.debug = True
    setup_logging(args.log_level, debug=args.debug, verbose=args.verbose, log_file=args.log_file)
    logger = logging.getLogger("coralation")

    # Build config
    low_vram = not args.high_vram  # --high-vram overrides default low_vram=True
    class_file_overrides = {}
    if args.class_file:
        class_file_overrides["class_file"] = Path(args.class_file)
    # Resolve classifier models: --classifiers takes precedence over --classifier
    classifier_models = args.classifiers if args.classifiers else [args.classifier]
    cfg = Config(
        classifier_model=classifier_models[0],
        classifier_models=classifier_models,
        vlm_model=args.vlm,
        editor_model=args.editor,
        output_dir=Path(args.output_dir),
        samples_per_class=args.samples,
        top_negative_classes=args.top_negative_classes,
        negative_samples_per_class=args.negative_samples,
        confidence_delta_threshold=args.delta_threshold,
        max_hypotheses_per_image=args.max_hypotheses,
        iterations=args.iterations,
        generations_per_edit=args.generations,
        low_vram=low_vram,
        attention_method=args.attention,
        use_statistical_validation=not args.no_stats,
        random_classes=args.random_classes,
        verify_edits=args.verify,
        hf_token=args.hf_token or load_hf_token(),
        class_source=args.class_source,
        **class_file_overrides,
    )

    # Print header
    print("\n" + "=" * 60)
    print("  CORALATION - Model Bias/Shortcut Discovery")
    print("=" * 60)
    print(f"  Output:     {cfg.output_dir}")
    print(f"  Samples:    {cfg.samples_per_class} positive, {cfg.negative_samples_per_class}x{cfg.top_negative_classes} negative")
    print(f"  Iterations: {cfg.iterations} per class")
    print(f"  Classifiers: {', '.join(cfg.classifier_models)}")
    print(f"  Threshold:  {cfg.confidence_delta_threshold}")
    print(f"  Generations: {cfg.generations_per_edit} per edit")
    print(f"  Attention:  {cfg.attention_method}")
    print(f"  Validation: {'statistical (t-test + effect size)' if cfg.use_statistical_validation else 'threshold-based'}")
    print(f"  VRAM mode:  {'low (sequential loading)' if cfg.low_vram else 'high (all models loaded)'}")
    print(f"  Pipeline:   phase-first (6 model swaps total)")
    print("=" * 60 + "\n")

    pipeline = AnalysisPipeline(cfg)

    # Determine which classes to analyze
    classes = resolve_classes(args, pipeline, cfg)

    class_list = " | ".join(classes[:10])
    print(f"Analyzing {len(classes)} classes: {class_list}"
          f"{'...' if len(classes) > 10 else ''}\n")

    # Run analysis with error handling to ensure reports are always generated
    from datetime import datetime, timezone
    start_time = datetime.now(timezone.utc)
    analysis_error = None

    if cfg.is_multi_model:
        multi_results = _run_multi_model(pipeline, classes, cfg, start_time)
    else:
        _run_single_model(pipeline, classes, cfg, start_time)


def _run_single_model(pipeline, classes, cfg, start_time):
    """Run single-model analysis and generate reports."""
    from datetime import datetime, timezone
    results = []
    analysis_error = None
    try:
        results = pipeline.run(classes)
    except Exception as e:
        analysis_error = e
        logger.error("Analysis pipeline failed: %s", e, exc_info=True)
        print(f"\n⚠ Analysis error: {e}")
        print("Generating reports with available results...\n")

    end_time = datetime.now(timezone.utc)
    config_dict = _build_config_dict(cfg, start_time, end_time)
    reporter = Reporter(cfg.reports_dir, config=config_dict)

    results_dicts = _safe_convert_results(results)
    paths = _generate_reports(reporter, results_dicts, cfg)
    _print_single_summary(results, paths, analysis_error)


def _run_multi_model(pipeline, classes, cfg, start_time):
    """Run multi-model analysis and generate per-model + comparison reports."""
    from datetime import datetime, timezone
    from src.pipeline import MultiModelResult
    analysis_error = None
    multi_results = []
    try:
        multi_results = pipeline.run_multi_model(classes)
    except Exception as e:
        analysis_error = e
        logger.error("Multi-model pipeline failed: %s", e, exc_info=True)
        print(f"\n⚠ Analysis error: {e}")

    end_time = datetime.now(timezone.utc)
    config_dict = _build_config_dict(cfg, start_time, end_time)

    # Generate per-model reports
    all_model_dicts: dict[str, list[dict]] = {}
    for model_name in cfg.classifier_models:
        model_results = [mr.per_model.get(model_name) for mr in multi_results
                         if model_name in mr.per_model]
        model_dicts = _safe_convert_results(model_results)
        all_model_dicts[model_name] = model_dicts

        model_config = dict(config_dict, classifier_model=model_name)
        reporter = Reporter(cfg.reports_dir, config=model_config, prefix=model_name)
        _generate_reports(reporter, model_dicts, cfg, cfg.reports_dir)
        print(f"  {model_name}: report saved to {cfg.reports_dir / f'{model_name}_report.html'}")

    # Generate comparison report
    comparison_reporter = Reporter(cfg.reports_dir, config=config_dict)
    comparison_reporter.generate_comparison(all_model_dicts, cfg.classifier_models)

    _print_multi_summary(multi_results, cfg, analysis_error)


def _build_config_dict(cfg, start_time, end_time):
    """Build config dict with timing info for reports."""
    config_dict = cfg.model_dump() if hasattr(cfg, 'model_dump') else cfg.dict()
    duration = end_time - start_time
    config_dict["timing"] = {
        "start": start_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "end": end_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "duration_seconds": int(duration.total_seconds()),
    }
    return config_dict


def _safe_convert_results(results):
    """Convert ClassAnalysisResult list to dicts, skipping failures."""
    dicts = []
    for r in results:
        if r is None:
            continue
        try:
            dicts.append(r.to_dict())
        except Exception as e:
            logger.warning("Failed to convert result: %s", e)
    return dicts


def _generate_reports(reporter, results_dicts, cfg, output_dir=None):
    """Generate HTML/MD/JSON reports, returning paths."""
    try:
        return reporter.generate_all(results_dicts)
    except Exception as e:
        logger.error("Failed to generate reports: %s", e, exc_info=True)
        d = output_dir or cfg.reports_dir
        d.mkdir(parents=True, exist_ok=True)
        paths = {
            'html': d / 'report.html',
            'markdown': d / 'report.md',
            'json': d / 'analysis_results.json',
        }
        paths['html'].write_text("<html><body><h1>Report generation failed</h1></body></html>")
        paths['json'].write_text('[]')
        paths['markdown'].write_text("# Report generation failed\n")
        return paths


def _print_single_summary(results, paths, analysis_error):
    """Print summary for single-model run."""
    total_confirmed = sum(len(r.confirmed_hypotheses) for r in results)
    total_edits = sum(len(r.edit_results) for r in results)

    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Classes analyzed:    {len(results)}")
    print(f"  Total edits tested:  {total_edits}")
    print(f"  Shortcuts confirmed: {total_confirmed}")
    if analysis_error:
        print(f"  ⚠ Warning: Analysis had errors (see log)")
    print("-" * 60)
    print(f"  Reports saved to:")
    print(f"    HTML:     {paths['html']}")
    print(f"    Markdown: {paths['markdown']}")
    print(f"    JSON:     {paths['json']}")


def _print_multi_summary(multi_results, cfg, analysis_error):
    """Print summary for multi-model run."""
    print("\n" + "=" * 60)
    print("  MULTI-MODEL ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Classes analyzed: {len(multi_results)}")
    print(f"  Models compared:  {', '.join(cfg.classifier_models)}")
    for model_name in cfg.classifier_models:
        model_rs = [mr.per_model.get(model_name) for mr in multi_results
                     if model_name in mr.per_model]
        confirmed = sum(len(r.confirmed_hypotheses) for r in model_rs if r)
        spurious = sum(
            1 for r in model_rs if r
            for e in r.confirmed_hypotheses
            if (e.feature_type == 'contextual' and e.target_type == 'positive' and e.mean_delta < -0.05)
            or (e.target_type == 'negative' and e.mean_delta > 0.05)
        )
        print(f"  {model_name:25s}: {confirmed} confirmed, {spurious} spurious")
    if analysis_error:
        print(f"  ⚠ Warning: Analysis had errors (see log)")
    print("-" * 60)
    print(f"  Reports saved to:")
    for model_name in cfg.classifier_models:
        print(f"    {model_name}: {cfg.reports_dir / f'{model_name}_report.html'}")
    print(f"    Comparison: {cfg.reports_dir / 'comparison_report.html'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import signal
    import traceback

    # Global logger for crash handling
    _crash_logger = None

    def handle_crash(sig_or_exc, frame_or_tb=None):
        """Handle crashes, signals, and uncaught exceptions."""
        global _crash_logger
        if _crash_logger is None:
            _crash_logger = logging.getLogger("coralation.crash")

        if isinstance(sig_or_exc, int):
            # Signal received
            sig_name = signal.Signals(sig_or_exc).name if hasattr(signal, 'Signals') else str(sig_or_exc)
            msg = f"Process interrupted by signal: {sig_name}"
            _crash_logger.error(msg)
            print(f"\n\n{'='*60}\n  INTERRUPTED: {sig_name}\n{'='*60}")
        else:
            # Exception
            msg = f"Uncaught exception: {sig_or_exc}"
            _crash_logger.error(msg, exc_info=(type(sig_or_exc), sig_or_exc, frame_or_tb))
            print(f"\n\n{'='*60}\n  CRASH: {sig_or_exc}\n{'='*60}")
            traceback.print_exception(type(sig_or_exc), sig_or_exc, frame_or_tb)

        # Flush all log handlers
        for handler in logging.root.handlers:
            handler.flush()

        print(f"\nCheck log file for details.\n")
        sys.exit(1)

    def signal_handler(signum, frame):
        handle_crash(signum, frame)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill command

    # Run with top-level exception handling
    try:
        main()
    except KeyboardInterrupt:
        handle_crash(signal.SIGINT)
    except Exception as e:
        # Setup minimal logging if not already done
        if not logging.root.handlers:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler("coralation_crash.log", mode="w")
                ]
            )
        handle_crash(e, e.__traceback__)

