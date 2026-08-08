"""VisionTag CLI — auto-tag images with Vertex AI Gemini."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from vision_tag.env import default_examples_images_dir, default_schema_path, load_env, package_root
from vision_tag.label_studio.import_builder import build_import_json
from vision_tag.label_studio.local_files import fix_local_files
from vision_tag.label_studio.xml_generator import generate_label_studio_xml
from vision_tag.schema import load_schema
from vision_tag.vertex import RunConfig, run_labeling


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def cmd_run(args: argparse.Namespace) -> int:
    schema = load_schema(Path(args.schema))
    load_env()

    config = RunConfig(
        images_dir=Path(args.images_dir),
        output_path=Path(args.output),
        checkpoint_path=Path(args.checkpoint),
        schema=schema,
        project=args.project or os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        location=args.location or os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        model=args.model or os.environ.get("VERTEX_MODEL", "gemini-2.5-flash"),
        limit=args.limit,
        sleep=args.sleep if args.sleep is not None else _float_env("VISIONTAG_SLEEP", 1.0),
        retry_passes=args.retry_passes
        if args.retry_passes is not None
        else _int_env("VISIONTAG_RETRY_PASSES", 5),
        retry_sleep=args.retry_sleep
        if args.retry_sleep is not None
        else _float_env("VISIONTAG_RETRY_SLEEP", 45.0),
        retry_attempts=args.retry_attempts
        if args.retry_attempts is not None
        else _int_env("VISIONTAG_RETRY_ATTEMPTS", 4),
    )
    return run_labeling(config)


def cmd_generate_label_studio_config(args: argparse.Namespace) -> int:
    schema = load_schema(Path(args.schema))
    xml = generate_label_studio_xml(schema)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(xml, encoding="utf-8")
    print(f"Wrote Label Studio config to {output}")
    return 0


def cmd_build_label_studio_import(args: argparse.Namespace) -> int:
    schema = load_schema(Path(args.schema))
    labels_csv = Path(args.labels_csv)
    if not labels_csv.exists():
        print(f"Error: labels CSV not found: {labels_csv}", file=sys.stderr)
        return 1

    project_root = Path(args.project_root).resolve()
    images_dir = Path(args.images_dir).resolve()
    output = Path(args.output)

    total, labeled = build_import_json(
        labels_csv=labels_csv,
        output_path=output,
        schema=schema,
        images_dir=images_dir,
        project_root=project_root,
        include_unlabeled=args.include_unlabeled,
    )
    print(f"Wrote {total} task(s) ({labeled} with predictions) to {output}")
    return 0


def cmd_fix_local_files(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    images_dir = Path(args.images_dir).resolve()
    for message in fix_local_files(project_root, images_dir, project_id=args.project_id):
        print(message)
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    root = package_root()
    schema_path = Path(args.schema)
    labels_csv = Path(args.labels_csv)
    import_json = Path(args.output_json)
    config_xml = Path(args.config_xml)

    if not args.skip_labeling:
        run_args = argparse.Namespace(
            schema=str(schema_path),
            images_dir=args.images_dir,
            output=str(labels_csv),
            checkpoint=args.checkpoint,
            project=args.project,
            location=args.location,
            model=args.model,
            limit=args.limit,
            sleep=args.sleep,
            retry_passes=args.retry_passes,
            retry_sleep=args.retry_sleep,
            retry_attempts=args.retry_attempts,
        )
        code = cmd_run(run_args)
        if code != 0 and not labels_csv.exists():
            return code
    elif not labels_csv.exists():
        print(f"Error: no labels CSV at {labels_csv}. Run without --skip-labeling.", file=sys.stderr)
        return 1

    gen_args = argparse.Namespace(schema=str(schema_path), output=str(config_xml))
    cmd_generate_label_studio_config(gen_args)

    import_args = argparse.Namespace(
        labels_csv=str(labels_csv),
        output=str(import_json),
        schema=str(schema_path),
        images_dir=args.images_dir,
        project_root=str(root),
        include_unlabeled=False,
    )
    cmd_build_label_studio_import(import_args)

    print()
    print("=" * 60)
    print("PIPELINE READY — review in Label Studio")
    print("=" * 60)
    print()
    print("1. Fix local image serving:")
    print("     visiontag fix-local-files")
    print()
    print("2. Start Label Studio:")
    print("     label-studio")
    print()
    print("3. Create/open project and paste config:")
    print(f"     {config_xml.resolve()}")
    print()
    print("4. Enable predictions in project Settings:")
    print("     Turn ON 'Use predictions to prelabel tasks'")
    print()
    print("5. Import tasks with AI pre-labels:")
    print(f"     {import_json.resolve()}")
    print()
    print("6. Review each task, fix wrong labels, Submit")
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = package_root()
    default_schema = str(default_schema_path())
    default_images = str(default_examples_images_dir())

    parser = argparse.ArgumentParser(
        prog="visiontag",
        description="VisionTag — auto-tag images with Vertex AI Gemini",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Label images with Vertex AI Gemini")
    run.add_argument("--schema", default=default_schema, help="Path to label_schema.yaml")
    run.add_argument("--images-dir", default=default_images, help="Folder of images to label")
    run.add_argument("-o", "--output", default="labels.csv", help="Output CSV path")
    run.add_argument("--checkpoint", default="labels_checkpoint.jsonl", help="Checkpoint JSONL path")
    run.add_argument("--project", default=None, help="GCP project ID (or GOOGLE_CLOUD_PROJECT)")
    run.add_argument("--location", default=None, help="Vertex region (default: global)")
    run.add_argument("--model", default=None, help="Gemini model (default: gemini-2.5-flash)")
    run.add_argument("--limit", type=int, default=None, help="Max images to process")
    run.add_argument("--sleep", type=float, default=None, help="Seconds between images (default: 1.0)")
    run.add_argument("--retry-passes", type=int, default=None, help="End-of-run retry passes (default: 5)")
    run.add_argument("--retry-sleep", type=float, default=None, help="429 backoff base seconds (default: 45)")
    run.add_argument("--retry-attempts", type=int, default=None, help="API attempts per retry pass (default: 4)")
    run.set_defaults(func=cmd_run)

    pipeline = subparsers.add_parser("pipeline", help="Label images then build Label Studio import")
    pipeline.add_argument("--schema", default=default_schema)
    pipeline.add_argument("--images-dir", default=default_images)
    pipeline.add_argument("--labels-csv", default="labels.csv")
    pipeline.add_argument("--checkpoint", default="labels_checkpoint.jsonl")
    pipeline.add_argument("--output-json", default="tasks_with_predictions.json")
    pipeline.add_argument("--config-xml", default="label_studio_config.xml")
    pipeline.add_argument("--project", default=None)
    pipeline.add_argument("--location", default=None)
    pipeline.add_argument("--model", default=None)
    pipeline.add_argument("--limit", type=int, default=None)
    pipeline.add_argument("--skip-labeling", action="store_true")
    pipeline.add_argument("--sleep", type=float, default=None)
    pipeline.add_argument("--retry-passes", type=int, default=None)
    pipeline.add_argument("--retry-sleep", type=float, default=None)
    pipeline.add_argument("--retry-attempts", type=int, default=None)
    pipeline.set_defaults(func=cmd_pipeline)

    gen = subparsers.add_parser(
        "generate-label-studio-config",
        help="Generate Label Studio XML from schema YAML",
    )
    gen.add_argument("--schema", default=default_schema)
    gen.add_argument("-o", "--output", default="label_studio_config.xml")
    gen.set_defaults(func=cmd_generate_label_studio_config)

    build_import = subparsers.add_parser(
        "build-label-studio-import",
        help="Build Label Studio import JSON with AI predictions",
    )
    build_import.add_argument("labels_csv", help="Path to labels.csv from visiontag run")
    build_import.add_argument("-o", "--output", default="tasks_with_predictions.json")
    build_import.add_argument("--schema", default=default_schema)
    build_import.add_argument("--images-dir", default=default_images)
    build_import.add_argument("--project-root", default=str(root), help="Label Studio document root")
    build_import.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Include all images from images-dir, not only CSV rows",
    )
    build_import.set_defaults(func=cmd_build_label_studio_import)

    fix = subparsers.add_parser("fix-local-files", help="Configure Label Studio local file serving")
    fix.add_argument("--project-root", default=str(root))
    fix.add_argument("--images-dir", default=default_images)
    fix.add_argument("--project-id", type=int, default=1, help="Label Studio project ID")
    fix.set_defaults(func=cmd_fix_local_files)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
