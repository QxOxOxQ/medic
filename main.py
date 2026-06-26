from __future__ import annotations

import argparse
import json
import logging

from rag.document_preparation import prepare_documents
from rag.full_process import FullProcess
from rag.setup_project import setup_project

def _configure_logging() -> None:
    from observability.logging_config import configure_logging

    configure_logging()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medic")
    subparsers = parser.add_subparsers(dest="command")
    setup_parser = subparsers.add_parser(
        "setup",
        help="Prepare local project files and ensure the Qdrant collection exists",
    )
    setup_parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Prepare files only, without checking PostgreSQL or Qdrant",
    )
    setup_parser.add_argument(
        "--skip-postgres",
        action="store_true",
        help="Skip PostgreSQL migrations, admin seed, and legacy document import",
    )
    setup_parser.add_argument(
        "--skip-qdrant",
        action="store_true",
        help="Skip checking or creating the Qdrant collection",
    )
    setup_parser.add_argument(
        "--no-create-env",
        action="store_true",
        help="Do not create .env from .env.example when .env is missing",
    )
    subparsers.add_parser("prepare", help="Prepare raw PDF files for later chunking")
    subparsers.add_parser("ingest", help="Prepare and index documents into Qdrant")
    subparsers.add_parser(
        "seed-demo",
        help="Upload and index the synthetic demo PDFs for the dashboard admin user",
    )
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run a blocking RAG quality evaluation",
    )
    evaluate_parser.add_argument("--suite", default="medical-demo-v1")
    evaluate_parser.add_argument("--dataset-version")
    bootstrap_parser = subparsers.add_parser(
        "evaluation-bootstrap-dataset",
        help="Create or verify the synthetic Langfuse dataset",
    )
    bootstrap_parser.add_argument("--suite", default="medical-demo-v1")
    subparsers.add_parser(
        "evaluation-calibrate",
        help="Verify that the configured RAGAS judge separates good and bad answers",
    )
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Run the Medic RAG web dashboard",
    )
    dashboard_parser.add_argument("--host", default="127.0.0.1")
    dashboard_parser.add_argument("--port", default=8000, type=int)
    return parser


def _run_seed_demo() -> int:
    import os

    from dotenv import dotenv_values

    from dashboard.services.demo_seeder import DemoSeedError, seed_demo_documents
    from rag.config import PROJECT_ROOT
    from rag.database import get_session_factory

    dotenv_settings = dotenv_values(PROJECT_ROOT / ".env")
    admin_username = os.getenv("MEDIC_DASHBOARD_USERNAME") or dotenv_settings.get(
        "MEDIC_DASHBOARD_USERNAME"
    )
    if not admin_username:
        print(
            json.dumps(
                {
                    "status": "failed_error",
                    "error": "MEDIC_DASHBOARD_USERNAME is required",
                }
            )
        )
        return 2
    try:
        summary = seed_demo_documents(
            admin_username=admin_username,
            database_session_factory=get_session_factory(),
            documents_dir=PROJECT_ROOT / "demo_documents",
        )
    except DemoSeedError as error:
        print(json.dumps({"status": "failed_error", "error": str(error)}))
        return 2
    print(summary.as_report_line())
    return 1 if summary.pipeline_failed else 0


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "setup":
        setup_summary = setup_project(
            create_env_file=not args.no_create_env,
            setup_database=not (args.skip_db or args.skip_postgres),
            setup_qdrant=not (args.skip_db or args.skip_qdrant),
        )
        print(setup_summary.as_report_line())
        return 0

    if args.command == "prepare":
        preparation_summary = prepare_documents()
        print(preparation_summary.as_report_line())
        return 1 if preparation_summary.failed else 0

    if args.command == "ingest":
        ingestion_summary = FullProcess().execute()
        return 1 if ingestion_summary.failed else 0

    if args.command == "seed-demo":
        return _run_seed_demo()

    if args.command in {
        "evaluate",
        "evaluation-bootstrap-dataset",
        "evaluation-calibrate",
    }:
        from evaluation.factory import (
            build_dataset_bootstrap_service,
            build_evaluation_services,
        )
        from evaluation.application.errors import EvaluationApplicationError
        from evaluation.presentation.cli import (
            run_calibration_cli,
            run_bootstrap_cli,
            run_evaluation_cli,
        )
        from rag.database import get_session_factory

        try:
            if args.command == "evaluation-bootstrap-dataset":
                bootstrap = build_dataset_bootstrap_service()
                return run_bootstrap_cli(bootstrap, profile_id=args.suite)
            services = build_evaluation_services(session_factory=get_session_factory())
        except EvaluationApplicationError as error:
            print(json.dumps({"status": "failed_error", "error": str(error)}))
            return 2
        except Exception as error:
            logging.getLogger(__name__).exception("Evaluation configuration failed")
            print(json.dumps({"status": "failed_error", "error": str(error)}))
            return 2
        if args.command == "evaluate":
            return run_evaluation_cli(
                services,
                profile_id=args.suite,
                dataset_version=args.dataset_version,
            )
        return run_calibration_cli(services)

    if args.command == "dashboard":
        import uvicorn

        from dashboard.app import create_app

        uvicorn.run(
            create_app(),
            host=args.host,
            port=args.port,
            log_config=None,
        )
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
