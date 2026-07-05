"""
CI/CD Pipeline — Local & CI Execution
───────────────────────────────────────────────────────────────────────────────
Orchestrates the CI/CD lifecycle:
  - Install: Validate dependencies
  - Lint: ruff format + ruff check
  - Type Check: mypy
  - Test: pytest with coverage
  - Build: Docker image
  - Deploy: Databricks Asset Bundles

Run locally:
    python scripts/ci_pipeline.py                    # full pipeline
    python scripts/ci_pipeline.py --stage lint       # single stage
    python scripts/ci_pipeline.py --skip-deploy      # all except deploy

In CI (GitHub Actions), each stage runs as an independent job.
This script provides the same checks for local development.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], cwd: Path = ROOT, label: str = "") -> bool:
    """Run a command, print output, return True on success."""
    label = label or cmd[0]
    print(f"\n{'=' * 60}")
    print(f"▶ {label}")
    print(f"{'=' * 60}")
    start = time.time()
    result = subprocess.run(cmd, cwd=cwd, capture_output=False, text=True)
    elapsed = time.time() - start
    status = "✅" if result.returncode == 0 else "❌"
    print(f"  {status} {label} finished in {elapsed:.1f}s (exit={result.returncode})")
    return result.returncode == 0


def stage_install() -> bool:
    """Validate dependencies are installable."""
    return _run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet", "--dry-run"],
        label="Install dependencies (dry-run)",
    )


def stage_lint() -> bool:
    """Run ruff format check and ruff lint."""
    ok = True
    ok &= _run(
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--check",
            "src/",
            "tests/",
            "orchestration/",
            "scripts/",
        ],
        label="ruff format check",
    )
    ok &= _run(
        [sys.executable, "-m", "ruff", "check", "src/", "tests/", "orchestration/", "scripts/"],
        label="ruff lint",
    )
    return ok


def stage_typecheck() -> bool:
    """Run mypy type checker."""
    return _run(
        [sys.executable, "-m", "mypy", "src/", "orchestration/", "scripts/"],
        label="mypy type check",
    )


def stage_test() -> bool:
    """Run pytest with coverage."""
    return _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "--cov=src",
            "--cov-report=term-missing",
        ],
        label="pytest",
    )


def stage_build() -> bool:
    """Build Docker image (skip if Docker unavailable)."""
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ⏭ Docker not available — skipping build")
        return True
    return _run(
        ["docker", "build", "-t", "mlops-platform:ci", "."],
        label="Docker build",
    )


def stage_terraform() -> bool:
    """Validate Terraform configuration."""
    tf_dir = ROOT / "infrastructure" / "terraform"
    ok = True
    ok &= _run(
        ["terraform", "init", "-backend=false"],
        cwd=tf_dir,
        label="terraform init",
    )
    ok &= _run(
        ["terraform", "validate"],
        cwd=tf_dir,
        label="terraform validate",
    )
    ok &= _run(
        ["terraform", "fmt", "-check", "-diff"],
        cwd=tf_dir,
        label="terraform fmt check",
    )
    return ok


def stage_deploy(target: str = "dev") -> bool:
    """Deploy with Databricks Asset Bundles (requires credentials)."""
    try:
        subprocess.run(["databricks", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ⏭ Databricks CLI not available — skipping deploy")
        return True
    ok = True
    ok &= _run(
        ["databricks", "bundle", "validate", "-t", target],
        label="databricks bundle validate",
    )
    ok &= _run(
        ["databricks", "bundle", "deploy", "-t", target],
        label="databricks bundle deploy",
    )
    return ok


def main():
    parser = argparse.ArgumentParser(description="MLOps CI/CD Pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "install", "lint", "typecheck", "test", "build", "terraform", "deploy"],
        default="all",
        help="Run a single stage",
    )
    parser.add_argument("--skip-deploy", action="store_true", help="Skip the deploy stage")
    parser.add_argument(
        "--deploy-target",
        default="dev",
        choices=["dev", "staging", "prod"],
        help="Databricks deployment target",
    )
    args = parser.parse_args()

    if args.stage != "all":
        stages = [args.stage]
    else:
        stages = ["install", "lint", "typecheck", "test", "build", "terraform"]
        if not args.skip_deploy:
            stages.append("deploy")

    results = {}
    for stage_name in stages:
        fn_name = f"stage_{stage_name}"
        fn = globals().get(fn_name)
        if fn is None:
            print(f"Unknown stage: {stage_name}")
            results[stage_name] = False
            continue
        if stage_name == "deploy":
            results[stage_name] = fn(args.deploy_target)
        else:
            results[stage_name] = fn()

    print(f"\n{'=' * 60}")
    print("CI/CD Pipeline — Summary")
    print(f"{'=' * 60}")
    all_ok = True
    for stage_name, ok in results.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {stage_name}")
        all_ok &= ok

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
