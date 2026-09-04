"""Pre-flight check: is this checkout ready to run?

Answers, in one place, the questions a new clone always raises -- is the PLATO
submodule populated, does its requirements file parse, is an API key visible,
which experiments can actually run. Reports every problem it finds rather than
stopping at the first, and never prints a secret.

    python scripts/check_setup.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OK, WARN, FAIL = "ok", "warn", "fail"
MARKS = {OK: "  OK  ", WARN: " WARN ", FAIL: " FAIL "}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))

    def render(self) -> int:
        width = max(len(name) for _, name, _ in self.rows)
        print()
        for status, name, detail in self.rows:
            print(f"[{MARKS[status]}] {name.ljust(width)}  {detail}")
        failures = sum(1 for s, _, _ in self.rows if s == FAIL)
        warnings = sum(1 for s, _, _ in self.rows if s == WARN)
        print()
        if failures:
            print(f"{failures} blocking problem(s), {warnings} warning(s).")
        elif warnings:
            print(f"Ready, with {warnings} warning(s).")
        else:
            print("Ready.")
        return 1 if failures else 0


def check_python(report: Report) -> None:
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}"
    if (major, minor) >= (3, 10):
        report.add(OK, "Python version", version)
    else:
        report.add(FAIL, "Python version", f"{version} -- this project needs 3.10 or newer")


def check_dependencies(report: Report) -> None:
    for module, package in (
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("openai", "openai"),
        ("PIL", "pillow"),
        ("pytest", "pytest"),
    ):
        if importlib.util.find_spec(module) is None:
            report.add(FAIL, f"dependency: {package}", "missing -- run: pip install -r requirements.txt")
        else:
            report.add(OK, f"dependency: {package}", "")


#: Names people end up with when an editor or Explorer mangles ".env".
#: Windows hides known extensions, so "New > Text Document" renamed to ".env"
#: silently becomes ".env.txt" -- and the code, correctly, does not read it.
MISNAMED_ENV_FILES = (".env.txt", ".env.env", "env", "env.txt", ".env.example.txt")


def check_misnamed_env(report: Report) -> None:
    strays = [name for name in MISNAMED_ENV_FILES if (ROOT / name).is_file()]
    if not strays:
        return
    for name in strays:
        size = (ROOT / name).stat().st_size
        state = "empty" if size == 0 else f"{size} bytes -- MAY CONTAIN YOUR KEY"
        report.add(
            WARN,
            f"misnamed env file: {name}",
            f"{state}. The code only reads '.env'. Rename it, then delete the stray. "
            "(It is gitignored either way.)",
        )


def check_api_key(report: Report) -> None:
    from config import models

    env_file = ROOT / ".env"
    if models.api_key():
        source = ".env" if env_file.is_file() else "environment"
        report.add(OK, "API key", f"present (from {source}); value never printed or logged")
        return

    if env_file.is_file():
        # The file exists but did not yield a key. Say WHY -- "not set" sends
        # people hunting for a missing file that is sitting right there.
        from config import env as env_mod

        text = env_file.read_text(encoding="utf-8-sig", errors="replace")
        names = set(env_mod.parse_env_text(text))
        if not names:
            hint = (
                "the file exists but has no KEY=value lines. It must read "
                "'OPENAI_API_KEY=sk-...', not the bare key on its own"
            )
        elif "OPENAI_API_KEY" not in names:
            hint = f"the file defines {sorted(names)} but not OPENAI_API_KEY"
        else:
            hint = "OPENAI_API_KEY is present but empty"
    elif any((ROOT / name).is_file() for name in MISNAMED_ENV_FILES):
        hint = "a misnamed env file is present -- see the warning above"
    elif not (ROOT / ".env.example").is_file():
        hint = ".env.example is missing too; restore it from git"
    else:
        hint = "copy .env.example to .env (do not type the key into a shell)"
    report.add(WARN, "API key", f"not set -- LLM stages will report as blocked. {hint}")


def check_models(report: Report) -> None:
    from config import models

    for label, resolved in (("cheap tier", models.cheap_model()), ("strong tier", models.strong_model())):
        if resolved in models.RETIRED_MODELS:
            report.add(FAIL, f"model: {label}", f"{resolved} is retired ({models.RETIRED_MODELS[resolved]})")
        else:
            report.add(OK, f"model: {label}", resolved)


def check_secret_hygiene(report: Report) -> None:
    gitignore = ROOT / ".gitignore"
    if not gitignore.is_file():
        report.add(FAIL, "secret hygiene", ".gitignore is missing -- .env could be committed")
        return
    lines = [line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()]
    if ".env" in lines:
        report.add(OK, "secret hygiene", ".env is gitignored")
    else:
        report.add(FAIL, "secret hygiene", ".gitignore does not exclude .env")


def check_plato_submodule(report: Report) -> None:
    plato = ROOT / "PLATO"
    if not plato.is_dir() or not any(plato.iterdir()):
        report.add(
            WARN,
            "PLATO checkout",
            "missing -- clone https://github.com/ArvindCar/PLATO.git to PLATO/, then apply "
            "docs/plato_portability_fixes.patch (see README). Not yet a real submodule; no "
            "fork exists to register one against.",
        )
        return

    git_dir = plato / ".git"
    if git_dir.exists():
        report.add(
            OK if (ROOT / "docs" / "plato_portability_fixes.patch").is_file() else WARN,
            "PLATO status",
            "present as a local checkout (deliberately NOT a submodule yet -- no fork exists "
            "to register one against; it is root-.gitignore'd). See docs/progress.md.",
        )

    requirements = plato / "PLATO" / "requirements.txt"
    if not requirements.is_file():
        report.add(WARN, "PLATO submodule", "present but requirements.txt not found")
        return

    try:
        from pip._vendor.packaging.requirements import Requirement
    except ImportError:
        report.add(WARN, "PLATO requirements", "cannot validate (pip internals unavailable)")
        return

    bad = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        spec = line.split("#")[0].strip()
        if not spec or spec.startswith("git+"):
            continue
        try:
            Requirement(spec)
        except Exception:
            bad.append(spec)
    if bad:
        report.add(FAIL, "PLATO requirements", f"unparseable by pip: {bad}")
    else:
        report.add(OK, "PLATO requirements", "parses")

    nested = [p.name for p in (plato / "grasping",).__iter__() if p.is_dir() and not any(p.iterdir())]
    if nested:
        report.add(WARN, "PLATO nested submodules", "grasping/os_tog empty -- real grasping unavailable")


def check_robomail_submodule(report: Report) -> None:
    """robomail IS a real submodule (rumilog/robomail exists and is public),
    unlike PLATO above. It only needs `git submodule update --init` to populate."""
    robomail = ROOT / "third_party" / "robomail"
    package = robomail / "robomail" / "__init__.py"
    if package.is_file():
        report.add(OK, "robomail submodule", "populated")
    elif robomail.is_dir():
        report.add(
            WARN,
            "robomail submodule",
            "registered but empty -- run: git submodule update --init third_party/robomail",
        )
    else:
        report.add(WARN, "robomail submodule", "not present -- run: git submodule update --init --recursive")


def check_fixtures(report: Report) -> None:
    fixtures = ROOT / "tests" / "fixtures"
    pngs = list(fixtures.glob("*.png"))
    if len(pngs) >= 28:
        report.add(OK, "image fixtures", f"{len(pngs)} present")
    else:
        report.add(
            WARN,
            "image fixtures",
            f"{len(pngs)} found, expected 28 -- run: python tests/fixtures/make_fixtures.py",
        )


def check_experiments(report: Report) -> None:
    import run_experiment

    runnable, blocked = [], []
    for key, spec in sorted(run_experiment.EXPERIMENTS.items()):
        if spec.default_page and Path(spec.default_page).is_file():
            runnable.append(key)
        else:
            blocked.append(key)
    report.add(OK, "experiments runnable", ", ".join(runnable) or "none")
    if blocked:
        report.add(
            WARN,
            "experiments blocked",
            f"{', '.join(blocked)} -- no booklet page photo yet; pass one with --page",
        )


def check_hardware(report: Report) -> None:
    from hardware.factory import resolve_mode

    try:
        mode = resolve_mode()
    except ValueError as exc:
        report.add(FAIL, "hardware mode", str(exc))
        return
    if mode == "mock":
        report.add(OK, "hardware mode", "mock -- no robot needed")
    else:
        report.add(WARN, "hardware mode", "real -- frankapy and a RealSense camera must be attached")


def main() -> int:
    print("Autonomous Robotic Chemist -- setup check")
    print(f"repository: {ROOT}")
    report = Report()
    for check in (
        check_python,
        check_dependencies,
        check_misnamed_env,
        check_api_key,
        check_models,
        check_secret_hygiene,
        check_plato_submodule,
        check_robomail_submodule,
        check_fixtures,
        check_experiments,
        check_hardware,
    ):
        try:
            check(report)
        except Exception as exc:  # a broken check must not hide the other results
            report.add(FAIL, check.__name__, f"check itself failed: {type(exc).__name__}: {exc}")
    return report.render()


if __name__ == "__main__":
    raise SystemExit(main())
