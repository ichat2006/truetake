"""Commit a green overnight step after auditing the staged path list."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    step, message = sys.argv[1:3]
    status = json.loads((ROOT / "outputs/overnight" / (step + "-tests.status.json")).read_text())
    if status["returncode"] != 0:
        raise SystemExit("Refusing commit: tests did not pass")
    subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True)
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only", "-z"], cwd=ROOT).decode().split("\0")
    bad = [name for name in staged if name and (name.startswith("data/") or any(part == ".env" or part.startswith(".env.") for part in Path(name).parts))]
    if bad:
        raise SystemExit("Unsafe staging detected; refusing commit: " + repr(bad))
    print("Staging audit passed: no .env or data/ paths", flush=True)
    configured = all(subprocess.run(["git", "config", "--get", key], cwd=ROOT,
                                   capture_output=True).returncode == 0 for key in ("user.name", "user.email"))
    identity = [] if configured else ["-c", "user.name=TrueTake overnight automation", "-c", "user.email=overnight@truetake.local"]
    subprocess.run(["git", *identity, "commit", "-m", message], cwd=ROOT, check=True)
