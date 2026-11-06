"""Run one named background job and atomically record its completion status."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/overnight"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    OUT.mkdir(parents=True, exist_ok=True)
    if args.name == "launch":
        name, command = command[0], command[1:]
        with (OUT / (name + ".log")).open("ab") as handle:
            process = subprocess.Popen(["nohup", sys.executable, __file__, name, "--", *command], cwd=ROOT,
                                       stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        print(json.dumps({"pid": process.pid, "name": name}))
    else:
        started = time.time()
        result = subprocess.run(command, cwd=ROOT)
        path = OUT / (args.name + ".status.json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"returncode": result.returncode, "elapsed_s": time.time()-started}))
        temporary.replace(path)
        raise SystemExit(result.returncode)
