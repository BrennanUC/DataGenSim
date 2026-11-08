"""Source-checkout command-line launcher."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src")]
from cropvolume.cli import main

if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"cropvolume: {error}", file=sys.stderr)
        raise SystemExit(2)
