import sys
from importlib.metadata import PackageNotFoundError, version

BANNER = r"""
   _                    _   ____       _          _
  / \   __ _  ___ _ __ | |_|  _ \ _   _| |__  _ __(_) ___
 / _ \ / _` |/ _ \ '_ \| __| |_) | | | | '_ \| '__| |/ __|
/ ___ \ (_| |  __/ | | | |_|  _ <| |_| | |_) | |  | | (__
/_/   \_\__, |\___|_| |_|\__|_| \_\\__,_|_.__/|_|  |_|\___|
        |___/
"""


def _version() -> str:
    try:
        return version("agentrubric")
    except PackageNotFoundError:
        return "dev"


def main() -> int:
    print(BANNER)
    print(f"v{_version()} — trajectory-level evaluation for LLM agents")
    print("Created by Madhumitha Kolkar")
    print()
    print("  from agentrubric import Trajectory, Step, evaluate")
    print()
    print("Docs & examples: https://github.com/MadhumithaKolkar/Agent-Rubric")
    return 0


if __name__ == "__main__":
    sys.exit(main())
