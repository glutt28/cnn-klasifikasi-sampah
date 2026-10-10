import sys
from .check_files import check_files
from .check_model import check_model
from .check_evaluate import check_evaluate


def main() -> None:
    steps = [
        ("File Structure", check_files),
        ("Model Contract", check_model),
        ("Evaluate Integration", check_evaluate),
    ]

    for name, action in steps:
        try:
            action()
            print(f"[PASS] {name}")
        except Exception as err:
            print(f"[FAIL] {name}: {err}", file=sys.stderr)
            sys.exit(1)

    print("All submission contracts satisfied.")


if __name__ == "__main__":
    main()
