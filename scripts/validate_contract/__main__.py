import sys
from .check_files import check_files
from .check_model import check_model
from .check_evaluate import check_evaluate
from .constants import CheckResult


def format_table(results: list[CheckResult]) -> str:
    col_name = "Check"
    col_status = "Status"
    col_detail = "Details"

    max_name = max(len(col_name), max((len(r.name) for r in results), default=0))
    max_status = max(len(col_status), max((len("PASS" if r.passed else "FAIL") for r in results), default=0))

    header = f"{col_name:<{max_name}}  {col_status:<{max_status}}  {col_detail}"
    separator = "-" * max(len(header), 80)

    lines = [separator, header, separator]
    for r in results:
        status_str = "PASS" if r.passed else "FAIL"
        lines.append(f"{r.name:<{max_name}}  {status_str:<{max_status}}  {r.detail}")
    lines.append(separator)

    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count
    lines.append(f"Summary: {passed_count} passed, {failed_count} failed")

    return "\n".join(lines)


def main() -> None:
    results: list[CheckResult] = []
    results.extend(check_files())
    results.extend(check_model())
    results.extend(check_evaluate())

    print(format_table(results))

    any_failed = any(not r.passed for r in results)
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
