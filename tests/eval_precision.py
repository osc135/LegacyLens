"""
Retrieval precision evaluation suite.

Runs a diverse set of queries against the retrieval pipeline and measures
whether expected subroutines appear in the top-K results. This gives a
deterministic, reproducible precision score — no LLM judge involved.

Matching rules:
- Handles _PART suffixes: DGESVD_PART1 matches expected "DGESVD"
- For broad/vague queries, accepts any type variant: SGETRF matches "DGETRF"
- Named subroutine queries require exact base-name match

Run:  python3 tests/eval_precision.py
"""
import sys
import os
import json
import re
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.retrieval import search


def strip_part_suffix(name: str) -> str:
    """DGESVD_PART1 -> DGESVD, DGESVD_PART12 -> DGESVD"""
    return re.sub(r'_PART\d+$', '', name)


def base_name_without_type(name: str) -> str:
    """DGESV -> GESV, SGETRF -> GETRF (strip leading type prefix S/D/C/Z)"""
    name = strip_part_suffix(name)
    if len(name) >= 4 and name[0] in "SDCZ":
        return name[1:]
    return name


# Each eval case: (query, match_mode, expected_names)
#   match_mode:
#     "exact"  — expected base name must appear (after stripping _PART suffixes)
#     "family" — any type variant of expected base name counts (SGETRF matches DGETRF)
#     "any"    — at least ONE expected name (or family member) found
EVAL_CASES = [
    # --- Named subroutine queries: must find the exact routine ---
    {
        "query": "Explain the DGESV linear solver",
        "mode": "explain",
        "expected": ["DGESV"],
        "match": "exact",
    },
    {
        "query": "What does ZHEEV do?",
        "mode": "ask",
        "expected": ["ZHEEV"],
        "match": "exact",
    },
    {
        "query": "How does DGETRF work?",
        "mode": "ask",
        "expected": ["DGETRF"],
        "match": "exact",
    },
    {
        "query": "Explain the DPOTRF subroutine",
        "mode": "explain",
        "expected": ["DPOTRF"],
        "match": "exact",
    },
    {
        "query": "What is DGESVD used for?",
        "mode": "ask",
        "expected": ["DGESVD"],
        "match": "exact",
    },

    # --- Broad conceptual queries: any type variant is fine ---
    {
        "query": "How does LAPACK solve Ax = b?",
        "mode": "ask",
        "expected": ["DGESV", "DGETC2", "DGESC2"],
        "match": "any",
    },
    {
        "query": "How does QR factorization work?",
        "mode": "ask",
        "expected": ["DGEQR2", "DGEQRF", "DLATSQR", "DGEDMDQ"],
        "match": "any",
    },
    {
        "query": "How does LAPACK compute eigenvalues?",
        "mode": "ask",
        "expected": ["DSYEV", "DGEEV", "DSYEVD"],
        "match": "any",
    },
    {
        "query": "What routines does LAPACK have for SVD?",
        "mode": "ask",
        "expected": ["DGESVD", "DGESVDQ", "DGESDD"],
        "match": "any",
    },
    {
        "query": "How does LU factorization work in LAPACK?",
        "mode": "ask",
        "expected": ["DGETRF", "DGETF2", "DSYTRF", "DGTSVX"],
        "match": "any",
    },

    # --- Vague / natural language queries ---
    {
        "query": "Where is the main entry point of this program?",
        "mode": "ask",
        "expected": ["DGESV", "SGESV", "CGESV", "ZGESV", "DGETRF", "DPOTRF"],
        "match": "any",
    },
    {
        "query": "How do I solve a linear system?",
        "mode": "ask",
        "expected": ["DGESV"],
        "match": "family",
    },
    {
        "query": "Find matrix factorization routines",
        "mode": "ask",
        "expected": ["DGETRF", "DPOTRF", "DGETF2", "DPOTF2", "DLAQPS", "DLAQP3RK"],
        "match": "any",
    },

    # --- Spec testing scenarios ---
    {
        "query": "What are the dependencies of DGELS?",
        "mode": "ask",
        "expected": ["DGELS"],
        "match": "exact",
    },
    {
        "query": "Show me error handling patterns in this codebase",
        "mode": "ask",
        "expected": ["XERBLA", "DGERFSX", "SGERFSX", "CGERFSX"],
        "match": "any",
    },
    {
        "query": "Explain what the DGEQRF subroutine does",
        "mode": "explain",
        "expected": ["DGEQRF"],
        "match": "exact",
    },

    # --- Documentation generation ---
    {
        "query": "Generate docs for DPOTRF",
        "mode": "docs",
        "expected": ["DPOTRF"],
        "match": "exact",
    },
    {
        "query": "Generate docs for DGETRF",
        "mode": "docs",
        "expected": ["DGETRF"],
        "match": "exact",
    },

    # --- Multi-routine queries ---
    {
        "query": "What eigenvalue algorithms does LAPACK provide?",
        "mode": "ask",
        "expected": ["DSYEV", "DGEEV", "DSYEVD", "DSTEV"],
        "match": "any",
    },
    {
        "query": "What Cholesky factorization routines are available?",
        "mode": "ask",
        "expected": ["DPOTRF", "DPOTF2"],
        "match": "any",
    },
    {
        "query": "How does LAPACK handle tridiagonal matrices?",
        "mode": "ask",
        "expected": ["DGTSV", "DSTEQR", "DSTEV"],
        "match": "any",
    },
]


def check_match(expected_names: list[str], found_names: list[str], match_mode: str) -> bool:
    """Check if found names satisfy the expected names given the match mode."""
    # Normalize found names by stripping _PART suffixes
    found_base = [strip_part_suffix(n) for n in found_names]
    found_families = [base_name_without_type(n) for n in found_names]

    if match_mode == "exact":
        # All expected base names must appear in found (after stripping _PART)
        return all(exp in found_base for exp in expected_names)

    elif match_mode == "family":
        # All expected names must have at least one family member in found
        for exp in expected_names:
            exp_family = base_name_without_type(exp)
            if exp not in found_base and exp_family not in found_families:
                return False
        return True

    elif match_mode == "any":
        # At least one expected name (or its family member) must appear
        for exp in expected_names:
            exp_family = base_name_without_type(exp)
            if exp in found_base or exp_family in found_families:
                return True
        return False

    return False


def run_eval():
    """Run all eval cases and report results."""
    results = []
    pass_count = 0
    total = len(EVAL_CASES)

    print(f"Running {total} eval cases...\n")
    print(f"{'#':<3} {'Pass':<6} {'Match':<7} {'Query':<50} {'Expected':<25} {'Got (top 5)'}")
    print("-" * 140)

    for i, case in enumerate(EVAL_CASES, 1):
        query = case["query"]
        expected_names = case["expected"]
        match_mode = case["match"]

        search_results = search(query, top_k=5)
        found_names = [r["name"] for r in search_results]

        passed = check_match(expected_names, found_names, match_mode)

        if passed:
            pass_count += 1

        status = "PASS" if passed else "FAIL"
        expected_str = ",".join(expected_names)
        found_str = ", ".join(found_names)

        print(f"{i:<3} {status:<6} {match_mode:<7} {query:<50} {expected_str:<25} {found_str}")

        results.append({
            "query": query,
            "mode": case["mode"],
            "match_mode": match_mode,
            "expected": expected_names,
            "found": found_names,
            "passed": passed,
        })

    precision = pass_count / total
    print("-" * 140)
    print(f"\nOverall: {pass_count}/{total} passed ({precision:.0%})")

    if precision < 0.7:
        print("\nFailed cases:")
        for r in results:
            if not r["passed"]:
                print(f"  - {r['query']}")
                print(f"    Expected: {r['expected']}")
                print(f"    Got:      {r['found']}")

    # Save results
    log_path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_results.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pass_count": pass_count,
            "total": total,
            "precision": precision,
            "cases": results,
        }, f, indent=2)
    print(f"\nResults saved to {log_path}")

    return precision


if __name__ == "__main__":
    precision = run_eval()
    sys.exit(0 if precision >= 0.7 else 1)
