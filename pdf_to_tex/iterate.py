#!/usr/bin/env python3
"""
iterate.py

Iteratively compile a .tex file, parse pdflatex errors, and ask an LLM (via Ollama) to regenerate
only the fragment that caused the error. Repeat until compilation succeeds or max iterations reached.

Usage:
    python iterate.py --tex output.tex --model qwen2.5vl:3b --max-iter 6

Notes:
 - Requires a working pdflatex on PATH (or change --pdflatex-cmd to a different LaTeX engine).
 - Uses the project's `compose` module to call the Ollama model.
"""
import argparse
import os
import re
import subprocess
from pathlib import Path
import sys
from typing import List

from compose import ensure_model, call_model_for_correction, extract_model_text, strip_fences


def run_pdflatex(tex_path: Path, workdir: Path, cmd: List[str]) -> (int, str):
    proc = subprocess.run(cmd + [str(tex_path.name)], cwd=str(workdir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


def parse_latex_errors(log_text: str):
    """Heuristic parser for pdflatex output: find '! <message>' followed by 'l.<num>' markers.

    Returns list of dicts {'line': int|None, 'message': str}.
    """
    errors = []
    lines = log_text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("!"):
            msg = ln[1:].strip()
            # scan forward for l.<num>
            lineno = None
            j = i + 1
            while j < len(lines) and j < i + 12:
                m = re.search(r"l\.(\d+)", lines[j])
                if m:
                    lineno = int(m.group(1))
                    break
                j += 1
            errors.append({"line": lineno, "message": msg})
            i = j
        else:
            i += 1
    return errors


def extract_window(lines: List[str], lineno: int, ctx: int = 8):
    """Return (start, end, fragment) for given 1-based lineno. If lineno is None, return last chunk."""
    if lineno is None:
        start = max(1, len(lines) - 40 + 1)
        end = len(lines)
    else:
        start = max(1, lineno - ctx)
        end = min(len(lines), lineno + ctx)
    fragment = "\n".join(lines[start - 1:end])
    return start, end, fragment


def prompt_for_fix(fragment: str, lineno: int, extra_instructions: str = None) -> str:
    instr = (
        "You are a LaTeX expert. The following fragment of a larger .tex file causes a compilation error.\n"
        f"Approximate offending line: {lineno}\n\n"
        "Task: Return a corrected LaTeX fragment that should replace the provided fragment. "
        "Return only the replacement LaTeX (no extra commentary). Preserve surrounding context where possible.\n\n"
    )
    if extra_instructions:
        instr += extra_instructions + "\n\n"
    instr += "FRAGMENT START\n```latex\n" + fragment + "\n```\n\nREPLACEMENT:"
    return instr


def apply_replacement(lines: List[str], start: int, end: int, replacement: str) -> List[str]:
    # start/end are 1-based inclusive
    rep_lines = replacement.splitlines()
    new_lines = []
    new_lines.extend(lines[: start - 1])
    new_lines.extend(rep_lines)
    new_lines.extend(lines[end:])
    return new_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", required=True, help="Path to .tex file to iterate on")
    parser.add_argument("--model", default="qwen2.5vl:3b", help="Ollama model for fixes (empty to skip)")
    parser.add_argument("--max-iter", type=int, default=6, help="Max iterations")
    parser.add_argument("--context-lines", type=int, default=8, help="Lines of context around error")
    parser.add_argument("--pdflatex-cmd", default="pdflatex", help="pdflatex command (can be full path)")
    parser.add_argument("--force-cpu", action="store_true", help="Set CUDA_VISIBLE_DEVICES='' before running model calls")
    args = parser.parse_args()

    if args.force_cpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    tex_path = Path(args.tex).resolve()
    if not tex_path.exists():
        print(f"File not found: {tex_path}")
        sys.exit(2)

    workdir = tex_path.parent

    if args.model:
        if not ensure_model(args.model):
            print(f"Model {args.model} unavailable and could not be pulled.")
            sys.exit(3)

    lines = tex_path.read_text(encoding="utf-8").splitlines()

    for it in range(1, args.max_iter + 1):
        print(f"[iter {it}] running {args.pdflatex_cmd}...")
        cmd = [args.pdflatex_cmd, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error"]
        rc, out = run_pdflatex(tex_path, workdir, cmd)
        if rc == 0:
            print("Compilation succeeded.")
            return
        print("Compilation failed; parsing output for errors...")
        errors = parse_latex_errors(out)
        if not errors:
            print("No parsable errors found. pdflatex output:\n", out)
            sys.exit(4)

        err = errors[0]
        lineno = err.get("line")
        message = err.get("message", "")
        print(f"Found error (approx line {lineno}): {message}")

        start, end, fragment = extract_window(lines, lineno, ctx=args.context_lines)
        print(f"Using lines {start}-{end} as fragment (len={len(fragment.splitlines())} lines)")

        if not args.model:
            print("No model specified; printing fragment and aborting.")
            print(fragment)
            sys.exit(0)

        prompt = prompt_for_fix(fragment, lineno)
        try:
            resp = call_model_for_correction(args.model, prompt, fragment)
            replacement = extract_model_text(resp)
            replacement = strip_fences(replacement).strip()
        except Exception as e:
            print(f"Model call failed: {e}")
            sys.exit(5)

        if not replacement:
            print("Model returned empty replacement; aborting.")
            sys.exit(6)

        # Apply replacement and write file
        new_lines = apply_replacement(lines, start, end, replacement)
        tex_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        # update in-memory lines for next iteration
        lines = new_lines
        print(f"Applied replacement for lines {start}-{end}; next iteration.")

    print(f"Reached max iterations ({args.max_iter}) without successful compilation.")
    sys.exit(7)


if __name__ == "__main__":
    main()
