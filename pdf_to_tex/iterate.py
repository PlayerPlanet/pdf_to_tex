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
from typing import List, Optional

from compose import ensure_model, call_model_for_correction, extract_model_text, strip_fences
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def run_pdflatex(tex_path: Path, workdir: Path, cmd: List[str]) -> (int, str):
    proc = subprocess.run(cmd + [str(tex_path.name)], cwd=str(workdir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


def parse_latex_errors(log_text: str):
    """Parse pdflatex output and return a list of errors.

    This is a tolerant parser which handles several common formats produced by
    TeX engines, including:

    - lines starting with "! <message>" followed by a "l.<num>" reference
    - lines like "./file.tex:6: LaTeX Error: <message>"
    - lines containing "LaTeX Error:" without a file prefix

    Returns list of dicts: {'line': int|None, 'message': str}.
    """
    errors = []
    lines = log_text.splitlines()

    # First, scan for explicit file:line: LaTeX Error: messages
    fileline_re = re.compile(r"^(?:\./)?([^:\s]+):(\d+):\s+LaTeX Error:\s*(.*)$")
    for ln in lines:
        m = fileline_re.match(ln.strip())
        if m:
            lineno = int(m.group(2))
            msg = m.group(3).strip()
            errors.append({"line": lineno, "message": msg})

    # Also accept generic file:line: message lines (e.g. "./output.tex:436: Undefined control sequence.")
    fileline_general_re = re.compile(r"^(?:\./)?([^:\s]+):(\d+):\s*(.*)$")
    for ln in lines:
        m = fileline_general_re.match(ln.strip())
        if m:
            lineno = int(m.group(2))
            msg = m.group(3).strip()
            # Avoid duplicating entries already captured
            if not any(e.get("line") == lineno and e.get("message") == msg for e in errors):
                # Heuristic: skip trivial lines that are just '(' or ')' etc.
                if msg and msg not in ("(", ")"):
                    errors.append({"line": lineno, "message": msg})

    # Second, fallback to old-style "! message" markers
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("!"):
            msg = ln[1:].strip()
            lineno = None
            j = i + 1
            # look forward a few lines for l.<num> or filename:line patterns
            while j < len(lines) and j < i + 12:
                m = re.search(r"l\.(\d+)", lines[j])
                if m:
                    lineno = int(m.group(1))
                    break
                m2 = fileline_re.search(lines[j])
                if m2:
                    lineno = int(m2.group(2))
                    break
                j += 1
            errors.append({"line": lineno, "message": msg})
            i = j
        else:
            i += 1

    # If nothing found, also look for standalone 'LaTeX Error:' lines
    if not errors:
        for ln in lines:
            if "LaTeX Error:" in ln:
                # try to extract text after the marker
                parts = ln.split("LaTeX Error:", 1)
                msg = parts[1].strip() if len(parts) > 1 else ln.strip()
                errors.append({"line": None, "message": msg})

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


# Mapping of common undefined commands to packages that provide them
COMMAND_PACKAGE_MAP = {
    r"\\blacksquare": "amssymb",
    r"\\mathbb": "amsfonts",
    r"\\mathscr": "mathrsfs",
    r"\\qedhere": "amsthm",
    r"\\square": "amssymb",
}


def find_preamble_end(lines: List[str]) -> int:
    """Return the 0-based index (inclusive) of the last preamble line (line index before \begin{document}).

    If \begin{document} not found, return 0 (insert after first line).
    """
    for i, ln in enumerate(lines):
        if ln.strip().startswith("\\begin{document}"):
            return max(0, i - 1)
    return 0


def package_in_preamble(lines: List[str], pkg: str) -> bool:
    pat = re.compile(r"\\usepackage(?:\[[^]]+\])?\{\s*" + re.escape(pkg) + r"\s*\}")
    pre_end = find_preamble_end(lines)
    for ln in lines[: pre_end + 1]:
        if pat.search(ln):
            return True
    return False


def insert_usepackage(lines: List[str], pkg: str) -> List[str]:
    # Insert after last \usepackage or after \documentclass
    pre_end = find_preamble_end(lines)
    insert_idx = None
    last_use_idx = None
    docclass_idx = None
    for i in range(0, pre_end + 1):
        ln = lines[i]
        if ln.strip().startswith("\\documentclass"):
            docclass_idx = i
        if ln.strip().startswith("\\usepackage"):
            last_use_idx = i

    if last_use_idx is not None:
        insert_idx = last_use_idx + 1
    elif docclass_idx is not None:
        insert_idx = docclass_idx + 1
    else:
        insert_idx = 0

    new_lines = lines[:insert_idx] + [f"\\usepackage{{{pkg}}}"] + lines[insert_idx:]
    return new_lines


def suggest_package_for_symbol(model_name: str, symbol: str, fragment: str, is_env: bool = False) -> Optional[str]:
    """Ask the model to suggest a LaTeX package that provides `symbol`.

    Returns a package name (string) or None if no suggestion found.
    """
    if not model_name:
        return None

    # small prompt asking for a package
    if is_env:
        q = (
            f"You are a LaTeX expert. Which LaTeX package provides the environment '{symbol}'? "
            "If no package is required (the environment is core LaTeX), reply with NONE. "
            "Return only the package name (e.g. amsmath) or NONE."
        )
    else:
        # symbol is like \cmd
        q = (
            f"You are a LaTeX expert. Which LaTeX package provides the command '{symbol}'? "
            "If it's a core command that requires no extra package, reply with NONE. "
            "Return only the package name (e.g. amssymb) or NONE."
        )

    try:
        resp = call_model_for_correction(model_name, q, fragment)
        txt = extract_model_text(resp)
        if txt is None:
            return None
        txt = strip_fences(txt).strip()
    except Exception:
        return None

    # Try parse common patterns like \usepackage{pkg} or just 'pkg'
    m = re.search(r"\\usepackage\{\s*([a-zA-Z0-9_\-]+)\s*\}", txt)
    if m:
        return m.group(1)
    # also accept plain words like 'amssymb' or sentences containing it
    # prefer known popular packages
    known = ["amssymb", "amsmath", "amsfonts", "mathrsfs", "amsthm", "latexsym", "graphicx", "xcolor", "mathtools", "bm"]
    for k in known:
        if re.search(rf"\b{k}\b", txt, re.IGNORECASE):
            return k

    # fallback: return first token that looks like a package name
    m2 = re.search(r"\b([a-zA-Z][a-zA-Z0-9_\-]{1,30})\b", txt)
    if m2:
        return m2.group(1)
    return None


def prompt_for_fix(fragment: str, lineno: int, extra_instructions: Optional[str] = None) -> str:
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

    iter_range = range(1, args.max_iter + 1)
    if tqdm is not None:
        iter_iter = tqdm(iter_range, desc="Iterations")
    else:
        iter_iter = iter_range

    attempt_counts = {}

    for it in iter_iter:
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

        # increment attempt counter for this lineno (use -1 for None)
        key = lineno if lineno is not None else -1
        attempt_counts[key] = attempt_counts.get(key, 0) + 1

        start, end, fragment = extract_window(lines, lineno, ctx=args.context_lines)
        print(f"Using lines {start}-{end} as fragment (len={len(fragment.splitlines())} lines)")

        # Heuristic: if Undefined control sequence, try to detect the control sequence
        # from the fragment and insert the providing package into the preamble.
        did_auto_fix = False
        if "Undefined control sequence" in message or "Undefined control sequence" in out:
            mcmd = re.search(r"(\\[A-Za-z@]+)", fragment)
            if mcmd:
                cmdname = mcmd.group(1)
                print(f"Detected control sequence in fragment: {cmdname}")
                # find mapping
                pkg = None
                for patt, p in COMMAND_PACKAGE_MAP.items():
                    if re.fullmatch(patt, cmdname):
                        pkg = p
                        break
                if pkg:
                    if not package_in_preamble(lines, pkg):
                        print(f"Inserting \\usepackage{{{pkg}}} into preamble (auto-fix)")
                        lines = insert_usepackage(lines, pkg)
                        tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                        did_auto_fix = True
                        # continue to next iteration without calling model
                        print(f"Applied package-insert for {pkg}; re-running pdflatex next iteration.")

        if did_auto_fix:
            # go to next iteration
            continue

        if not args.model:
            print("No model specified; printing fragment and aborting.")
            print(fragment)
            sys.exit(0)

        # If we've tried several times on the same line, escalate instructions to allow
        # the model to propose preamble changes. The model can return a preamble patch
        # between markers PREAMBLE_PATCH_START / PREAMBLE_PATCH_END at the start of the response.
        extra = None
        if attempt_counts.get(key, 0) >= 3:
            extra = (
                "If fixing this fragment requires changes to the document preamble, "
                "return the preamble patch first between lines:\nPREAMBLE_PATCH_START\n...\nPREAMBLE_PATCH_END\n"
                "followed by the replacement fragment. Otherwise return only the replacement fragment."
            )

        prompt = prompt_for_fix(fragment, lineno, extra_instructions=extra)
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

        # If the model returned a preamble patch, apply it first.
        preamble_patch = None
        if "PREAMBLE_PATCH_START" in replacement and "PREAMBLE_PATCH_END" in replacement:
            parts = replacement.split("PREAMBLE_PATCH_START", 1)[1]
            patch, rest = parts.split("PREAMBLE_PATCH_END", 1)
            preamble_patch = patch.strip()
            replacement = rest.strip()

        if preamble_patch:
            # Insert preamble lines (allow multiple lines)
            patch_lines = [l for l in preamble_patch.splitlines() if l.strip()]
            if patch_lines:
                # insert after documentclass/usepackage region
                lines = insert_usepackage(lines, "%__MODEL_INSERT__%")  # placeholder to get position
                # Replace placeholder with actual patch lines
                for i, ln in enumerate(lines):
                    if ln.strip() == "\\usepackage{%__MODEL_INSERT__%}":
                        lines = lines[:i] + patch_lines + lines[i+1:]
                        break
                tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                print("Applied preamble patch returned by model.")

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
