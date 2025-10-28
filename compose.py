#!/usr/bin/env python3
"""
compose.py

Extract every response="..." occurrence from a text file and write the concatenated
responses into a single output file.

Usage:
    python compose.py --input page127_model.txt --output composed.tex

Options:
    --strip-fences    Remove triple-backtick fences (```...```) from each response
    --sep TEXT        Separator to insert between parts (default: LaTeX comment line)
    --show-count      Print how many responses were found
"""
import argparse
import re
from pathlib import Path
import sys
import ollama


def ensure_model(model_name: str) -> bool:
    try:
        models = ollama.list()
    except Exception:
        models = {}
    if not models.get(model_name):
        try:
            print(f"Pulling model {model_name}...")
            ollama.pull(model_name)
        except Exception as e:
            print(f"Failed to pull model {model_name}: {e}", file=sys.stderr)
            return False
    return True


def call_model_for_correction(model_name: str, prompt: str, text: str):
    try:
        # send prompt and text together; model clients may accept multi-part prompts
        full_prompt = prompt + "\n\n" + text
        resp = ollama.generate(model=model_name, prompt=full_prompt)
    except Exception as e:
        raise RuntimeError(f"ollama.generate failed: {e}")
    return resp


def extract_model_text(resp):
    """Extract a textual response from various possible ollama.generate return types.

    - If resp is a dict-like, try common keys.
    - If resp is a string, try to extract response="..." or response='...'.
    - If resp has attributes, try common attribute names.
    - Otherwise, fallback to str(resp).
    """
    # dict-like
    try:
        if isinstance(resp, dict):
            for key in ("response", "content", "text", "output"):
                if key in resp and resp[key]:
                    return resp[key]
    except Exception:
        pass

    # object with attributes
    try:
        for attr in ("response", "content", "text", "output"):
            if hasattr(resp, attr):
                val = getattr(resp, attr)
                if val:
                    return val
    except Exception:
        pass

    # string: try to extract response="..." or response='...'
    try:
        if isinstance(resp, str):
            m = re.search(r'response\s*=\s*"([\s\S]*?)"', resp)
            if not m:
                m = re.search(r"response\s*=\s*'([\s\S]*?)'", resp)
            if m:
                return m.group(1)
            # fallback: return whole string
            return resp
    except Exception:
        pass

    # list/iterable
    try:
        if isinstance(resp, (list, tuple)) and len(resp) > 0:
            # join elements as strings
            return "\n".join(str(x) for x in resp)
    except Exception:
        pass

    # final fallback
    return str(resp)


def extract_responses(text: str):
    """Extract response=... occurrences where the value is quoted.

    Handles both single- and double-quoted values and basic backslash-escaping.
    Returns list of raw string contents (unescaped where possible).
    """
    responses = []

    # Find occurrences of 'response' followed by '='
    for m in re.finditer(r"\bresponse\s*=\s*", text):
        pos = m.end()
        # skip whitespace
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        quote = text[pos]
        if quote not in ('"', "'"):
            # not a quoted response; skip
            continue
        pos += 1
        buf = []
        escaped = False
        while pos < len(text):
            ch = text[pos]
            if escaped:
                # keep the escaped char literally
                buf.append(ch)
                escaped = False
            else:
                if ch == '\\':
                    escaped = True
                elif ch == quote:
                    # end of quoted string
                    pos += 1
                    break
                else:
                    buf.append(ch)
            pos += 1

        responses.append(''.join(buf))

    return responses


def strip_fences(s: str):
    # remove leading/trailing ```lang or ``` and trailing ```
    s2 = re.sub(r"^\s*```[a-zA-Z0-9_-]*\n", "", s)
    s2 = re.sub(r"\n```\s*$", "", s2)
    return s2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input text file containing model outputs")
    parser.add_argument("--output", default="composed_output.txt", help="Output file to write concatenated responses")
    parser.add_argument("--strip-fences", action="store_true", help="Strip ``` code fences from responses")
    parser.add_argument("--sep", default="% --- PART ---\n", help="Separator inserted between parts")
    parser.add_argument("--show-count", action="store_true", help="Print number of extracted responses")
    parser.add_argument("--model", default="qwen2.5vl:3b", help="Ollama model to use for final correction (or set to empty to skip)")
    parser.add_argument("--final-output", default="composed_corrected.tex", help="File to write the final corrected LaTeX")
    parser.add_argument("--final-prompt", default=None, help="Optional custom final correction prompt (overrides built-in) ")

    args = parser.parse_args()

    p = Path(args.input)
    if not p.exists():
        print(f"Input file not found: {p}", file=sys.stderr)
        sys.exit(2)

    text = p.read_text(encoding="utf-8")
    responses = extract_responses(text)

    if args.show_count:
        print(f"Found {len(responses)} response(...) occurrences")

    out_lines = []
    for idx, r in enumerate(responses, start=1):
        part = r
        if args.strip_fences:
            part = strip_fences(part)
        # Ensure parts end with a newline
        if not part.endswith('\n'):
            part = part + '\n'
        out_lines.append(args.sep.replace("{i}", str(idx)))
        out_lines.append(part)

    out_path = Path(args.output)
    out_path.write_text(''.join(out_lines), encoding="utf-8")
    print(f"Wrote {len(responses)} parts to {out_path}")

    # If a model is provided, call it to correct the concatenated LaTeX
    if args.model:
        if len(responses) == 0:
            print("No responses to send to model; skipping correction.")
            return

        concat_text = ''.join(out_lines)

        final_prompt = args.final_prompt
        if final_prompt is None:
            final_prompt = (
                "You are a LaTeX expert. The following text contains LaTeX fragments extracted from OCR/LLM responses. "
                "Please correct any LaTeX syntax errors, fix obvious typos in math mode, ensure environments are balanced, and output only the corrected LaTeX code (no explanations)."
            )

        if not ensure_model(args.model):
            print("Model unavailable and could not be pulled.", file=sys.stderr)
            sys.exit(11)

        try:
            corrected = call_model_for_correction(args.model, final_prompt, concat_text)
        except Exception as e:
            print(f"Model correction failed: {e}", file=sys.stderr)
            sys.exit(12)

        # Extract only the textual response from the model return value
        final_text = extract_model_text(corrected)
        if args.strip_fences:
            final_text = strip_fences(final_text)

        final_out = Path(args.final_output)
        try:
            final_out.write_text(final_text, encoding="utf-8")
        except Exception as e:
            print(f"Failed to write final output file: {e}", file=sys.stderr)
            sys.exit(13)

        print(f"Wrote corrected LaTeX to {final_out}")


if __name__ == "__main__":
    main()
