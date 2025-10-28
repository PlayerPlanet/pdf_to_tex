#!/usr/bin/env python3
"""
diagnosis.py

Extract text from a given page of a PDF and write it to a text file.

Usage example:
    python diagnosis.py --path path/to/file.pdf --page 1

The script writes the extracted text to `diagnosis_output.txt` by default.
"""
import argparse
import sys
from pathlib import Path
import fitz  # pymupdf


def extract_page_text(pdf_path: str, page_number: int) -> str:
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(p))
    try:
        page_count = doc.page_count
    except Exception:
        # fallback for older pymupdf versions
        page_count = len(doc)

    if page_number < 1 or page_number > page_count:
        raise IndexError(f"page {page_number} out of range (1-{page_count})")

    page = doc[page_number - 1]
    # Use 'text' to get plain text. Other options: 'blocks', 'dict', 'html'
    text = page.get_text("text")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text of a page from PDF to diagnosis_output.txt"
    )
    parser.add_argument("--path", required=True, help="Path to PDF file")
    parser.add_argument("--page", type=int, required=True, help="Page number (1-based)")
    parser.add_argument("--output", default="diagnosis_output.txt", help="Output text file")

    args = parser.parse_args()

    try:
        text = extract_page_text(args.path, args.page)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except IndexError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(4)

    out_path = Path(args.output)
    try:
        out_path.write_text(text, encoding="utf-8")
    except Exception as e:
        print(f"Failed to write output file: {e}", file=sys.stderr)
        sys.exit(5)

    print(f"Wrote page {args.page} text to {out_path}")


if __name__ == "__main__":
    main()
