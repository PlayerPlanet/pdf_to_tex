#!/usr/bin/env python3
"""
pipeline.py

PDF -> LaTeX pipeline that combines `image_diagnosis.py` and `compose.py` logic.

Steps:
 - Find an image-splitting configuration that the Ollama runner can handle by
   rendering the first page at a given scale and trying parts=1..max_parts.
 - For each page: render, split into parts, call the model on each part, collect
   extracted LaTeX fragments and the plain text of the page (from `diagnosis.py`).
 - For each page, call the model one final time to "compose" a single corrected
   LaTeX page using the extracted LaTeX fragments and the plain text as backup.
 - Append each corrected page to an output LaTeX file that already contains the
   preamble (documentclass, packages, begin{document}).

This script uses the existing modules in the repo: `image_diagnosis.py`,
`compose.py` and `diagnosis.py`.
"""
import argparse
import os
from pathlib import Path
import sys
from typing import List, Tuple

from image_diagnosis import render_page_to_png, split_image_vertically, call_model_with_image
from pdf_to_tex.compose import extract_model_text, ensure_model, call_model_for_correction, strip_fences
from diagnosis import extract_page_text

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


def find_working_parts(pdf_path: str, page: int, scale: float, max_parts: int, lr_margin: float, images_dir: str, model: str) -> int:
    """Render `page` and try split counts 1..max_parts. Return first parts that work.

    A parts value "works" if every generated slice returns a model response without raising an exception.
    """
    images_dir_p = Path(images_dir)
    images_dir_p.mkdir(parents=True, exist_ok=True)
    test_image = images_dir_p / f"_pipeline_test_page_{page}.png"
    render_page_to_png(pdf_path, page, str(test_image), scale=scale)

    for parts in range(1, max_parts + 1):
        try:
            slices = split_image_vertically(str(test_image), parts=parts, overlap=0.1, out_dir=str(images_dir_p), lr_margin=lr_margin)
        except Exception as e:
            # splitting failed for this parts configuration: try next
            print(f"split_image_vertically failed for parts={parts}: {e}")
            continue

        failed = False
        for slice_path, _coords in slices:
            try:
                _resp = call_model_with_image(model, "", slice_path)
            except Exception as e:
                print(f"Model failed on parts={parts}, slice={slice_path}: {e}")
                failed = True
                break

        if not failed:
            print(f"Working configuration found: parts={parts}")
            return parts

    raise RuntimeError(f"No working parts configuration found up to max_parts={max_parts}")


def compose_page_with_model(model: str, page_fragments: List[str], page_text: str, final_prompt: str = None, strip_fences_flag: bool = True) -> str:
    """Concatenate fragments and page_text and ask the model to produce one corrected LaTeX page.

    Returns the textual response (string) from the model.
    """
    if final_prompt is None:
        final_prompt = (
            "You are a LaTeX expert. Compose a single-page LaTeX fragment (only the page body) given the following extracted LaTeX fragments and the OCR/extracted plain text. "
            "Assume the preamble contains `\\documentclass{article}`, `\\usepackage{amsmath}`, `\\usepackage{amsfonts}`, and `\\begin{document}`. "
            "Output only the corrected LaTeX for that page (no preamble, no explanations). Use the extracted text as a backup when fragments are incomplete."
        )

    # Build the input text for the final correction step
    fragments_text = "\n\n".join(page_fragments)
    full_input = f"--- FRAGMENTS ---\n{fragments_text}\n\n--- TEXT ---\n{page_text}\n"

    resp = call_model_for_correction(model, final_prompt, full_input)
    text = extract_model_text(resp)
    if strip_fences_flag:
        text = strip_fences(text)
    return text


def run_pipeline(pdf_path: str, output_tex: str, model: str, scale: float, max_parts: int, lr_margin: float, images_dir: str, start_page: int, end_page: int) -> Tuple[List[int], List[Tuple[int, str]]]:
    """Run the pipeline over pages start_page..end_page inclusive.

    Returns (failed_pages, failed_slices) where failed_pages is list of page numbers that failed final compose,
    and failed_slices is list of (page, slice_path) tuples that failed model extraction.
    """
    failed_pages = []
    failed_slices = []

    Path(images_dir).mkdir(parents=True, exist_ok=True)

    # Find parts configuration using the first page in range
    sample_page = start_page
    parts = find_working_parts(pdf_path, sample_page, scale, max_parts, lr_margin, images_dir, model)

    # Prepare output tex with preamble
    out_path = Path(output_tex)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amsfonts}\n\\begin{document}\n\n")

    for page in tqdm(range(start_page, end_page + 1), desc="Pages"):
        print(f"Processing page {page} ...")
        image_path = Path(images_dir) / f"page_{page}.png"
        try:
            render_page_to_png(pdf_path, page, str(image_path), scale=scale)
        except Exception as e:
            print(f"Failed to render page {page}: {e}")
            failed_pages.append(page)
            continue

        try:
            slices = split_image_vertically(str(image_path), parts=parts, overlap=0.1, out_dir=images_dir, lr_margin=lr_margin)
        except Exception as e:
            print(f"Failed to split page {page}: {e}")
            failed_pages.append(page)
            continue

        page_fragments = []
        for slice_path, _coords in slices:
            try:
                resp = call_model_with_image(model, "", slice_path)
                frag = extract_model_text(resp)
                frag = strip_fences(frag)
                page_fragments.append(frag)
            except Exception as e:
                print(f"Model failed on page {page} slice {slice_path}: {e}")
                failed_slices.append((page, slice_path))

        # Also extract the plain text of the page for backup
        try:
            page_text = extract_page_text(pdf_path, page)
        except Exception as e:
            print(f"Failed to extract plain text for page {page}: {e}")
            page_text = ""

        # Final composition per page
        try:
            corrected_page = compose_page_with_model(model, page_fragments, page_text)
        except Exception as e:
            print(f"Final composition failed for page {page}: {e}")
            failed_pages.append(page)
            continue

        # Append corrected page to output
        with out_path.open("a", encoding="utf-8") as f:
            f.write(corrected_page)
            f.write("\n\n")

    # finish document
    with out_path.open("a", encoding="utf-8") as f:
        f.write("\\end{document}\n")

    return failed_pages, failed_slices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--output", default="output.tex", help="Output LaTeX file")
    parser.add_argument("--model", default="qwen2.5vl:7b", help="Ollama model to use")
    parser.add_argument("--scale", type=float, default=2.0, help="Initial render scale to try")
    parser.add_argument("--max-parts", type=int, default=6, help="Max number of vertical parts to try")
    parser.add_argument("--lr-margin", type=float, default=0.15, help="Left/right margin fraction to trim")
    parser.add_argument("--images-dir", default="images", help="Directory to store intermediate images")
    parser.add_argument("--start-page", type=int, default=1, help="First page to process (1-based)")
    parser.add_argument("--end-page", type=int, default=None, help="Last page to process (inclusive). If omitted, process only start-page")
    parser.add_argument("--force-cpu", action="store_true", help="Set CUDA_VISIBLE_DEVICES='' before running to force CPU")

    args = parser.parse_args()

    if args.force_cpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    pdf_path = args.pdf
    if args.end_page is None:
        args.end_page = args.start_page

    # ensure model is available
    if not ensure_model(args.model):
        print(f"Model {args.model} unavailable and could not be pulled.", file=sys.stderr)
        sys.exit(2)

    failed_pages, failed_slices = run_pipeline(pdf_path, args.output, args.model, args.scale, args.max_parts, args.lr_margin, args.images_dir, args.start_page, args.end_page)

    print("Pipeline finished.")
    if failed_pages:
        print("Pages that failed final composition:", failed_pages)
    if failed_slices:
        print("Failed slices (page, path):")
        for p, s in failed_slices:
            print(p, s)


if __name__ == "__main__":
    main()
