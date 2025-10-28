#!/usr/bin/env python3
"""
image_diagnosis.py

Render a high-resolution PNG of a specified PDF page and send it with a prompt to Ollama.

Usage:
    python image_diagnosis.py --path file.pdf --page 1 --output result.txt

Defaults:
    model: qwen2.5vl:latest
    output: diagnosis_image_output.txt
    scale: 3.0 (roughly 216 DPI if source is 72 DPI)
"""
import argparse
import sys
from pathlib import Path
import fitz  # pymupdf
import ollama
import os
import math

# image handling
try:
    from PIL import Image
except Exception:
    Image = None

# progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


PROMPT = """
Your task is to extract mathematical expressions from a latexPDF that is provided to you as a png. 
Extract all lemmas, definitions and theorems. 
Output the result in latex format. Only output the latex code, do not include any of your thinking or explanations.
"""


def render_page_to_png(pdf_path: str, page_number: int, out_path: str, scale: float = 3.0) -> None:
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(p))
    try:
        page_count = doc.page_count
    except Exception:
        page_count = len(doc)

    if page_number < 1 or page_number > page_count:
        raise IndexError(f"page {page_number} out of range (1-{page_count})")

    page = doc[page_number - 1]
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix)

    out_dir = Path(out_path).parent
    if out_dir and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        f.write(pix.tobytes("png"))


def ensure_model(model_name: str) -> bool:
    try:
        models = ollama.list()
    except Exception:
        # If ollama.list() fails, try to continue and let ollama.generate raise a clearer error
        models = {}

    if not models.get(model_name):
        try:
            print(f"Pulling model {model_name}...")
            ollama.pull(model_name)
        except Exception as e:
            print(f"Failed to pull model {model_name}: {e}", file=sys.stderr)
            return False
    return True


def call_model_with_image(model_name: str, prompt: str, image_path: str):
    try:
        response = ollama.generate(model=model_name, prompt=prompt, images=[image_path])
    except Exception as e:
        raise RuntimeError(f"ollama.generate failed: {e}")
    return response


def split_image_vertically(input_image_path: str, parts: int = 6, overlap: float = 0.1, out_dir: str = "images", lr_margin: float = 0.15):
    """Split the image top->bottom into `parts` overlapping slices.

    Steps:
    - Trim left/right margins defined by lr_margin (fraction of width).
    - Split the remaining central region into `parts` slices with `overlap` fraction overlap.

    overlap is fraction (0.0-0.9) of per-part height to overlap.
    Returns list of (path, (top, bottom, left_crop, right_crop)) tuples.
    """
    if Image is None:
        raise RuntimeError("Pillow is required for image splitting. Install with: pip install pillow")

    img = Image.open(input_image_path)
    w, h = img.size

    # Validate lr_margin
    if lr_margin < 0 or lr_margin >= 0.45:
        # prevent trimming too much; 45% ensures at least some width remains
        raise ValueError("lr_margin must be between 0.0 and 0.45 (fraction)")

    # Crop left/right margins first
    left_crop = int(round(w * lr_margin))
    right_crop = w - left_crop
    if left_crop >= right_crop:
        raise ValueError("Calculated crop bounds are invalid; reduce lr_margin")

    img_central = img.crop((left_crop, 0, right_crop, h))
    w_c, h_c = img_central.size

    # Nominal per-part height (use central height)
    nominal = math.ceil(h_c / parts)
    overlap_pixels = int(nominal * overlap)
    step = nominal - overlap_pixels
    if step <= 0:
        step = 1

    starts = list(range(0, h_c, step))

    slices = []
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, start in enumerate(starts):
        if count >= parts:
            break
        end = start + nominal
        if end > h_c:
            end = h_c
        if start >= end:
            break

        crop = img_central.crop((0, start, w_c, end))
        out_path = out_dir_path / f"{Path(input_image_path).stem}_part_{count + 1}.png"
        crop.save(out_path)
        # Translate coords back to original image coordinates for reference
        slices.append((str(out_path), (start, end, left_crop, right_crop)))
        count += 1
        if end == h_c:
            break

    return slices


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a PDF page to PNG and send to Ollama model")
    parser.add_argument("--path", required=True, help="Path to the PDF file")
    parser.add_argument("--page", type=int, required=True, help="Page number (1-based)")
    parser.add_argument("--output", default="diagnosis_image_output.txt", help="Output text file for model response")
    parser.add_argument("--model", default="qwen2.5vl:3b", help="Ollama model name")
    parser.add_argument("--scale", type=float, default=3.0, help="Render scale multiplier (1.0 = native, 3.0 = high res)")
    parser.add_argument("--images-dir", default="images", help="Directory to store rendered images")
    parser.add_argument("--parts", type=int, default=6, help="Number of vertical parts to split into (default 6)")
    parser.add_argument("--overlap", type=float, default=0.1, help="Overlap fraction between parts (default 0.1 = 10%)")
    parser.add_argument("--lr-margin", type=float, default=0.15, help="Left/right margin fraction to trim before splitting (default 0.15 = 15%)")

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    image_path = images_dir / f"page_{args.page}.png"

    try:
        render_page_to_png(args.path, args.page, str(image_path), scale=args.scale)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except IndexError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"Failed to render page: {e}", file=sys.stderr)
        sys.exit(4)

    print(f"Rendered page {args.page} to {image_path}")

    if not ensure_model(args.model):
        print("Model unavailable and could not be pulled.", file=sys.stderr)
        sys.exit(5)
    # Split rendered image into overlapping vertical slices
    try:
        slices = split_image_vertically(str(image_path), parts=args.parts, overlap=args.overlap, out_dir=str(images_dir), lr_margin=args.lr_margin)
    except Exception as e:
        print(f"Failed to split image: {e}", file=sys.stderr)
        sys.exit(8)

    if not slices:
        print("No image slices generated.", file=sys.stderr)
        sys.exit(9)

    # Prepare progress iterator
    if tqdm is None:
        def simple_tqdm(it, total=None):
            for idx, item in enumerate(it, 1):
                print(f"Processing part {idx}/{total or '?'}")
                yield item
        progress_iter = simple_tqdm(slices, total=len(slices))
    else:
        progress_iter = tqdm(slices, desc="Model parts", unit="part")

    responses = []
    for idx, (slice_path, coords) in enumerate(progress_iter, start=1):
        try:
            resp = call_model_with_image(args.model, PROMPT, slice_path)
        except Exception as e:
            print(f"Model call failed for slice {idx}: {e}", file=sys.stderr)
            resp = f"""% ERROR on slice {idx}: {e}"""
        responses.append((idx, slice_path, resp))

    # Write concatenated responses to output. Use LaTeX-safe comment separators so concatenation stays valid
    out_path = Path(args.output)
    try:
        with out_path.open("w", encoding="utf-8") as f:
            for idx, slice_path, resp in responses:
                f.write(f"% --- PART {idx} from {Path(slice_path).name} ---\n")
                f.write(str(resp))
                f.write("\n\n")
    except Exception as e:
        print(f"Failed to write output file: {e}", file=sys.stderr)
        sys.exit(10)

    print(f"Wrote model responses for {len(responses)} parts to {out_path}")


if __name__ == "__main__":
    main()
