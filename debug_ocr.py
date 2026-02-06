#!/usr/bin/env python3
"""Debug script to see raw PaddleOCR v5 output format."""

import sys
import os

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import fitz

# Test PDF
pdf_path = "../test_plans/research_parking_test.pdf"

# Convert first page to image
print("Converting PDF to image...")
doc = fitz.open(pdf_path)
page = doc[0]
mat = fitz.Matrix(200 / 72, 200 / 72)
pix = page.get_pixmap(matrix=mat)
img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
doc.close()

# Resize for speed
max_dim = 1600
w, h = img.size
if max(w, h) > max_dim:
    scale = max_dim / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    print(f"Resized to {img.size}")

# Run OCR
print("Initializing PaddleOCR...")
ocr = PaddleOCR(
    lang='en',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

print("Running OCR...")
result = ocr.ocr(np.array(img))

print(f"\nResult[0] type: {type(result[0])}")

res = result[0].json['res']
print(f"\nKeys in res: {list(res.keys())}")

# Show the actual OCR data
texts = res.get('rec_texts', [])
scores = res.get('rec_scores', [])
polys = res.get('rec_polys', [])
boxes = res.get('rec_boxes', [])
dt_polys = res.get('dt_polys', [])

print(f"\nrec_texts: {len(texts)} items")
print(f"rec_scores: {len(scores)} items")
print(f"rec_polys: {len(polys)} items")
print(f"rec_boxes: {len(boxes)} items")
print(f"dt_polys: {len(dt_polys)} items")

# Show first few text results
print("\n--- First 5 OCR Results ---")
for i in range(min(5, len(texts))):
    print(f"\nText {i}: '{texts[i]}'")
    print(f"  Score: {scores[i] if i < len(scores) else 'N/A'}")
    if i < len(polys):
        poly = polys[i]
        print(f"  Poly type: {type(poly)}, len: {len(poly) if hasattr(poly, '__len__') else 'N/A'}")
        if hasattr(poly, 'tolist'):
            print(f"  Poly value: {poly.tolist()[:2]}...")
        else:
            print(f"  Poly value: {poly[:2] if len(poly) > 2 else poly}...")
    if i < len(boxes):
        box = boxes[i]
        print(f"  Box type: {type(box)}, shape: {box.shape if hasattr(box, 'shape') else 'N/A'}")
        if hasattr(box, 'tolist'):
            print(f"  Box value: {box.tolist()}")
        else:
            print(f"  Box value: {box}")

# Show how it matches with dt_polys
print(f"\n\n--- Detection polys (dt_polys) ---")
if dt_polys:
    for i in range(min(3, len(dt_polys))):
        poly = dt_polys[i]
        print(f"dt_poly {i}: type={type(poly)}")
        if hasattr(poly, 'tolist'):
            print(f"  value: {poly.tolist()}")
        elif hasattr(poly, '__len__'):
            print(f"  value: {poly}")
