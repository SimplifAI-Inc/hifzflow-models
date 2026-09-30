"""Quantize the exported checker to int8 per channel: ayah-opening-checker.int8.onnx.

Run after export.py (which writes ~/fcq/out/fastconformer_ar_ctc_fp32.onnx).
Per-channel int8 kept the checker's results closest to the full-precision
model on HifzFlow's recitation benchmark; plain int8 flagged more correct
openings, and fp16 did not load in onnxruntime.
"""
import os
from onnxruntime.quantization import QuantType, quantize_dynamic

out = os.path.expanduser("~/fcq/out")
src = os.path.join(out, "fastconformer_ar_ctc_fp32.onnx")
dst = os.path.join(out, "ayah-opening-checker.int8.onnx")
quantize_dynamic(src, dst, weight_type=QuantType.QInt8, per_channel=True)
print(dst, round(os.path.getsize(dst) / 1e6, 1), "MB")
