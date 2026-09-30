# HifzFlow models

Models that HifzFlow runs on learners' own devices, published so the apps can download them.

## Ayah-opening checker

`ayah-opening-checker.int8.onnx` (133 MB) and `ayah-opening-checker.vocab.json`, attached to the [`opening-check-v1` release](https://github.com/SimplifAI-Inc/hifzflow-models/releases/tag/opening-check-v1).

After a learner recites an ayah, HifzFlow transcribes that ayah's audio with this model and checks whether the ayah began with the right word. It runs on the device; no audio leaves it.

### Where it comes from

This is **NVIDIA's [stt_ar_fastconformer_hybrid_large_pcd_v1.0](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0)** (Arabic FastConformer, about 115M parameters), **licensed [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)** by NVIDIA. The weights have not been retrained.

**Changes made** (scripts in [`build/`](build)):

- Exported only the CTC branch of the hybrid model to ONNX (opset 17).
- Built the model's log-mel feature extraction into the graph, as a fixed real-valued DFT (`conv1d`), so the input is raw 16 kHz mono audio (`audio_signal` `[1, N]` float32, `length` `[1]` int64). The output is `logprobs` `[1, T, 1025]`, with blank id 1024.
- Quantized the weights to int8 per channel (onnxruntime dynamic quantization).

**Checked:**

- Features match NeMo's own preprocessor to within 0.002.
- Transcripts match NeMo's own transcription of the same audio.
- The vocabulary (`ayah-opening-checker.vocab.json`, token id to text) is the model's SentencePiece vocabulary plus `1024: <blank>`.

| File | Bytes | SHA-256 |
|---|---|---|
| `ayah-opening-checker.int8.onnx` | 132,714,002 | `0cc2cdcb49b3f169b4cf19c112ebb1dc9f32f839ccf43e928aa1522f3a4bcbfc` |
| `ayah-opening-checker.vocab.json` | 21,062 | `c55877f3bff8bc3aaefc160e8c2fb88cb349088d092513d40210ccfe535e671b` |

### Rebuild it

The scripts need Linux (WSL works) with a CUDA-capable GPU, or CPU only (slower):

```bash
bash build/setup.sh                 # Python 3.11 venv with NVIDIA NeMo
python build/export.py some.wav     # export and check against NeMo on some.wav (16 kHz mono)
python build/quantize.py            # int8 per channel: ayah-opening-checker.int8.onnx
```

## Licences

- **The model files** are NVIDIA's model with the changes above, under **CC-BY-4.0**; see [`CC-BY-4.0.txt`](CC-BY-4.0.txt). Credit: *Arabic FastConformer model by NVIDIA (stt_ar_fastconformer_hybrid_large_pcd_v1.0), CC-BY-4.0.*
- **The scripts in `build/`** are MIT; see [`LICENSE`](LICENSE).
