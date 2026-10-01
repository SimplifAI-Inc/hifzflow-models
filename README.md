# HifzFlow models

Models that HifzFlow runs on learners' own devices, published so the apps can download them.

## Ayah checker v3 (also trained on practice situations)

`ayah-opening-checker.int8.onnx` (133 MB, same format and vocabulary as v1 and v2) and `ayah-opening-checker.vocab.json`, attached to the [`opening-check-v3` release](https://github.com/SimplifAI-Inc/hifzflow-models/releases/tag/opening-check-v3).

v2, trained further on how learners actually practise (scripts in [`train/`](train): `prep_practice.py`, `mix_practice.py`). In HifzFlow it is the second opinion on every red word: a word turns red only if this model did not hear it either.

**Added training data:** 20,000 practice situations built from whole professional segments of obadx/muaalem-annotated-v3 (MIT, the same 18 reciters and held-out rules as v2). They cover a phrase repeated 2-4 times, going back an ayah, long pauses, another reciter quietly in the background, an ayah followed by a similar passage elsewhere, and a cough or microphone knock. Each is labelled with the words actually recited, repetitions kept. Mixed with v2's training mix and continued from v2 (one epoch, learning rate 2e-5), chosen at step 2,814.

**Measured** (sets never used for training or for choosing this version):

| | v2 | v3 |
|---|---|---|
| Word error, held-out reciters | 0.40% | 0.39% |
| Real learners (RetaSy test half), correct recitations where a word was "not heard" | 7% (3.8% of words) | 4% (2.0% of words) |
| Real learners, recitations with mistakes where a word was not heard | 25% (13.9% of words) | 20% (11.4% of words) |
| As HifzFlow's second opinion, mistake benchmark (700 recordings), mistakes caught | | same as v2, a few kinds 4-8 points higher |

| File | Bytes | SHA-256 |
|---|---|---|
| `ayah-opening-checker.int8.onnx` | 132,714,002 | `9176f407abc47a82608226118c5e0f6bb1acf4705245c0a46b5e3a68ef8efe37` |
| `ayah-opening-checker.vocab.json` | 21,062 | `c55877f3bff8bc3aaefc160e8c2fb88cb349088d092513d40210ccfe535e671b` |

### Retrain it

After v2's steps (see below): `python train/prep_practice.py train` (needs the relabelled clips from HifzFlow's following-model data preparation), `python train/mix_practice.py full/train.jsonl practice_train.jsonl mix_learners.jsonl mix_practice.jsonl`, then `python train/train_ctc.py mix_practice.jsonl held.jsonl runs/prac 1 2e-5 --learners=retasy_val.jsonl --init=runs/learn/step1000.nemo --eval-every=500`.

## Ayah checker v2 (trained on Quran recitation)

`ayah-opening-checker.int8.onnx` (133 MB, same format and vocabulary as v1) and `ayah-opening-checker.vocab.json`, attached to the [`opening-check-v2` release](https://github.com/SimplifAI-Inc/hifzflow-models/releases/tag/opening-check-v2).

The same NVIDIA model as v1 below, **retrained by HifzFlow on Quran recitation** (CTC branch only). The scripts are in [`train/`](train).

**Training data:**

- [obadx/muaalem-annotated-v3](https://huggingface.co/datasets/obadx/muaalem-annotated-v3) (MIT), from 18 professional reciters, about 880 hours including planted mistakes and harder copies. Left out: Al-Husary, Al-Minshawi and Mishary Al-Afasy (HifzFlow's test reciters) and Ahmed Amer (held-out scoring).
  - **Planted mistakes:** a segment skipped, repeated, or followed by one from another surah. Each is labelled with what was actually said.
  - **Harder copies:** noise, phone microphone, higher voice, faster or slower pace, room echo.
- 1,351 recordings by 248 non-Arab learners from the [RetaSy Quranic Audio Dataset](https://huggingface.co/datasets/RetaSy/quranic_audio_dataset) (CC-BY-4.0). These are unlabelled recordings that v1 heard word for word, labelled with their ayah. Learners in the test halves were left out.

**Measured** (test sets never used for training or for choosing this version):

| | v1 (NVIDIA as released) | v2 |
|---|---|---|
| Word error, held-out reciter Ahmed Amer and Al-Minshawi | 5.68% | 0.40% |
| Correctly recited words not heard, HifzFlow's benchmark (5 reciters × 4 conditions) | 1.2–2.2% | 0.0–0.2% |
| Real learners, correct recitations flagged (RetaSy test half) | 5.8% | 5.2% |
| Real learners, recitations with mistakes flagged | 21.7% | 18.7% |

Chosen at 1,000 steps of the learner stage. Later steps flagged fewer correct learners but also fewer mistakes.

| File | Bytes | SHA-256 |
|---|---|---|
| `ayah-opening-checker.int8.onnx` | 132,714,002 | `ee27e775873667bd809b7165297b0d4253b2b55f34853f171ae88700fec899ad` |
| `ayah-opening-checker.vocab.json` | 21,062 | `c55877f3bff8bc3aaefc160e8c2fb88cb349088d092513d40210ccfe535e671b` |

### Retrain it

Linux (WSL works) with a CUDA GPU; a 12 GB RTX 3080 Ti takes under an hour per run. Set up with `build/setup.sh`, then:

```bash
python train/prep_full.py train                  # professional segments, planted mistakes, harder copies
python train/train_ctc.py full/train.jsonl held.jsonl runs/full 1.5 3e-5 --learners=retasy_val.jsonl
python train/prep_retasy_unlabelled.py           # learner recordings (run where quran_transcript is installed)
python train/filter_learners.py unlabelled.jsonl full/train.jsonl mix.jsonl 6 40000
python train/train_ctc.py mix.jsonl held.jsonl runs/learn 2 2e-5 --learners=retasy_val.jsonl --init=runs/full/<last>.nemo
EXPORT_NEMO=runs/learn/step1000.nemo EXPORT_OUT=out python train/export_ckpt.py some.wav
```

## Ayah-opening checker (v1)

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
- **v2 and v3 were also trained on** *muaalem-annotated-v3 by obadx* (MIT) and the *Quranic Audio Dataset: Crowdsourced and Labeled Recitation from Non-Arabic Speakers by RetaSy* (CC-BY-4.0).
- **The scripts in `build/` and `train/`** are MIT; see [`LICENSE`](LICENSE).
