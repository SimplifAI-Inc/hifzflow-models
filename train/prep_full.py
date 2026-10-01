"""Pilot training data from obadx/muaalem-annotated-v3 (MIT): each segment as 16 kHz WAV
with its exact diacritized text, plus planted mistakes (labelled with what was actually
said) and harder copies (noise, phone microphone, higher voice). NeMo JSONL manifests.

  python prep_pilot.py held|train|all
"""
import glob, io, json, os, random, re, sys
import numpy as np, pyarrow.parquet as pq, soundfile as sf, librosa

random.seed(7)
rng = np.random.default_rng(7)
root = os.path.expanduser("~/quran-data")
out = f"{root}/full"
os.makedirs(f"{out}/wav", exist_ok=True)
RATE = 16000
# Every downloaded reciter except the held-out ones (the benchmark's reciters were never downloaded).
TRAIN = tuple(sorted(d for d in os.listdir(f"{root}/muaalem") if d.startswith("moshaf_") and "metadata" not in d and d not in ("moshaf_5.0", "moshaf_1.0")))
HELD = ("moshaf_5.0", "moshaf_1.0")


def text_of(imlaey):
    # The model has no token for the dagger alif; standard spelling leaves it out.
    return re.sub(r"\s+", " ", imlaey.replace("ٰ", "")).strip()


def noisy(w):
    n = np.cumsum(rng.normal(size=w.size)).astype(np.float32)
    n -= np.convolve(n, np.ones(400) / 400, mode="same")
    snr = rng.uniform(5, 20)
    return (w + n * np.sqrt(np.mean(w**2) / (np.mean(n**2) + 1e-9)) * 10 ** (-snr / 20)).astype(np.float32)


def phone(w):
    s = np.fft.rfft(w)
    f = np.fft.rfftfreq(w.size, 1 / RATE)
    s[(f < 300) | (f > 3400)] = 0
    return np.clip(np.fft.irfft(s, n=w.size) * 1.8, -0.5, 0.5).astype(np.float32)


def speed(w):
    # A learner's own pace: 0.88-1.12 times as fast (pitch moves a little too).
    f = float(rng.uniform(0.88, 1.12))
    return librosa.resample(w, orig_sr=int(RATE * f), target_sr=RATE).astype(np.float32)


def room(w):
    # A small room's echo: decaying noise as the impulse response.
    n = int(RATE * rng.uniform(0.15, 0.5))
    ir = rng.normal(size=n).astype(np.float32) * np.exp(-np.linspace(0, rng.uniform(4, 9), n)).astype(np.float32)
    ir[0] = 1.0
    out = np.convolve(w, ir)[: w.size]
    return (out / (np.abs(out).max() + 1e-6) * max(np.abs(w).max(), 1e-3)).astype(np.float32)


def higher(w):
    return librosa.effects.pitch_shift(w, sr=RATE, n_steps=float(rng.uniform(2, 5))).astype(np.float32)


def _unused_segments(moshafs):
    cols = ["audio", "imlaey", "has_quran", "match_ratio", "duration_seconds", "segment_index", "sura_or_aya_index", "moshaf_id"]
    for m in moshafs:
        for f in sorted(glob.glob(f"{root}/muaalem/{m}/*.parquet")):
            for r in pq.read_table(f, columns=cols).to_pylist():
                if not r["has_quran"] or r["match_ratio"] < 0.95 or not (1.0 <= r["duration_seconds"] <= 20):
                    continue
                w, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if w.ndim > 1:
                    w = w.mean(axis=1)
                if sr != RATE:
                    w = librosa.resample(w, orig_sr=sr, target_sr=RATE)
                yield {"id": f"{r['moshaf_id']}-{r['segment_index']}", "sura": r["sura_or_aya_index"], "wave": w.astype(np.float32), "text": text_of(r["imlaey"])}


def build_file(args):
    """One parquet file, streamed: a few recent segments kept for planted mistakes."""
    f, plant, part = args
    local = np.random.default_rng(abs(hash(f)) % 2**32)
    lrandom = random.Random(f)
    counts = {}
    recent, pool = [], []
    with open(part, "w", encoding="utf8") as fh:
        def emit(key, wave, text, kind):
            path = f"{out}/wav/{key}.wav"
            sf.write(path, wave, RATE, subtype="PCM_16")
            fh.write(json.dumps({"audio_filepath": path, "duration": round(wave.size / RATE, 3), "text": text, "kind": kind}, ensure_ascii=False) + "\n")
            counts[kind] = counts.get(kind, 0) + 1

        for it in segments_of(f):
            emit(it["id"], it["wave"], it["text"], "correct")
            if plant:
                roll = local.random()
                if roll < 0.13:
                    emit(it["id"] + "-noisy", noisy(it["wave"]), it["text"], "noisy")
                elif roll < 0.26:
                    emit(it["id"] + "-phone", phone(it["wave"]), it["text"], "phone")
                elif roll < 0.36:
                    emit(it["id"] + "-higher", higher(it["wave"]), it["text"], "higher")
                elif roll < 0.45:
                    emit(it["id"] + "-speed", speed(it["wave"]), it["text"], "speed")
                elif roll < 0.52:
                    emit(it["id"] + "-room", phone(room(it["wave"])) if local.random() < 0.5 else room(it["wave"]), it["text"], "room")
                gap = np.zeros(int(local.uniform(0.1, 0.5) * RATE), np.float32)
                roll = local.random()
                # Skip: the segment two back, then this one (the one between left out).
                back = recent[-2] if len(recent) >= 2 and recent[-2]["sura"] == it["sura"] else None
                if roll < 0.06 and back and back["wave"].size + it["wave"].size < 25 * RATE:
                    emit(back["id"] + "-skip", np.concatenate([back["wave"], gap, it["wave"]]), back["text"] + " " + it["text"], "skip")
                elif roll < 0.09 and it["wave"].size * 2 < 25 * RATE:
                    emit(it["id"] + "-repeat", np.concatenate([it["wave"], gap, it["wave"]]), it["text"] + " " + it["text"], "repeat")
                elif roll < 0.13 and pool:
                    other = pool[lrandom.randrange(len(pool))]
                    if other["sura"] != it["sura"] and it["wave"].size + other["wave"].size < 25 * RATE:
                        emit(it["id"] + "-other", np.concatenate([it["wave"], gap, other["wave"]]), it["text"] + " " + other["text"], "other")
                recent = (recent + [it])[-2:]
                if len(pool) < 60:
                    pool.append(it)
                elif local.random() < 0.05:
                    pool[lrandom.randrange(60)] = it
    return counts


def segments_of(f):
    cols = ["audio", "imlaey", "has_quran", "match_ratio", "duration_seconds", "segment_index", "sura_or_aya_index", "moshaf_id"]
    for r in pq.read_table(f, columns=cols).to_pylist():
        if not r["has_quran"] or r["match_ratio"] < 0.95 or not (1.0 <= r["duration_seconds"] <= 20):
            continue
        w, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if w.ndim > 1:
            w = w.mean(axis=1)
        if sr != RATE:
            w = librosa.resample(w, orig_sr=sr, target_sr=RATE)
        yield {"id": f"{r['moshaf_id']}-{r['segment_index']}", "sura": r["sura_or_aya_index"], "wave": w.astype(np.float32), "text": text_of(r["imlaey"])}


def build(moshafs, manifest, plant):
    from multiprocessing import Pool
    files = [f for m in moshafs for f in sorted(glob.glob(f"{root}/muaalem/{m}/*.parquet"))]
    parts = [f"{out}/{manifest}.part{i}" for i in range(len(files))]
    with Pool(8) as pool:
        results = pool.map(build_file, [(f, plant, p) for f, p in zip(files, parts)])
    totals = {}
    for c in results:
        for k, v in c.items():
            totals[k] = totals.get(k, 0) + v
    with open(f"{out}/{manifest}", "w", encoding="utf8") as fh:
        for p in parts:
            fh.write(open(p, encoding="utf8").read())
            os.remove(p)
    print("wrote", manifest, totals, flush=True)


if __name__ == "__main__":
    which = sys.argv[1]
    if which in ("held", "all"):
        build(HELD, "held.jsonl", plant=False)
    if which in ("train", "all"):
        build(TRAIN, "train.jsonl", plant=True)
