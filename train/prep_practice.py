"""Practice situations for the following model, built from whole professional
segments (never cut inside a word), each labelled with exactly what was said:

  repeat    a phrase recited 2-4 times, as when memorising
  back      ayah n, n+1, then back to n (or n+1 again): going backwards
  pause     a long pause before the opening, or between two ayahs
  voice     another reciter quietly in the background (a teacher, a class,
            audio playing); only the learner's words are the label
  similar   an ayah, then a passage elsewhere that opens with the same sounds
            (similar-ayah confusion)
  bump      a short noise between ayahs (cough, microphone knock)

Each label is the sounds actually recited in order, repetitions kept. From the
training reciters only; `held` builds the same situations from the held-out
reciters, for testing.

  python prep_practice.py train|held
"""
import json, os, random, re, sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import soundfile as sf

RATE = 16000
MAX = 25 * RATE
which = sys.argv[1]
home = os.path.expanduser("~")
src = f"{home}/zf/data/{'train' if which == 'train' else 'held'}.jsonl"
out = f"{home}/zf/data/practice_{which}"
# PRACTICE_OUT: write the same clips elsewhere (e.g. again, now with their parts listed).
out = os.environ.get("PRACTICE_OUT", out)
os.makedirs(f"{out}/wav", exist_ok=True)
COUNT = 48000 if which == "train" else 900

rows = [json.loads(l) for l in open(src, encoding="utf8")]
rows = [r for r in rows if r["kind"] == "correct"]
by_surah = defaultdict(list)
by_moshaf = defaultdict(list)
for r in rows:
    m = re.match(r"^([0-9.]+)-(\d+)\.(\d+)$", r["id"])
    r["moshaf"], r["surah"], r["n"] = m.group(1), int(m.group(2)), int(m.group(3))
    by_surah[(r["moshaf"], r["surah"])].append(r)
    by_moshaf[r["moshaf"]].append(r)
for v in by_surah.values():
    v.sort(key=lambda r: r["n"])
# Openings shared with another place: the first 10 sounds (spaces dropped).
opening = defaultdict(list)
for r in rows:
    opening[(r["moshaf"], r["text"].replace(" ", "")[:10])].append(r)


def load(r):
    w, _ = sf.read(r["path"], dtype="float32")
    return w


def gap(rng, lo, hi):
    n = int(rng.uniform(lo, hi) * RATE)
    return (rng.normal(size=n) * rng.uniform(1e-4, 2e-3)).astype(np.float32)


def bump(rng):
    n = int(rng.uniform(0.15, 0.7) * RATE)
    burst = rng.normal(size=n).astype(np.float32) * np.exp(-np.linspace(0, rng.uniform(3, 8), n)).astype(np.float32)
    return burst * rng.uniform(0.05, 0.4)


def phone(w):
    s = np.fft.rfft(w)
    f = np.fft.rfftfreq(w.size, 1 / RATE)
    s[(f < 300) | (f > 3400)] = 0
    return np.clip(np.fft.irfft(s, n=w.size) * 1.8, -0.5, 0.5).astype(np.float32)


KINDS = ["repeat", "repeat", "back", "back", "pause", "pause", "voice", "voice", "similar", "bump"]


def make(i):
    # The situation is fixed by i; a passage that does not fit is tried again elsewhere.
    kind = KINDS[i % len(KINDS)]
    for attempt in range(12):
        made = make_once(i, kind, attempt)
        if made:
            return made
    return None


def make_once(i, kind, attempt):
    rng = np.random.default_rng(1000 + i * 16 + attempt)
    pick = random.Random(1000 + i * 16 + attempt)
    seq = by_surah[pick.choice(list(by_surah))]
    k = pick.randrange(len(seq))
    parts, said, ids = [], [], []

    def say(r, before=None):
        if before is not None:
            parts.append(before)
        parts.append(load(r))
        said.append(r["text"])
        ids.append(r["id"])

    if kind == "repeat":
        r = seq[k]
        for t in range(pick.randint(2, 4)):
            say(r, gap(rng, 0.2, 1.5) if t else None)
        if k + 1 < len(seq) and pick.random() < 0.5:
            say(seq[k + 1], gap(rng, 0.2, 1.0))
    elif kind == "back":
        if k + 2 >= len(seq):
            return None
        say(seq[k])
        say(seq[k + 1], gap(rng, 0.1, 0.8))
        say(seq[k] if pick.random() < 0.6 else seq[k + 1], gap(rng, 0.3, 2.0))
        if pick.random() < 0.5:
            say(seq[k + 1], gap(rng, 0.1, 0.8))
    elif kind == "pause":
        if pick.random() < 0.5 or k + 1 >= len(seq):
            say(seq[k], gap(rng, 1.5, 6.0))
        else:
            say(seq[k])
            say(seq[k + 1], gap(rng, 2.0, 6.0))
    elif kind == "voice":
        say(seq[k])
        if k + 1 < len(seq) and pick.random() < 0.5:
            say(seq[k + 1], gap(rng, 0.2, 1.0))
    elif kind == "similar":
        r = seq[k]
        nxt = seq[k + 1] if k + 1 < len(seq) else None
        if nxt is None:
            return None
        others = [o for o in opening[(r["moshaf"], nxt["text"].replace(" ", "")[:10])] if (o["surah"], o["n"]) != (nxt["surah"], nxt["n"]) and o["text"] != nxt["text"]]
        if not others:
            return None
        say(r)
        say(pick.choice(others), gap(rng, 0.1, 0.8))
    else:  # bump
        if k + 1 >= len(seq):
            return None
        say(seq[k])
        say(seq[k + 1], np.concatenate([gap(rng, 0.05, 0.3), bump(rng), gap(rng, 0.05, 0.4)]))
    if pick.random() < 0.3:
        parts.append(gap(rng, 0.2, 1.5))
    if sum(p.size for p in parts) > MAX:
        return None
    w = np.concatenate(parts)
    if kind == "voice":
        other = load(pick.choice(by_moshaf[pick.choice([m for m in by_moshaf if m != seq[0]["moshaf"]] or list(by_moshaf))]))
        other = np.resize(other, w.size) if other.size < w.size else other[: w.size]
        snr = rng.uniform(15, 30)
        scale = np.sqrt(np.mean(w**2) / (np.mean(other**2) + 1e-9)) * 10 ** (-snr / 20)
        w = w + other * scale
    if pick.random() < 0.2:
        w = phone(w)
    w = np.clip(w, -1, 1).astype(np.float32)
    path = f"{out}/wav/p{i}-{kind}.wav"
    sf.write(path, w, RATE, subtype="PCM_16")
    return {"id": f"practice{'' if which == 'train' else 'held'}-{i}-{kind}", "path": path, "duration": round(w.size / RATE, 3), "text": " ".join(said), "kind": kind, "parts": ids}


if __name__ == "__main__":
    with Pool(8) as pool:
        made = [m for m in pool.map(make, range(int(COUNT * 1.15)), chunksize=100) if m][:COUNT]
    with open(f"{out}.jsonl" if "PRACTICE_OUT" in os.environ else f"{home}/zf/data/practice_{which}.jsonl", "w", encoding="utf8") as fh:
        for m in made:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    counts = defaultdict(int)
    for m in made:
        counts[m["kind"]] += 1
    print(which, len(made), dict(counts), round(sum(m["duration"] for m in made) / 3600, 1), "hours")
