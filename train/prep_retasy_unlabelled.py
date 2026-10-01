"""RetaSy recordings with no label, from learners in neither test half: 16 kHz WAV plus the
ayah they recite (matched by text), for training on learner voices. Which of them are
used is decided later (only those the original model heard word for word)."""
import glob, io, json, os, pathlib, re, difflib
import numpy as np, pyarrow.parquet as pq, soundfile as sf, librosa
from quran_transcript import Aya

here = pathlib.Path(__file__).parent
out = here / "unlabelled"; out.mkdir(exist_ok=True)
SURAH = {"Al-Faatihah": 1, "Al-Ikhlas": 112, "An-Nas": 114, "Al-Kafiroon": 109, "Ayat al-Kursi": 2, "Al-Falaq": 113,
         "Al-Asr": 103, "Al-Kauthar": 108, "Al-Masad": 111, "Al-Humazah": 104, "An-Nasr": 110, "Quraish": 106,
         "Al-Maun": 107, "Al-Fil": 105, "At-Takathur": 102, "Al-Qariah": 101}
letters = lambda s: re.sub(r"[^ء-غف-يٱ]", "", s.replace("ٱ", "ا"))
texts = {}


def ayahs(s):
    if s not in texts:
        texts[s] = []
        for a in range(1, (7 if s == 1 else 286 if s == 2 else 30) + 1):
            try:
                ay = Aya(s, a).get()
                texts[s].append((a, letters(ay.uthmani), getattr(ay, "imlaey", None) or ay.uthmani))
            except Exception:
                break
    return texts[s]


tested = {json.loads(l)["reciter"] for f in ("retasy_val.jsonl", "retasy_test.jsonl") for l in open(here / f, encoding="utf8")}
t = pq.read_table(glob.glob(str(here / "raw/data/*.parquet"))).to_pandas()
t = t[t.final_label.isna() & (t.reciter_id != "Unknown") & (~t.reciter_id.isin(tested))]
rows, skipped = [], 0
for i, r in t.iterrows():
    s = SURAH.get(r["Surah"])
    if not s:
        continue
    want = letters(r["Aya"])
    best = max(ayahs(s), key=lambda x: difflib.SequenceMatcher(None, want, x[1]).ratio())
    if difflib.SequenceMatcher(None, want, best[1]).ratio() < 0.85 or (s == 2 and best[0] != 255):
        skipped += 1
        continue
    wave, rate = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if rate != 16000:
        wave = librosa.resample(wave, orig_sr=rate, target_sr=16000)
    if not (1.0 <= wave.size / 16000 <= 25):
        continue
    name = f"u{i}.wav"
    sf.write(out / name, wave, 16000, subtype="PCM_16")
    rows.append({"audio_filepath": os.path.expanduser(f"~/quran-data/retasy/unlabelled/{name}"), "duration": round(wave.size / 16000, 3),
                 "text": best[2], "key": f"{s}:{best[0]}", "reciter": r["reciter_id"]})
with open(here / "unlabelled.jsonl", "w", encoding="utf8") as fh:
    for row in rows:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
print(len(rows), "recordings,", len({r["reciter"] for r in rows}), "learners,", round(sum(r["duration"] for r in rows) / 3600, 2), "h; skipped", skipped)
