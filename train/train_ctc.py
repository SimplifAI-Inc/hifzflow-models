"""Fine-tune NVIDIA FastConformer (stt_ar_fastconformer_hybrid_large_pcd_v1.0, CC-BY-4.0),
CTC branch only, on Quran recitation. Measures word error on held-out reciters first
(baseline) and during training; keeps the best model.

  python train_ctc.py <train.jsonl> <held.jsonl> <out dir> [epochs] [peak lr] [--baseline-only]
"""
import json, math, os, random, re, sys, time
import numpy as np, soundfile as sf, torch
import nemo.collections.asr as nemo_asr

train_path, held_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
EPOCHS = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
PEAK = float(sys.argv[5]) if len(sys.argv) > 5 else 5e-5
BASELINE_ONLY = "--baseline-only" in sys.argv
LEARNERS = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--learners=")), None)
MAX_SECONDS, MAX_BATCH = 320, 32
EVAL_EVERY = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--eval-every=")), "2000"))
os.makedirs(out, exist_ok=True)
random.seed(1)
torch.manual_seed(1)

INIT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--init=")), None)
# --init=<model.nemo> continues from an earlier run instead of NVIDIA's model.
m = (nemo_asr.models.ASRModel.restore_from(INIT, map_location="cuda") if INIT
     else nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0", map_location="cuda"))
tok = m.tokenizer
BLANK = m.ctc_decoder.num_classes_with_blank - 1
for name, p in m.named_parameters():
    p.requires_grad = not (name.startswith("decoder.") or name.startswith("joint."))
ctc = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)


def load(path):
    rows = [json.loads(line) for line in open(path, encoding="utf8")]
    for r in rows:
        r["ids"] = tok.text_to_ids(r["text"].replace("ٰ", ""))
        if "duration" not in r:
            r["duration"] = sf.info(r["audio_filepath"]).duration
    return [r for r in rows if r["ids"]]


def batches(rows, shuffle):
    rows = sorted(rows, key=lambda r: r["duration"])
    out_b, cur = [], []
    for r in rows:
        if cur and (len(cur) >= MAX_BATCH or (len(cur) + 1) * r["duration"] > MAX_SECONDS):
            out_b.append(cur)
            cur = []
        cur.append(r)
    if cur:
        out_b.append(cur)
    if shuffle:
        random.shuffle(out_b)
    return out_b


def collate(b):
    waves = [sf.read(r["audio_filepath"], dtype="float32")[0] for r in b]
    n = max(w.size for w in waves)
    audio = torch.zeros(len(b), n)
    lens = torch.tensor([w.size for w in waves])
    for i, w in enumerate(waves):
        audio[i, : w.size] = torch.from_numpy(w)
    t = max(len(r["ids"]) for r in b)
    targets = torch.zeros(len(b), t, dtype=torch.long)
    tlen = torch.tensor([len(r["ids"]) for r in b])
    for i, r in enumerate(b):
        targets[i, : len(r["ids"])] = torch.tensor(r["ids"])
    return audio, lens, targets, tlen


class Batches(torch.utils.data.Dataset):
    def __init__(self, bs):
        self.bs = bs

    def __len__(self):
        return len(self.bs)

    def __getitem__(self, i):
        return collate(self.bs[i])


def logits(audio, lens, train):
    sig, sl = m.preprocessor(input_signal=audio, length=lens)
    if train and getattr(m, "spec_augmentation", None) is not None:
        sig = m.spec_augmentation(input_spec=sig, length=sl)
    enc, el = m.encoder(audio_signal=sig, length=sl)
    return m.ctc_decoder(encoder_output=enc), el


TASHKEEL = re.compile("[ً-ٰٟۖ-ۭـ]")


def plain(s):
    """Letters only, as the app compares words (no diacritics, spelling variants, doubled letters)."""
    s = TASHKEEL.sub("", s)
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ٱ", "ا"), ("ى", "ي"), ("ة", "ه"), ("ؤ", "و"), ("ئ", "ي")):
        s = s.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", re.sub(r"[^ء-ي ]", " ", s)).split()


def edits(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(b)]


def similar(a, b):
    return 1 - edits(list(a), list(b)) / max(1, len(a), len(b))


def same_word(e, h):
    # As the app compares (opening-check.ts): close enough, or cut short.
    return similar(e, h) >= 0.5 or (min(len(e), len(h)) >= 2 and (e.startswith(h) or h.startswith(e)))


def not_heard(expected, heard):
    """Expected words (indexes) not heard: left out or heard as another word (opening-check.ts notHeard)."""
    m_, n_ = len(expected), len(heard)
    cost = lambda i, j: 0 if same_word(expected[i], heard[j]) else 1
    d = [[i if j == 0 else (j if i == 0 else 0) for j in range(n_ + 1)] for i in range(m_ + 1)]
    for i in range(1, m_ + 1):
        for j in range(1, n_ + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost(i - 1, j - 1))
    missing, i, j = set(), m_, n_
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + cost(i - 1, j - 1):
            if cost(i - 1, j - 1):
                missing.add(i - 1)
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            missing.add(i - 1)
            i -= 1
        else:
            j -= 1
    return missing


@torch.no_grad()
def transcribe(rows):
    m.eval()
    heard = {}
    for b in batches(rows, False):
        audio, lens, _, _ = collate(b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp, el = logits(audio.cuda(), lens.cuda(), False)
        best = lp.argmax(-1).cpu().numpy()
        for i, r in enumerate(b):
            seq = best[i][: int(el[i])]
            ids = [int(t) for j, t in enumerate(seq) if t != BLANK and (j == 0 or t != seq[j - 1])]
            heard[r["audio_filepath"]] = tok.ids_to_text(ids) if ids else ""
    m.train()
    return heard


def learner_flags(rows):
    """Share of real learners' recordings with a word not heard: correct ones (false alarms) and ones with mistakes."""
    heard = transcribe(rows)
    tally = {}
    for r in rows:
        missing = not_heard(plain(r["text"]), plain(heard[r["audio_filepath"]]))
        t = tally.setdefault(r["label"], [0, 0])
        t[0] += 1
        t[1] += 1 if missing else 0
    return {k: round(100 * v[1] / v[0], 1) for k, v in tally.items()}


@torch.no_grad()
def evaluate(rows, limit=1500):
    m.eval()
    werr = wtot = cerr = ctot = 0
    for b in batches(rows[:limit], False):
        audio, lens, _, _ = collate(b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp, el = logits(audio.cuda(), lens.cuda(), False)
        best = lp.argmax(-1).cpu().numpy()
        for i, r in enumerate(b):
            seq = best[i][: int(el[i])]
            ids = [int(t) for j, t in enumerate(seq) if t != BLANK and (j == 0 or t != seq[j - 1])]
            heard = tok.ids_to_text(ids) if ids else ""
            ref, hyp = plain(r["text"]), plain(heard)
            werr += edits(ref, hyp)
            wtot += len(ref)
            rc, hc = r["text"].replace(" ", ""), heard.replace(" ", "")
            cerr += edits(list(rc), list(hc))
            ctot += len(rc)
    m.train()
    return 100 * werr / max(1, wtot), 100 * cerr / max(1, ctot)


held = load(held_path)
random.shuffle(held)
learners = load(LEARNERS) if LEARNERS else []
base = evaluate(held)
base_l = learner_flags(learners) if learners else {}
print(f"held {len(held)} clips; baseline held-out word error {base[0]:.2f}%, diacritized char error {base[1]:.2f}%; learners flagged {base_l}", flush=True)
log = open(f"{out}/log.jsonl", "a")
log.write(json.dumps({"step": 0, "wer": base[0], "cer": base[1], "learners": base_l}) + "\n")
log.flush()
if BASELINE_ONLY:
    sys.exit(0)

train = load(train_path)
print(f"train {len(train)} clips {sum(r['duration'] for r in train) / 3600:.1f} h", flush=True)
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=PEAK, weight_decay=1e-3)
plan = batches(train, True)
total = int(len(plan) * EPOCHS)
warm = max(1, min(500, total // 10))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
)
best, step, t0 = base[0], 0, time.time()
m.train()
while step < total:
    loader = torch.utils.data.DataLoader(Batches(batches(train, True)), batch_size=None, shuffle=False, num_workers=8, prefetch_factor=4)
    for audio, lens, targets, tlen in loader:
        if step >= total:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp, el = logits(audio.cuda(non_blocking=True), lens.cuda(non_blocking=True), True)
        loss = ctc(lp.float().transpose(0, 1), targets.cuda(), el, tlen.cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 100 == 0:
            print(f"step {step}/{total} loss {loss.item():.3f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        if step % EVAL_EVERY == 0 or step == total:
            wer, cer = evaluate(held)
            fl = learner_flags(learners) if learners else {}
            print(f"step {step}: held-out word error {wer:.2f}% (baseline {base[0]:.2f}%), char error {cer:.2f}%; learners flagged {fl} (baseline {base_l})", flush=True)
            log.write(json.dumps({"step": step, "wer": wer, "cer": cer, "learners": fl}) + "\n")
            log.flush()
            # Every checkpoint is kept: the one to ship is chosen on real learners, not word error alone.
            m.save_to(f"{out}/step{step}.nemo")
            if wer < best:
                best = wer
print(f"done: best held-out word error {best:.2f}% (baseline {base[0]:.2f}%)", flush=True)
