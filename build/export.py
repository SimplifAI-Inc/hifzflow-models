"""Rebuild the FastConformer first-word checker from NVIDIA's public model
(nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0, CC-BY-4.0).

Exports the CTC branch to ONNX with raw 16 kHz audio in and log-probabilities
out: the log-mel features are computed in the graph with a real-valued DFT
(conv1d), so browsers and phones need no audio code of their own. Checks the
export against NeMo's own features and transcript, then quantizes to int8.
"""
import json, os, sys, time
import numpy as np, torch, soundfile as sf
import nemo.collections.asr as nemo_asr

out = os.path.expanduser("~/fcq/out"); os.makedirs(out, exist_ok=True)
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0", map_location="cpu")
model.eval()
model.change_decoding_strategy(decoder_type="ctc")
pre = model.preprocessor
feat = pre.featurizer
cfg = model.cfg.preprocessor
print("preprocessor:", json.dumps({k: cfg.get(k) for k in ["sample_rate", "window_size", "window_stride", "n_fft", "features", "window", "normalize", "dither", "pad_to", "preemph", "log", "log_zero_guard_type", "log_zero_guard_value", "mag_power", "frame_splicing"]}, default=str))


class InGraphFeatures(torch.nn.Module):
    """NeMo's log-mel (FilterbankFeatures) with the STFT as a fixed conv1d."""
    def __init__(self, feat):
        super().__init__()
        self.n_fft = feat.n_fft; self.hop = feat.hop_length; self.win = feat.win_length
        self.preemph = feat.preemph
        window = feat.window if feat.window is not None else torch.hann_window(self.win, periodic=False)
        pad = (self.n_fft - self.win) // 2
        w = torch.zeros(self.n_fft); w[pad:pad + self.win] = window
        k = torch.arange(self.n_fft // 2 + 1).float()[:, None]; n = torch.arange(self.n_fft).float()[None, :]
        ang = 2 * torch.pi * k * n / self.n_fft
        self.register_buffer("cos", (torch.cos(ang) * w)[:, None, :])
        self.register_buffer("sin", (-torch.sin(ang) * w)[:, None, :])
        self.register_buffer("fb", feat.fb.squeeze(0).clone())  # (mels, bins)
        self.guard = float(feat.log_zero_guard_value) if not isinstance(feat.log_zero_guard_value, str) else 2 ** -24
        self.mag_power = feat.mag_power

    def forward(self, audio, length):
        x = torch.cat([audio[:, :1], audio[:, 1:] - self.preemph * audio[:, :-1]], dim=1) if self.preemph else audio
        x = torch.nn.functional.pad(x[:, None, :], (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        re = torch.nn.functional.conv1d(x, self.cos, stride=self.hop)
        im = torch.nn.functional.conv1d(x, self.sin, stride=self.hop)
        power = re * re + im * im
        if self.mag_power != 2.0:
            power = power.sqrt() ** self.mag_power
        mel = torch.matmul(self.fb, power)
        logmel = torch.log(mel + self.guard)
        frames = torch.div(length, self.hop, rounding_mode="floor") + 1
        t = torch.arange(logmel.shape[2], device=logmel.device)[None, None, :]
        mask = (t < frames[:, None, None]).float()
        n = mask.sum(dim=2, keepdim=True)
        mean = (logmel * mask).sum(dim=2, keepdim=True) / n
        var = (((logmel - mean) * mask) ** 2).sum(dim=2, keepdim=True) / (n - 1)
        norm = (logmel - mean) / (var.sqrt() + 1e-5)
        return norm * mask, frames


class Checker(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = InGraphFeatures(model.preprocessor.featurizer)
        self.encoder = model.encoder
        self.ctc = model.ctc_decoder

    def forward(self, audio_signal, length):
        f, flen = self.features(audio_signal, length)
        enc, elen = self.encoder(audio_signal=f, length=flen)
        return self.ctc(encoder_output=enc)


checker = Checker(model).eval()
wav_path = sys.argv[1]
wav, sr = sf.read(wav_path, dtype="float32")
assert sr == 16000
audio = torch.from_numpy(wav)[None, :]; length = torch.tensor([audio.shape[1]])
with torch.no_grad():
    ref_f, ref_len = pre(input_signal=audio, length=length)
    my_f, my_len = checker.features(audio, length)
    t = min(ref_f.shape[2], my_f.shape[2])
    print("feature frames", ref_f.shape[2], my_f.shape[2], "max abs diff", float((ref_f[:, :, :t] - my_f[:, :, :t]).abs().max()))
    logp = checker(audio, length)
vocab = model.tokenizer.vocab if hasattr(model.tokenizer, "vocab") else [model.tokenizer.ids_to_tokens([i])[0] for i in range(model.tokenizer.vocab_size)]
blank = logp.shape[-1] - 1
ids = logp[0].argmax(-1).tolist(); col = [a for i, a in enumerate(ids) if a != blank and (i == 0 or a != ids[i - 1])]
mine = model.tokenizer.ids_to_text(col)
nemo_text = model.transcribe([wav_path], verbose=False)[0]
nemo_text = nemo_text.text if hasattr(nemo_text, "text") else nemo_text
print("NeMo :", nemo_text); print("ours :", mine)

onnx_path = os.path.join(out, "fastconformer_ar_ctc_fp32.onnx")
torch.onnx.export(checker, (audio, length), onnx_path, input_names=["audio_signal", "length"], output_names=["logprobs"],
                  dynamic_axes={"audio_signal": {0: "B", 1: "N"}, "length": {0: "B"}, "logprobs": {0: "B", 1: "T"}}, opset_version=17, dynamo=False)
json.dump({"vocab": {str(i): model.tokenizer.ids_to_tokens([i])[0] for i in range(model.tokenizer.vocab_size)}, "blank": blank}, open(os.path.join(out, "vocab.json"), "w"), ensure_ascii=False)
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
q_path = os.path.join(out, "fastconformer_ar_ctc_int8.onnx")
quantize_dynamic(onnx_path, q_path, weight_type=QuantType.QInt8)
for path in (onnx_path, q_path):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    t0 = time.time(); lp = s.run(None, {"audio_signal": wav[None, :], "length": np.array([len(wav)], dtype=np.int64)})[0]; dt = time.time() - t0
    ids = lp[0].argmax(-1).tolist(); col = [a for i, a in enumerate(ids) if a != blank and (i == 0 or a != ids[i - 1])]
    print(os.path.basename(path), f"{os.path.getsize(path)/1e6:.1f} MB", f"{dt:.2f}s", "|", model.tokenizer.ids_to_text(col))
