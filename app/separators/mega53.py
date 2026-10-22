# 출처: https://github.com/ZFTurbo/Music-Source-Separation-Training
#       utils/model_utils.py 의 demix() @ v1.0.21
# 바꾼 것: 선택한 스템의 마스크 헤드만 계산하고 결과 버퍼도 그만큼만 잡는다. 원본 부분의 고지는 아래와 같다.
#
# MIT License
#
# Copyright (c) 2024 Roman Solovyev (ZFTurbo)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""MVSep Mega 53 Stems (BS-RoFormer) 추론.

MSST 의 utils.model_utils.demix() 를 기반으로 하되, 결과 누적 텐서를 선택한 스템만큼만 잡는다.
원본은 53개 스템 전부를 곡 길이만큼 float32 로 두 벌(result, counter) 잡아서 5분 곡이면 10GB를 넘는다.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "MSST"))

from utils.model_utils import _getWindowingArray  # noqa: E402
from utils.settings import get_model_from_config  # noqa: E402

MODEL_DIR = ROOT / "models" / "mega53"
CONFIG_PATH = MODEL_DIR / "config.yaml"
CKPT_PATH = MODEL_DIR / "model.ckpt"
SAMPLE_RATE = 44100
# yaml 의 inference.chunk_size(20초)는 MPS 에서 시간축 어텐션 행렬이 수 GB가 되어 16GB Mac 이 스왑에 빠진다.
# 학습 청크 크기(audio.chunk_size, 10초)로 추론한다.
CHUNK_SIZE = 441000

PRESETS = {
    "orchestra": ["violin", "viola", "cello", "double-bass", "flute", "oboe", "clarinet", "bassoon",
                  "french-horn", "trumpet", "trombone", "tuba", "timpani", "harp", "percussion"],
    "families": ["strings", "bowed_strings", "woodwind", "wind", "brass", "percussion"],
}
PRESETS["default"] = PRESETS["orchestra"] + [s for s in PRESETS["families"] if s not in PRESETS["orchestra"]]


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def instruments():
    import yaml
    with open(CONFIG_PATH) as f:
        return yaml.load(f, Loader=yaml.FullLoader)["training"]["instruments"]


def load_model(dev=None):
    model, config = get_model_from_config("bs_roformer", str(CONFIG_PATH))
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt)
    del ckpt
    model.eval().to(dev or device())
    return model, config


@torch.inference_mode()
def demix_subset(config, model, mix, stems, dev, progress=None, chunk_size=CHUNK_SIZE):
    """mix: (channels, time) float32. stems: 남길 스템 이름 목록. 반환 {이름: (time, channels)}."""
    all_names = list(config.training.instruments)
    idx = [all_names.index(s) for s in stems]

    mix = torch.tensor(mix, dtype=torch.float32)
    num_overlap = config.inference.num_overlap
    fade_size = chunk_size // 10
    step = chunk_size // num_overlap
    border = chunk_size - step
    length_init = mix.shape[-1]
    windowing_array = _getWindowingArray(chunk_size, fade_size)
    if length_init > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode="reflect")

    result = torch.zeros((len(idx),) + mix.shape, dtype=torch.float32)
    counter = torch.zeros(mix.shape[-1], dtype=torch.float32)  # 창 가중치는 스템·채널 공통
    n_chunks = (mix.shape[1] + step - 1) // step

    i = 0
    k = 0
    while i < mix.shape[1]:
        part = mix[:, i:i + chunk_size]
        chunk_len = part.shape[-1]
        pad_mode = "reflect" if chunk_len > chunk_size // 2 else "constant"
        part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode=pad_mode, value=0)
        x = model(part[None].to(dev), active_stem_ids=idx)[0].cpu()  # 선택한 스템의 마스크 헤드만 계산
        window = windowing_array.clone()
        if i == 0:
            window[:fade_size] = 1
        elif i + step >= mix.shape[1]:
            window[-fade_size:] = 1
        result[..., i:i + chunk_len] += x[..., :chunk_len] * window[:chunk_len]
        counter[i:i + chunk_len] += window[:chunk_len]
        i += step
        k += 1
        if progress:
            progress(k, n_chunks)

    est = (result / counter).numpy()
    np.nan_to_num(est, copy=False, nan=0.0)
    if length_init > 2 * border and border > 0:
        est = est[..., border:-border]
    return {name: est[j].T for j, name in enumerate(stems)}


def load_stereo(path):
    import librosa
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    audio = audio.T
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    audio = audio[:2]
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def run(in_path, out_dir, stems=None, progress=None, model_and_config=None, chunk_size=CHUNK_SIZE):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = device()
    model, config = model_and_config or load_model(dev)
    stems = stems or PRESETS["default"]
    mix = load_stereo(in_path)
    out = demix_subset(config, model, mix, stems, dev, progress, chunk_size)
    for name, est in out.items():
        sf.write(out_dir / f"{name}.wav", est, SAMPLE_RATE, subtype="FLOAT")
    return list(out)


if __name__ == "__main__":
    import argparse
    import time

    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("out_dir")
    p.add_argument("--stems", nargs="*")
    a = p.parse_args()
    t = time.time()
    names = run(a.input, a.out_dir, a.stems, progress=lambda i, n: print(f"chunk {i}/{n}", flush=True))
    print(f"{len(names)} stems, {time.time() - t:.1f}s")
