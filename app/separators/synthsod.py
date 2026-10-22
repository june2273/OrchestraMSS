# 출처: https://github.com/repertorium/SynthSOD-Baseline
#       eval.py 의 separate()·istft() @ 4de6c7ad63974b08f16cddcda1dbac1422a40f1b
# 원 저작자: SynthSOD-Baseline 저자들 (J. Garcia-Martinez et al.), AGPL-3.0
# 바꾼 것: 추론에 필요한 부분만 남김(museval·datasets 제거), 위너 필터를 프레임 블록 단위로 나눔,
#          출력 길이를 입력에 맞춤. 바꾼 이 파일도 원본과 같은 AGPL-3.0 을 따른다(저장소 루트 LICENSE).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
"""SynthSOD 베이스라인 (X-UMX 4개: 현악/목관/금관/타악) 추론.

third_party/SynthSOD-Baseline/eval.py 의 separate() 를 옮겨 왔다. eval.py 를 그대로 import 하면
museval·datasets 까지 따라오므로 추론에 필요한 부분만 가져오고, 위너 필터는 긴 곡에서 메모리가
폭증하지 않도록 프레임 블록 단위로 나눠 돌린다.
"""
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import torch

import norbert
from asteroid.complex_nn import torch_complex_from_magphase

ROOT = Path(__file__).resolve().parents[2]


def _load_synthsod_utils():
    # MSST 에도 최상위 utils 패키지가 있어서 `import utils` 는 충돌한다 → 파일 경로로 고유 이름을 붙여 로드
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "synthsod_utils", ROOT / "third_party" / "SynthSOD-Baseline" / "utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


load_model = _load_synthsod_utils().load_model

MODEL_DIR = ROOT / "models" / "synthsod"
FAMILIES = ["strings", "woodwinds", "brass", "percussion"]
VARIANTS = ["synthsod", "ensembleset"]
SAMPLE_RATE = 44100
WIENER_BLOCK_FRAMES = 2000  # 약 46초 (hop 1024 @ 44.1kHz)


def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    # eval.py 와 동일
    t, audio = scipy.signal.istft(
        X / (n_fft / 2), rate, nperseg=n_fft, noverlap=n_fft - n_hopsize, boundary=True
    )
    return audio


# 체크포인트 안의 학습 통계가 numpy 스칼라로 저장돼 있어, torch>=2.6 의 weights_only 로드에 허용 목록이 필요하다
SAFE_GLOBALS = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype, torch.torch_version.TorchVersion,
                *(type(np.dtype(t)) for t in (np.float32, np.float64, np.int32, np.int64, np.bool_))]


def load_models(variant="synthsod"):
    assert variant in VARIANTS
    with torch.serialization.safe_globals(SAFE_GLOBALS):
        return [load_model(str(MODEL_DIR / f"{variant}_{fam}.pth"), "cpu") for fam in FAMILIES]


@torch.no_grad()
def separate(audio, models, niter=1, max_frames=1024, progress=None):
    """audio: (nb_samples, 1) 모노 44.1kHz. 반환: {악기명: (nb_samples, 1)} + residual."""
    targets = [s for m in models for s in m.sources]
    audio_torch = torch.tensor(audio.T[None, ...]).float()
    spec_mag, spec_phase = models[0].encoder(audio_torch)

    masked_mixtures = []
    for k, model in enumerate(models):
        segs = []
        for i in range(0, spec_mag.shape[0], max_frames):
            seg = spec_mag[i:i + max_frames]
            segs.append(model.apply_masks(seg, model.forward_masker(seg.clone())))
        masked_mixtures.append(torch.concatenate(segs, axis=1))
        if progress:
            progress(k + 1, len(models) + 1)
    masked_mixtures = torch.concatenate(masked_mixtures, axis=0)

    V = np.transpose(masked_mixtures[:, :, 0].numpy(), (1, 3, 2, 0))  # frames, bins, ch, sources
    X = torch_complex_from_magphase(spec_mag.permute(1, 2, 3, 0), spec_phase)
    X = X.numpy()[0].transpose(2, 1, 0)  # frames, bins, ch

    V = norbert.residual_model(V, X, 1)
    names = targets + ["residual"]

    Y = np.zeros(V.shape, dtype=np.complex64)
    for i in range(0, X.shape[0], WIENER_BLOCK_FRAMES):
        sl = slice(i, i + WIENER_BLOCK_FRAMES)
        Y[sl] = norbert.wiener(V[sl], X[sl].astype(np.complex128), niter)

    m0 = models[0]
    out = {}
    for j, name in enumerate(names):
        est = istft(Y[..., j].T, rate=m0.sample_rate, n_fft=m0.in_chan, n_hopsize=m0.n_hop).T
        est = est[: audio.shape[0]]
        out[name] = np.pad(est, ((0, audio.shape[0] - est.shape[0]), (0, 0)))  # STFT 끝 프레임 손실 보정
    if progress:
        progress(len(models) + 1, len(models) + 1)
    return out


def load_mono(path):
    import librosa
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio[:, None]


def run(in_path, out_dir, variant="synthsod", progress=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = load_models(variant)
    audio = load_mono(in_path)
    stems = separate(audio, models, progress=progress)
    for name, est in stems.items():
        sf.write(out_dir / f"{name}.wav", est, SAMPLE_RATE, subtype="FLOAT")
    return list(stems)


if __name__ == "__main__":
    import argparse
    import time

    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("out_dir")
    p.add_argument("--variant", default="synthsod", choices=VARIANTS)
    a = p.parse_args()
    t = time.time()
    names = run(a.input, a.out_dir, a.variant, progress=lambda i, n: print(f"{i}/{n}"))
    print(f"{len(names)} stems, {time.time() - t:.1f}s:", names)
