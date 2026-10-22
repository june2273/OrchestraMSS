#!/bin/bash
# 의존성 설치 + 코드/가중치 다운로드. 재실행해도 이미 받은 것은 건너뜀.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -x "$PY" ] || python3.11 -m venv .venv

$PY -m pip install -U pip
$PY -m pip install -r requirements.txt

mkdir -p third_party models/synthsod models/mega53

# SynthSOD 베이스라인 코드
[ -d third_party/SynthSOD-Baseline ] || git clone --depth 1 https://github.com/repertorium/SynthSOD-Baseline third_party/SynthSOD-Baseline

# MSST 코드 (v1.0.21)
if [ ! -d third_party/MSST ]; then
  curl -L -o third_party/msst.zip https://github.com/ZFTurbo/Music-Source-Separation-Training/archive/refs/tags/v1.0.21.zip
  unzip -q third_party/msst.zip -d third_party && mv third_party/Music-Source-Separation-Training-1.0.21 third_party/MSST
  rm third_party/msst.zip
fi

# SynthSOD 사전학습 모델 (zip 770MB → 모델 8개)
if [ ! -f models/synthsod/synthsod_strings.pth ]; then
  curl -L -o models/synthsod/pretrained.zip https://github.com/repertorium/SynthSOD-Baseline/releases/download/pretrained-models/SynthSOD-Baseline-pretrained-models.zip
  unzip -q models/synthsod/pretrained.zip -d models/synthsod/tmp
  for set in synthsod ensembleset; do
    for fam in strings woodwinds brass percussion; do
      mv models/synthsod/tmp/exp/train_xumx_${set}_${fam}/best_model.pth models/synthsod/${set}_${fam}.pth
    done
  done
  rm -rf models/synthsod/tmp models/synthsod/pretrained.zip
fi

# MVSep Mega 53 Stems (1.37GB)
R=https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.21
[ -f models/mega53/config.yaml ] || curl -L -o models/mega53/config.yaml $R/mvsep_mega_model_bs_roformer_53_stems.yaml
if [ ! -f models/mega53/model.ckpt ]; then
  curl -L -C - -o models/mega53/model.ckpt.part $R/mvsep_mega_model_bs_roformer_53_stems_v1.ckpt
  mv models/mega53/model.ckpt.part models/mega53/model.ckpt
fi

echo "== 디스크 사용량 =="
du -sh .venv third_party models/synthsod models/mega53
