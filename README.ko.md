[English](README.md)

# OrchestraMSS

같은 오케스트라 녹음을 **SynthSOD 베이스라인**(X-UMX)과 **MVSep Mega 53 Stems**(BS-RoFormer)로 분리하고, 브라우저에서 악기별로 나란히 듣고 A/B로 비교하는 로컬 웹앱입니다.

## 실행

```bash
./setup.sh
```

```bash
.venv/bin/python main.py
```

브라우저에서 http://127.0.0.1:8000 을 엽니다. PyCharm에서는 `main.py`를 그대로 실행해도 됩니다.

## 디스크 용량 (실측, Mac mini M4)

| 항목 | 크기 |
|---|---|
| Python 환경 `.venv` (torch, asteroid, librosa, fastapi 등) | 1.3 GB |
| SynthSOD 체크포인트 8개 `models/synthsod` (SynthSOD 학습본 4 + EnsembleSet 학습본 4) | 811 MB |
| Mega 53 체크포인트 `models/mega53` | 1.3 GB |
| 코드 `third_party` (SynthSOD-Baseline, MSST v1.0.21) | 2.4 MB |
| **합계** | **약 3.4 GB** |

EnsembleSet 학습본을 쓰지 않을 거라면 `models/synthsod/ensembleset_*.pth`를 지워도 됩니다(-405MB). 분리 결과는 `runs/<작업ID>/`에 쌓입니다. 20초 입력 1건이 float32 wav로 약 200MB입니다.

## 속도와 메모리 (실측, M4 16GB)

| 모델 | 입력 | 소요 | 피크 메모리 |
|---|---|---|---|
| SynthSOD (CPU) | 30초 | 약 4초 | 3.4 GB |
| Mega 53 (MPS, 10초 청크, 스템 20개) | 30초 | 62초 | 9.5 GB |
| Mega 53 (MPS, 10초 청크, 스템 20개) | 150초 | 164초 | 10.0 GB |
| Mega 53 (MPS, 5초 청크) | 30초 | 34초 | 8.3 GB |

## 원본 코드에서 바꾼 점

- **Mega 53 청크 20초 → 10초**: yaml의 `inference.chunk_size`(20초)로 돌리면 MPS에서 시간축 어텐션 행렬이 수 GB가 되어 스왑에 빠지고 사실상 멈춥니다. 그래서 학습 청크 크기(10초)로 추론합니다. UI에서 5초(더 빠름)도 고를 수 있습니다.
- **Mega 53은 선택한 스템만 계산·누적**: `active_stem_ids`로 선택한 마스크 헤드만 돌리고, 결과 버퍼도 그만큼만 잡습니다. MSST 기본 `demix()`는 53개 스템 전부를 곡 길이만큼 두 벌 잡기 때문에 5분 곡이면 10GB를 넘습니다.
- **SynthSOD**: `eval.py`의 `separate()`를 옮겨 왔습니다(museval 불필요). 위너 필터는 약 46초 블록 단위로 나눠 돌리고, 출력 길이는 입력과 같게 맞춥니다. 체크포인트는 `weights_only` 로드에 필요한 numpy 타입만 허용 목록에 넣어 읽습니다. 입력은 모노로 다운믹스합니다(학습 설정 `nb_channels: 1`).

## 화면 구성

1. 입력: 드래그앤드롭, 파형 표시, 구간 선택(기본 앞 30초), 구간 미리듣기
2. 모델 설정: SynthSOD 학습 세트 선택, Mega 53 청크 크기와 스템 프리셋/개별 선택, 예상 소요 시간
3. 진행 상황: 모델별 진행 바. 작업은 한 번에 하나씩 순서대로 처리합니다(메모리 보호).
4. 결과 비교: 악기 × [SynthSOD | Mega 53] 매트릭스. 파형은 dB 스케일이고 RMS dB를 함께 표시합니다. 파형을 클릭하면 그 위치부터 재생합니다. A/B 동시 재생은 버튼 또는 `A`/`B` 키로 전환하고, `Space`로 정지합니다. 악기군 행(SynthSOD 개별 스템 합산 vs Mega 53 악기군 스템)과 zip 다운로드가 있습니다.

## License

이 저장소는 **AGPL-3.0**입니다([LICENSE](LICENSE)). `app/separators/synthsod.py`가 AGPL-3.0인 [SynthSOD-Baseline](https://github.com/repertorium/SynthSOD-Baseline)의 `eval.py`를 고쳐 옮겨 온 코드라서 저장소 전체를 같은 라이선스로 둡니다.

- `app/separators/mega53.py`는 [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)(MIT, © 2024 Roman Solovyev)의 `demix()`를 고친 코드입니다. 원본 MIT 고지를 파일 머리에 그대로 두었습니다.
- 외부 코드 원본(`third_party/`)과 모델 가중치(`models/`)는 저장소에 넣지 않습니다. `setup.sh`가 원 저장소와 릴리스에서 받아 오며, 각각 원래 라이선스를 따릅니다.
- AGPL-3.0이라 이 앱을 고쳐 네트워크로 서비스하면(`share.sh`의 공개 링크 포함) 이용자에게 고친 소스를 제공해야 합니다.

## References

- J. Garcia-Martinez, D. Diaz-Guerra, A. Politis, T. Virtanen, J. J. Carabias-Orti, P. Vera-Candeas, "SynthSOD: Developing an Heterogeneous Dataset for Orchestra Music Source Separation," *IEEE Open Journal of Signal Processing* 6, pp. 129–137, 2025. arXiv:2409.10995 (dataset and X-UMX baseline, `third_party/SynthSOD-Baseline`)
- R. Sawata et al., "All for One and One for All: Improving Music Separation by Bridging Networks," *ICASSP*, 2021. (X-UMX)
- W.-T. Lu et al., "Music Source Separation with Band-Split RoPE Transformer," *ICASSP*, 2024. (BS-RoFormer)
- R. Solovyev, A. Stempkovskiy, T. Habruseva, "Benchmarks and Leaderboards for Sound Demixing Tasks," arXiv:2305.07489, 2023. (Music-Source-Separation-Training, `third_party/MSST`)
- MVSep "Mega 53 Stems" model. https://mvsep.com
