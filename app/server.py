"""로컬 비교 서버: wav 업로드 → SynthSOD / Mega 53 분리 → 브라우저에서 스템 비교."""
import gc
import io
import json
import queue
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
STATIC = Path(__file__).parent / "static"
RUNS.mkdir(exist_ok=True)

# 비교 매트릭스 행: (라벨, SynthSOD 파일명, Mega 53 스템명)
INSTRUMENT_ROWS = [
    ("바이올린", "Violin", "violin"),
    ("비올라", "Viola", "viola"),
    ("첼로", "Cello", "cello"),
    ("더블베이스", "Bass", "double-bass"),
    ("플루트", "Flute", "flute"),
    ("오보에", "Oboe", "oboe"),
    ("클라리넷", "Clarinet", "clarinet"),
    ("바순", "Bassoon", "bassoon"),
    ("호른", "Horn", "french-horn"),
    ("트럼펫", "Trumpet", "trumpet"),
    ("트롬본", "Trombone", "trombone"),
    ("튜바", "Tuba", "tuba"),
    ("하프", "Harp", "harp"),
    ("팀파니", "Timpani", "timpani"),
    ("무음정 타악", "untunedpercussion", "percussion"),
    ("나머지(residual)", "residual", None),
]
# 악기군: SynthSOD 개별 스템 합산 vs Mega 53 악기군 스템
FAMILY_ROWS = [
    ("현악", "family_strings", "bowed_strings", ["Violin", "Viola", "Cello", "Bass"]),
    ("목관", "family_woodwinds", "woodwind", ["Flute", "Oboe", "Clarinet", "Bassoon"]),
    ("금관", "family_brass", "brass", ["Horn", "Trumpet", "Trombone", "Tuba"]),
    ("타악(하프 포함)", "family_percussion", None, ["Harp", "Timpani", "untunedpercussion"]),
]

app = FastAPI(title="OrchestraMSS")
jobs: dict[str, dict] = {}
job_queue: "queue.Queue[str]" = queue.Queue()


def save_meta(job):
    (RUNS / job["id"] / "meta.json").write_text(json.dumps(job, ensure_ascii=False, indent=1))


def build_rows(job):
    d = RUNS / job["id"]
    has = lambda model, name: name is not None and (d / model / f"{name}.wav").exists()
    rows, used = [], set()
    for label, s, m in INSTRUMENT_ROWS:
        if has("synthsod", s) or has("mega53", m):
            rows.append({"group": "악기", "label": label,
                         "synthsod": f"synthsod/{s}.wav" if has("synthsod", s) else None,
                         "mega53": f"mega53/{m}.wav" if has("mega53", m) else None})
            used.add(m)
    for label, s, m, _ in FAMILY_ROWS:
        if has("synthsod", s) or has("mega53", m):
            rows.append({"group": "악기군", "label": label,
                         "synthsod": f"synthsod/{s}.wav" if has("synthsod", s) else None,
                         "mega53": f"mega53/{m}.wav" if has("mega53", m) else None})
            used.add(m)
    if (d / "mega53").exists():
        for f in sorted((d / "mega53").glob("*.wav")):
            if f.stem not in used:
                rows.append({"group": "Mega 53 전용", "label": f.stem, "synthsod": None,
                             "mega53": f"mega53/{f.name}"})
    return rows


def write_family_sums(out_dir):
    for _, name, _, members in FAMILY_ROWS:
        parts = [sf.read(out_dir / f"{m}.wav", dtype="float32") for m in members]
        sf.write(out_dir / f"{name}.wav", sum(p[0] for p in parts), parts[0][1], subtype="FLOAT")


def free_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


class Cancelled(Exception):
    pass


def run_job(job_id):
    from app.separators import mega53, synthsod

    job = jobs[job_id]
    d = RUNS / job_id
    job["status"] = "running"
    for model_name in job["models"]:
        if job.get("cancel"):
            raise Cancelled
        p = job["progress"][model_name]
        p.update(status="running", stage="모델 로딩", started=time.time())
        save_meta(job)

        def progress(i, n, p=p):
            if job.get("cancel"):  # 청크(모델) 단위로 취소 확인
                raise Cancelled
            p.update(stage=f"분리 중 {i}/{n}", done=i, total=n)

        try:
            if model_name == "synthsod":
                p["stage"] = "모델 로딩 + 분리"
                synthsod.run(d / "input.wav", d / "synthsod", job["synthsod_variant"], progress)
                write_family_sums(d / "synthsod")
            else:
                mc = mega53.load_model()
                p["stage"] = "분리 중"
                mega53.run(d / "input.wav", d / "mega53", job["mega_stems"], progress, mc,
                           int(job.get("mega_chunk", 10) * mega53.SAMPLE_RATE))
                del mc
            p.update(status="done", stage="완료", elapsed=round(time.time() - p["started"], 1))
        except Cancelled:
            raise
        except Exception as e:
            traceback.print_exc()
            p.update(status="error", stage=f"오류: {e}", elapsed=round(time.time() - p["started"], 1))
        free_memory()
        job["rows"] = build_rows(job)
        save_meta(job)
    if job.get("cancel"):
        raise Cancelled
    job["status"] = "error" if any(p["status"] == "error" for p in job["progress"].values()) else "done"
    save_meta(job)


def worker():
    while True:
        job_id = job_queue.get()
        try:
            run_job(job_id)
        except Cancelled:
            free_memory()
            shutil.rmtree(RUNS / job_id, ignore_errors=True)
            jobs.pop(job_id, None)
            print(f"작업 {job_id} 취소·삭제됨")
        except Exception as e:
            traceback.print_exc()
            job = jobs[job_id]
            job["status"] = "error"
            for p in job["progress"].values():
                if p["status"] in ("queued", "running"):
                    p.update(status="error", stage=f"오류: {e}")
            save_meta(job)
        finally:
            job_queue.task_done()


def load_history():
    for meta in RUNS.glob("*/meta.json"):
        job = json.loads(meta.read_text())
        if job.get("cancel"):  # 취소 처리 중 서버가 꺼진 작업
            shutil.rmtree(meta.parent, ignore_errors=True)
            continue
        if job["status"] in ("queued", "running"):  # 서버가 중간에 꺼진 작업
            job["status"] = "error"
            for p in job["progress"].values():
                if p["status"] in ("queued", "running"):
                    p.update(status="error", stage="서버 재시작으로 중단됨")
        jobs[job["id"]] = job


load_history()
threading.Thread(target=worker, daemon=True).start()


@app.get("/api/stems")
def stems():
    from app.separators import mega53
    return {"instruments": mega53.instruments(), "presets": mega53.PRESETS}


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    models: str = Form("synthsod,mega53"),
    synthsod_variant: str = Form("synthsod"),
    mega_stems: str = Form(""),
    start: float = Form(0.0),
    end: float = Form(0.0),
    mega_chunk: float = Form(10.0),
):
    model_list = [m for m in models.split(",") if m in ("synthsod", "mega53")]
    if not model_list:
        raise HTTPException(400, "모델을 하나 이상 선택하세요")
    try:
        audio, sr = sf.read(io.BytesIO(await file.read()), dtype="float32", always_2d=True)
    except Exception as e:
        raise HTTPException(400, f"오디오를 읽을 수 없습니다: {e}")
    s = max(0, int(start * sr))
    e = int(end * sr) if end > start else len(audio)
    audio = audio[s:e]
    if len(audio) < sr:
        raise HTTPException(400, "구간이 1초보다 짧습니다")

    job_id = datetime.now().strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    d = RUNS / job_id
    d.mkdir()
    sf.write(d / "input.wav", audio, sr, subtype="FLOAT")
    job = {
        "id": job_id,
        "name": file.filename,
        "created": datetime.now().isoformat(timespec="seconds"),
        "start": round(s / sr, 2),
        "duration": round(len(audio) / sr, 2),
        "models": model_list,
        "synthsod_variant": synthsod_variant,
        "mega_stems": [x for x in mega_stems.split(",") if x] or None,
        "mega_chunk": 5.0 if mega_chunk <= 5 else 10.0,
        "status": "queued",
        "progress": {m: {"status": "queued", "stage": "대기", "done": 0, "total": 0} for m in model_list},
        "rows": [],
    }
    jobs[job_id] = job
    save_meta(job)
    job_queue.put(job_id)
    return job


@app.get("/api/jobs")
def list_jobs():
    return sorted(
        ({k: j[k] for k in ("id", "name", "created", "duration", "status", "models")}
         for j in jobs.values() if not j.get("cancel")),
        key=lambda j: j["id"], reverse=True)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404)
    job = jobs[job_id]
    for p in job["progress"].values():
        if p["status"] == "running":
            p["elapsed"] = round(time.time() - p["started"], 1)
    return job


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404)
    if job["status"] in ("queued", "running"):
        # 워커가 다음 청크에서 멈추고 메모리 정리 후 폴더를 지운다
        job["cancel"] = True
        return {"ok": True, "cancelling": True}
    shutil.rmtree(RUNS / job_id)
    del jobs[job_id]
    return {"ok": True}


@app.get("/api/jobs/{job_id}/zip")
def job_zip(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] in ("queued", "running"):
        raise HTTPException(404)
    d = RUNS / job_id
    zpath = d / "results.zip"
    if not zpath.exists():
        # wav 는 deflate 로 거의 안 줄고 느리기만 해서 무압축으로 묶는다
        with zipfile.ZipFile(zpath.with_suffix(".part"), "w", zipfile.ZIP_STORED) as z:
            for f in sorted(d.rglob("*.wav")):
                z.write(f, f.relative_to(d))
        zpath.with_suffix(".part").rename(zpath)
    return FileResponse(zpath, media_type="application/zip", filename=f"{job_id}.zip")


app.mount("/runs", StaticFiles(directory=RUNS), name="runs")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
