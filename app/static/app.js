"use strict";
const $ = (s) => document.querySelector(s);
const fmt = (t) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// 실측(M4 16GB): SynthSOD 30초 입력 약 4초 / Mega 53 10초 청크 150초 입력 약 163초, 20초 입력 약 39초 (로딩 약 15초 포함)
const SPEED = { synthsod: 0.13, mega10: 1.1, mega5: 0.65 };

const state = { file: null, buffer: null, stems: null, selected: new Set(), job: null, poll: null, rowsKey: "" };
let audioCtx = null;
const ctx = () => (audioCtx ||= new (window.AudioContext || window.webkitAudioContext)());

/* ---------- 파형 ---------- */
function peaksOf(buffer, width, t0 = 0, t1 = null) {
  const n = buffer.numberOfChannels, sr = buffer.sampleRate;
  const off = Math.floor(t0 * sr), len = Math.min(buffer.length, t1 == null ? buffer.length : Math.floor(t1 * sr)) - off;
  const step = Math.max(1, Math.floor(len / width));
  const chans = Array.from({ length: n }, (_, c) => buffer.getChannelData(c).subarray(off, off + len));
  const peaks = new Float32Array(width);
  let sq = 0;
  for (let i = 0; i < width; i++) {
    let m = 0;
    for (let j = i * step, e = Math.min(len, j + step); j < e; j += 4) {
      for (const ch of chans) { const v = Math.abs(ch[j]); if (v > m) m = v; }
    }
    peaks[i] = m;
  }
  for (const ch of chans) for (let j = 0; j < len; j += 16) sq += ch[j] * ch[j];
  const rms = Math.sqrt(sq / (n * Math.ceil(len / 16)));
  return { peaks, rms, duration: buffer.duration };
}

// 스템은 대부분 조용해서 선형 스케일로는 평평하게 보인다 → -60dB~0dB 로그 스케일로 그린다
const dbScale = (v) => Math.max(0, (20 * Math.log10(v + 1e-9) + 60) / 60);

function drawWave(canvas, peaks, color) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  g.fillStyle = color;
  const mid = h / 2;
  for (let x = 0; x < w; x++) {
    const v = dbScale(peaks[Math.floor((x / w) * peaks.length)]);
    const bh = Math.max(0.5, Math.min(1, v) * (h / 2 - 1));
    g.fillRect(x, mid - bh, 1, bh * 2);
  }
}

const waveCache = new Map();
let decodeChain = Promise.resolve();
function loadWave(url) {
  if (!waveCache.has(url)) {
    // 디코딩은 한 번에 하나씩 (메모리)
    const p = decodeChain.then(async () => {
      const buf = await (await fetch(url)).arrayBuffer();
      return peaksOf(await ctx().decodeAudioData(buf), 2000);
    });
    decodeChain = p.catch(() => {});
    waveCache.set(url, p);
  }
  return waveCache.get(url);
}

/* ---------- 재생 ---------- */
const player = { audios: [], urls: [], rowEl: null, btn: null, abBtn: null, which: "a", endAt: null, rangeStart: 0,
  previewPeaks: null, label: "", raf: 0, waveTok: 0 };

function stopAll() {
  player.audios.forEach((a) => { a.pause(); a.src = ""; });
  player.audios = [];
  player.rowEl?.classList.remove("playing");
  if (player.btn) player.btn.textContent = "▶";
  player.abBtn?.classList.remove("on-a", "on-b");
  Object.assign(player, { rowEl: null, btn: null, abBtn: null, endAt: null, previewPeaks: null });
  cancelAnimationFrame(player.raf);
  $("#player-bar").hidden = true;
  $("#player-toggle").textContent = "❚❚";
}

function togglePause() {
  const [a, b] = player.audios;
  if (!a) return;
  if (a.paused) {
    if (b) b.currentTime = a.currentTime;
    player.audios.forEach((x) => x.play());
  } else {
    player.audios.forEach((x) => x.pause());
  }
  $("#player-toggle").textContent = a.paused ? "▶" : "❚❚";
}

function seekBy(delta) {
  const a = player.audios[0];
  if (!a) return;
  const end = (player.endAt ?? a.duration) - 0.05;
  const t = Math.min(Math.max(a.currentTime + delta, player.rangeStart), end);
  player.audios.forEach((x) => { x.currentTime = t; });
}

// 재생바 클릭/드래그: 그 시점으로 이동해 바로 재생
function seekTo(frac) {
  const a = player.audios[0];
  if (!a) return;
  const end = player.endAt ?? a.duration;
  if (!isFinite(end)) return;
  const t = Math.min(player.rangeStart + frac * (end - player.rangeStart), end - 0.05);
  player.audios.forEach((x) => { x.currentTime = t; });
  if (a.paused) {
    player.audios.forEach((x) => x.play());
    $("#player-toggle").textContent = "❚❚";
  }
}

// 재생바 위 파형: 현재 들리는 트랙의 표시 구간(전체 또는 미리듣기 구간)
async function drawPlayerWave() {
  const tok = ++player.waveTok;
  const cv = $("#player-wave"), cv2 = $("#player-wave-played");
  for (const c of [cv, cv2]) c.getContext("2d").clearRect(0, 0, c.width, c.height);
  if (!player.audios.length) return;
  let peaks = player.previewPeaks;
  if (!peaks) {
    let w;
    try { w = await loadWave(player.urls[player.which === "b" ? 1 : 0]); } catch { return; }
    if (tok !== player.waveTok) return;
    const n = w.peaks.length, end = player.endAt ?? w.duration;
    peaks = w.peaks.slice(Math.floor((player.rangeStart / w.duration) * n), Math.ceil((end / w.duration) * n));
  }
  $("#player-played").style.setProperty("--pb-w", `${$("#player-track").clientWidth}px`);
  drawWave(cv, peaks, css("--muted"));
  drawWave(cv2, peaks, css("--accent"));
}

{
  const track = $("#player-track");
  const fracAt = (e) => { const r = track.getBoundingClientRect(); return Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)); };
  const move = (e) => seekTo(fracAt(e));
  track.addEventListener("pointerdown", (e) => {
    track.setPointerCapture(e.pointerId);
    seekTo(fracAt(e));
    track.addEventListener("pointermove", move);
  });
  for (const ev of ["pointerup", "pointercancel"]) track.addEventListener(ev, () => track.removeEventListener("pointermove", move));
}

function tick() {
  const a = player.audios[0];
  if (!a) return;
  const end = player.endAt ?? a.duration;
  if (player.endAt != null && a.currentTime >= player.endAt) return stopAll();
  const start = player.rangeStart;
  const pct = `${Math.min(100, Math.max(0, (100 * (a.currentTime - start)) / Math.max(0.01, end - start)))}%`;
  $("#player-pos").style.width = pct;
  $("#player-played").style.width = pct;
  $("#player-head").style.left = pct;
  $("#player-time").textContent = `${fmt(a.currentTime)} / ${fmt(end || 0)}`;
  player.raf = requestAnimationFrame(tick);
}

function startPlayer(urls, { label, rowEl, btn, abBtn, at = 0, endAt = null, rangeStart = 0, previewPeaks = null }) {
  stopAll();
  const audios = urls.map((u) => { const a = new Audio(u); a.preload = "auto"; return a; });
  Object.assign(player, { audios, urls, rowEl, btn, abBtn, endAt, rangeStart, previewPeaks, label, which: "a" });
  $("#player-played").style.width = $("#player-pos").style.width = "0%";
  $("#player-head").style.left = "0%";
  rowEl?.classList.add("playing");
  if (btn) btn.textContent = "■";
  $("#player-label").textContent = label;
  $("#player-bar").hidden = false;
  drawPlayerWave();
  audios[0].addEventListener("ended", stopAll);
  if (audios[1]) { audios[1].muted = true; abBtn.classList.add("on-a"); }
  Promise.all(audios.map((a) => new Promise((r) => {
    if (a.readyState >= 3) r(); else a.addEventListener("canplaythrough", r, { once: true });
  }))).then(() => {
    if (player.audios !== audios) return;
    audios.forEach((a) => { a.currentTime = at; });
    return Promise.all(audios.map((a) => a.play()));
  }).then(() => { player.raf = requestAnimationFrame(tick); });
}

function switchAB(which) {
  if (player.audios.length !== 2) return;
  player.which = which ?? (player.which === "a" ? "b" : "a");
  const [a, b] = player.audios;
  if (Math.abs(a.currentTime - b.currentTime) > 0.05) b.currentTime = a.currentTime; // 드리프트 보정
  a.muted = player.which !== "a";
  b.muted = player.which !== "b";
  player.abBtn.classList.toggle("on-a", player.which === "a");
  player.abBtn.classList.toggle("on-b", player.which === "b");
  $("#player-label").textContent = `${player.label} — ${player.which === "a" ? "A: SynthSOD" : "B: Mega 53"}`;
  drawPlayerWave();
}

document.addEventListener("keydown", (e) => {
  if (e.target.matches?.("input, textarea")) return;
  if (e.key === " " && player.audios.length) { e.preventDefault(); togglePause(); }
  if (e.key === "ArrowLeft" && player.audios.length) { e.preventDefault(); seekBy(-3); }
  if (e.key === "ArrowRight" && player.audios.length) { e.preventDefault(); seekBy(3); }
  if (e.key === "a" || e.key === "A") switchAB("a");
  if (e.key === "b" || e.key === "B") switchAB("b");
});
$("#player-stop").onclick = stopAll;
$("#player-toggle").onclick = togglePause;
$("#player-back").onclick = () => seekBy(-3);
$("#player-fwd").onclick = () => seekBy(3);
// 재생바 버튼이 포커스를 가져가면 Space 가 버튼 클릭으로도 처리돼 두 번 토글된다
document.querySelectorAll("#player-bar button").forEach((b) => b.addEventListener("mousedown", (e) => e.preventDefault()));

/* ---------- 1. 입력 ---------- */
const drop = $("#drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });
$("#file").addEventListener("change", (e) => e.target.files[0] && setFile(e.target.files[0]));

async function setFile(file) {
  state.file = file;
  $("#file-name").textContent = file.name;
  $("#file-meta").textContent = "디코딩 중…";
  $("#input-view").hidden = false;
  try {
    state.buffer = await ctx().decodeAudioData(await file.arrayBuffer());
  } catch {
    state.buffer = null;
    $("#file-meta").textContent = "브라우저에서 파형을 읽을 수 없는 형식입니다 (서버에서는 처리 가능할 수 있음)";
    updateRun();
    return;
  }
  const b = state.buffer, d = b.duration;
  $("#file-meta").textContent = `${fmt(d)} · ${b.numberOfChannels}ch · ${(file.size / 1e6).toFixed(1)} MB`;
  for (const id of ["#start", "#end"]) $(id).max = d.toFixed(1);
  $("#start").value = 0;
  $("#end").value = Math.min(30, d).toFixed(1);
  state.inputPeaks = peaksOf(b, 1200).peaks;
  drawInput();
  onRange();
}

function drawInput() {
  if (state.inputPeaks) drawWave($("#input-wave"), state.inputPeaks, css("--muted"));
}

function onRange(e) {
  let s = +$("#start").value, en = +$("#end").value;
  if (en - s < 1) {
    if (e?.target.id === "start") s = Math.max(0, en - 1); else en = Math.min(+$("#end").max, s + 1);
    $("#start").value = s; $("#end").value = en;
  }
  const d = state.buffer?.duration || 1;
  $("#start-out").value = fmt(s);
  $("#end-out").value = fmt(en);
  $("#sel-len").textContent = `${(en - s).toFixed(1)}초`;
  Object.assign($("#sel").style, { left: `${(100 * s) / d}%`, width: `${(100 * (en - s)) / d}%` });
  updateRun();
}
$("#start").addEventListener("input", onRange);
$("#end").addEventListener("input", onRange);
$("#input-wave").addEventListener("click", (e) => {
  if (!state.buffer) return;
  // 클릭한 위치를 가까운 쪽 경계로
  const t = (e.offsetX / e.target.clientWidth) * state.buffer.duration;
  const s = +$("#start").value, en = +$("#end").value;
  (Math.abs(t - s) < Math.abs(t - en) ? $("#start") : $("#end")).value = t.toFixed(1);
  onRange();
});
$("#preview").onclick = () => {
  if (!state.file) return;
  if (player.label === "원본 구간 미리듣기") return stopAll();
  const s = +$("#start").value, e = +$("#end").value;
  startPlayer([URL.createObjectURL(state.file)], {
    label: "원본 구간 미리듣기", at: s, endAt: e, rangeStart: s,
    previewPeaks: state.buffer ? peaksOf(state.buffer, 1500, s, e).peaks : null,
  });
};

/* ---------- 2. 모델 ---------- */
async function loadStems() {
  state.stems = await (await fetch("/api/stems")).json();
  state.stems.presets.all = state.stems.instruments;
  const grid = $("#stem-grid");
  grid.innerHTML = "";
  for (const s of state.stems.instruments) {
    const l = document.createElement("label");
    l.innerHTML = `<input type="checkbox" value="${s}"> ${s}`;
    l.querySelector("input").onchange = (e) => {
      e.target.checked ? state.selected.add(s) : state.selected.delete(s);
      syncStems(null);
    };
    grid.append(l);
  }
  applyPreset("default");
}
function applyPreset(name) {
  state.selected = new Set(state.stems.presets[name]);
  syncStems(name);
}
function syncStems(preset) {
  document.querySelectorAll("#stem-grid input").forEach((i) => { i.checked = state.selected.has(i.value); });
  document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("on", c.dataset.preset === preset));
  $("#stem-count").textContent = state.selected.size;
  updateRun();
}
document.querySelectorAll(".chip").forEach((c) => (c.onclick = () => applyPreset(c.dataset.preset)));
for (const el of document.querySelectorAll("#use-synthsod, #use-mega, input[name=chunk]")) el.addEventListener("change", updateRun);

function chosenModels() {
  const m = [];
  if ($("#use-synthsod").checked) m.push("synthsod");
  if ($("#use-mega").checked && state.selected.size) m.push("mega53");
  return m;
}
function updateRun() {
  const models = chosenModels();
  const busy = state.job && ["queued", "running"].includes(state.job.status);
  $("#run").disabled = !state.file || !models.length || busy;
  if (!state.file) { $("#eta").textContent = "먼저 오디오 파일을 선택하세요"; return; }
  const len = state.buffer ? +$("#end").value - +$("#start").value : null;
  if (!len) { $("#eta").textContent = ""; return; }
  let sec = 0;
  if (models.includes("synthsod")) sec += 3 + SPEED.synthsod * len;
  if (models.includes("mega53")) sec += 15 + (document.querySelector("input[name=chunk]:checked").value === "5" ? SPEED.mega5 : SPEED.mega10) * len;
  $("#eta").textContent = busy ? "이전 작업이 진행 중입니다" : `예상 소요 약 ${sec < 90 ? Math.round(sec) + "초" : (sec / 60).toFixed(1) + "분"} (M4 기준)`;
}

$("#run").onclick = async () => {
  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("models", chosenModels().join(","));
  fd.append("synthsod_variant", document.querySelector("input[name=variant]:checked").value);
  fd.append("mega_stems", [...state.selected].join(","));
  fd.append("mega_chunk", document.querySelector("input[name=chunk]:checked").value);
  if (state.buffer) { fd.append("start", $("#start").value); fd.append("end", $("#end").value); }
  $("#run").disabled = true;
  $("#eta").textContent = "업로드 중…";
  const r = await fetch("/api/jobs", { method: "POST", body: fd });
  if (!r.ok) { $("#eta").textContent = (await r.json()).detail || "실패"; updateRun(); return; }
  await openJob((await r.json()).id);
  loadHistory();
  $("#step-progress").scrollIntoView({ behavior: "smooth", block: "start" });
};

/* ---------- 3/4. 작업 ---------- */
const MODEL_NAMES = { synthsod: "SynthSOD 베이스라인", mega53: "Mega 53 Stems" };

async function openJob(id) {
  clearTimeout(state.poll);
  stopAll();
  state.rowsKey = "";
  const job = await (await fetch(`/api/jobs/${id}`)).json();
  state.job = job;
  document.querySelectorAll("#history-list li").forEach((li) => li.classList.toggle("active", li.dataset.id === id));
  renderJob();
}

async function pollJob() {
  if (!state.job) return;
  const id = state.job.id;
  const r = await fetch(`/api/jobs/${id}`);
  if (!r.ok || state.job?.id !== id) return; // 그 사이 삭제·전환된 작업
  state.job = await r.json();
  renderJob();
}

function renderJob() {
  const job = state.job;
  $("#step-progress").hidden = false;
  $("#job-id").textContent = `${job.name} · ${job.id}`;
  $("#cancel").hidden = !["queued", "running"].includes(job.status);
  $("#progress-list").innerHTML = job.models.map((m) => {
    const p = job.progress[m];
    const pct = p.status === "done" ? 100 : p.total ? (100 * p.done) / p.total : p.status === "running" ? 5 : 0;
    const t = p.elapsed != null ? ` · ${p.elapsed.toFixed(0)}초` : "";
    return `<div class="prog ${p.status}"><b>${MODEL_NAMES[m]}</b><div class="bar"><div style="width:${pct}%"></div></div><span class="stage">${p.stage}${t}</span></div>`;
  }).join("");

  const key = JSON.stringify(job.rows);
  if (job.rows.length && key !== state.rowsKey) { state.rowsKey = key; renderResults(); }
  $("#step-results").hidden = !job.rows.length;

  updateRun();
  if (["queued", "running"].includes(job.status)) state.poll = setTimeout(pollJob, 1000);
  else loadHistory();
}

function renderResults() {
  const job = state.job, base = `/runs/${job.id}/`;
  const variant = job.synthsod_variant === "ensembleset" ? "EnsembleSet 학습본" : "SynthSOD 학습본";
  $("#result-meta").textContent = `${job.name} · ${job.start}초부터 ${job.duration}초 · SynthSOD: ${variant} · Mega 53: ${job.mega_chunk ?? 10}초 청크`;
  $("#zip").href = `/api/jobs/${job.id}/zip`;

  const mix = $("#mix-cell");
  mix.innerHTML = "";
  mix.append(makeCell(base + "input.wav", "원본 믹스", "mix", null));

  const m = $("#matrix");
  m.innerHTML = `<div class="m-head"><span>악기</span><span class="a">A · SynthSOD 베이스라인</span><span class="b">B · Mega 53 Stems</span><span>A/B</span></div>`;
  let group = null;
  for (const row of job.rows) {
    if (row.group !== group) {
      group = row.group;
      const g = document.createElement("div");
      g.className = "m-group";
      g.textContent = group === "악기군" ? "악기군 (SynthSOD는 개별 스템 합산)" : group;
      m.append(g);
    }
    const r = document.createElement("div");
    r.className = "m-row";
    r.innerHTML = `<span class="row-label">${row.label}</span>`;
    r.append(makeCell(row.synthsod && base + row.synthsod, row.label + " · SynthSOD", "a", r));
    r.append(makeCell(row.mega53 && base + row.mega53, row.label + " · Mega 53", "b", r));
    const ab = document.createElement("button");
    ab.className = "ab";
    ab.textContent = "A/B";
    ab.disabled = !(row.synthsod && row.mega53);
    ab.title = "두 모델 결과를 같은 시점에 동시 재생, 다시 누르면 전환";
    ab.onclick = () => {
      if (player.abBtn === ab) return switchAB();
      startPlayer([base + row.synthsod, base + row.mega53], { label: row.label, rowEl: r, abBtn: ab });
      $("#player-label").textContent = `${row.label} — A: SynthSOD`;
    };
    r.append(ab);
    m.append(r);
  }
}

function makeCell(url, label, cls, rowEl) {
  const c = document.createElement("div");
  c.dataset.model = cls === "a" ? "A · SynthSOD" : cls === "b" ? "B · Mega 53" : "";
  if (!url) { c.className = "cell empty"; c.textContent = "해당 스템 없음"; return c; }
  c.className = `cell ${cls}`;
  c.innerHTML = `<button class="play" title="재생">▶</button><canvas></canvas><span class="meta"><span class="db">…</span><a href="${url}" download>저장</a></span>`;
  const btn = c.querySelector(".play"), cv = c.querySelector("canvas");
  const play = (at = 0) => {
    if (player.btn === btn && at === 0) return stopAll();
    startPlayer([url], { label, rowEl, btn, at });
  };
  btn.onclick = () => play();
  cv.onclick = (e) => {
    const d = cv._duration;
    if (d) play((e.offsetX / cv.clientWidth) * d);
  };
  loadWave(url).then(({ peaks, rms, duration }) => {
    cv._duration = duration;
    cv._peaks = peaks;
    const db = 20 * Math.log10(rms + 1e-9);
    c.querySelector(".db").textContent = db < -80 ? "무음" : `${db.toFixed(0)} dB`;
    c.classList.toggle("silent", db < -50);
    drawWave(cv, peaks, css(cls === "a" ? "--a" : cls === "b" ? "--b" : "--muted"));
  }).catch(() => { c.querySelector(".db").textContent = "읽기 실패"; });
  return c;
}

/* ---------- 기록 ---------- */
async function loadHistory() {
  const list = await (await fetch("/api/jobs")).json();
  const ul = $("#history-list");
  ul.innerHTML = list.length ? "" : `<li class="muted">아직 없음</li>`;
  const label = { done: "완료", error: "오류", running: "진행 중", queued: "대기" };
  for (const j of list) {
    const li = document.createElement("li");
    li.dataset.id = j.id;
    li.classList.toggle("active", state.job?.id === j.id);
    li.innerHTML = `<div class="h-name"></div><div class="h-meta">${j.id.slice(0, 11)} · ${j.duration}초 · ${label[j.status] || j.status}</div>`;
    li.querySelector(".h-name").textContent = j.name;
    li.onclick = () => { openJob(j.id); $("#history").classList.remove("open"); };
    ul.append(li);
  }
}

async function deleteJob() {
  if (!state.job) return;
  const busy = ["queued", "running"].includes(state.job.status);
  const msg = busy
    ? "진행 중인 분리를 취소하고 이 작업을 삭제할까요?\n(현재 청크가 끝나는 대로 몇 초 안에 멈춥니다)"
    : "이 작업의 결과 파일을 모두 삭제할까요?";
  if (!confirm(msg)) return;
  const r = await fetch(`/api/jobs/${state.job.id}`, { method: "DELETE" });
  if (!r.ok) return alert((await r.json()).detail);
  clearTimeout(state.poll);
  stopAll();
  state.job = null;
  $("#step-progress").hidden = $("#step-results").hidden = true;
  loadHistory();
  updateRun();
}
$("#delete").onclick = deleteJob;
$("#cancel").onclick = deleteJob;
$("#toggle-history").onclick = () => $("#history").classList.toggle("open");

// 테마·크기 변경 시 파형 다시 그리기
let resizeT;
const redraw = () => {
  drawInput();
  if (!$("#player-bar").hidden) drawPlayerWave();
  document.querySelectorAll(".cell canvas").forEach((cv) => {
    if (!cv._peaks) return;
    const cls = cv.parentElement.classList;
    drawWave(cv, cv._peaks, css(cls.contains("a") ? "--a" : cls.contains("b") ? "--b" : "--muted"));
  });
};
window.addEventListener("resize", () => { clearTimeout(resizeT); resizeT = setTimeout(redraw, 150); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);

loadStems();
loadHistory();
updateRun();
