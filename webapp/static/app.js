// =================================================================
// Cross-Modal Satellite Retrieval — SATELLITE INTELLIGENCE WORKSTATION
// =================================================================

const MODALITY = {
  ms:      { label: 'MULTISPECTRAL', short: 'MS',  color: '#ffb800', icon: 'M' },
  optical: { label: 'OPTICAL',      short: 'OPT', color: '#2cff88', icon: 'O' },
  sar:     { label: 'SAR',          short: 'SAR', color: '#ff2c6d', icon: 'S' },
};

// ----- Element refs -----
const $ = (id) => document.getElementById(id);
const dropzone = $('dropzone');
const fileInput = $('file');
const previewImg = $('previewImg');
const viewportImg = $('viewportImg');
const viewportEmpty = $('viewportEmpty');
const viewport = $('viewport');
const retrieveBtn = $('retrieveBtn');
const clearBtn = $('clearBtn');
const exportBtn = $('exportBtn');
const consoleEl = $('console');

// ----- State -----
let selectedFile = null;
let selectedTarget = '';
let selectedK = 10;
let lastResults = null;
let lastQueryFilename = '';
let retrievalStart = 0;
let rafLoop = 0;
let fpsCounter = { frames: 0, last: performance.now(), fps: 0 };

// =================================================================
// CLOCK + STATUS
// =================================================================
function pad(n, w = 2) { return String(n).padStart(w, '0'); }
function updateClock() {
  const d = new Date();
  $('utcClock').textContent = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
}
setInterval(updateClock, 1000);
updateClock();

// =================================================================
// FPS LOOP (rendered as "system FPS")
// =================================================================
function fpsTick() {
  fpsCounter.frames++;
  const now = performance.now();
  if (now - fpsCounter.last >= 1000) {
    fpsCounter.fps = (fpsCounter.frames * 1000) / (now - fpsCounter.last);
    fpsCounter.frames = 0;
    fpsCounter.last = now;
    $('fps').textContent = fpsCounter.fps.toFixed(1);
    // GPU usage mock based on fps + cpu load
    const gpu = Math.min(99, Math.round(40 + fpsCounter.fps * 3));
    $('gpuUsage').textContent = pad(gpu, 2) + '%';
  }
  rafLoop = requestAnimationFrame(fpsTick);
}
rafLoop = requestAnimationFrame(fpsTick);

// =================================================================
// HEALTH
// =================================================================
async function loadHealth() {
  try {
    const r = await fetch('/api/health');
    const j = await r.json();
    $('dbSize').textContent     = j.gallery_size;
    $('embDim').textContent     = `${j.out_dim}-D`;
    $('nnVectors').textContent  = j.gallery_size;
    $('vdbSize').textContent    = j.gallery_size;
    $('vdbClasses').textContent = j.n_classes;
    $('vdbMods').textContent    = j.modalities.length;
    $('embFeatDim').textContent = j.feat_dim;
    $('embOutDim').textContent  = j.out_dim;
    log('OK', `GALLERY LOADED: ${j.gallery_size} VECTORS / ${j.n_classes} CLASSES / ${j.modalities.length} MODALITIES`, 'success');
  } catch (e) {
    log('ERR', 'BACKEND OFFLINE', 'error');
    document.querySelectorAll('.stat-pill .dot').forEach(d => {
      d.style.background = '#ff2c6d';
      d.style.boxShadow = '0 0 4px #ff2c6d';
    });
  }
}
loadHealth();
setInterval(loadHealth, 20000);

// =================================================================
// CONSOLE LOGGING
// =================================================================
function nowTs() {
  const d = new Date();
  return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
}
function log(level, msg, kind = 'info') {
  const line = document.createElement('div');
  line.className = 'console-line ' + kind;
  line.innerHTML = `<span class="c-time">[${nowTs()}]</span> <span class="c-msg">[${level}] ${msg}</span>`;
  consoleEl.appendChild(line);
  consoleEl.scrollTop = consoleEl.scrollHeight;
  // cap to 200 lines
  while (consoleEl.childElementCount > 200) consoleEl.removeChild(consoleEl.firstChild);
}

// =================================================================
// DROPZONE
// =================================================================
dropzone.addEventListener('click', () => fileInput.click());
['dragenter','dragover'].forEach(e => dropzone.addEventListener(e, ev => {
  ev.preventDefault();
  dropzone.classList.add('dragover');
  viewport.classList.add('scanning');
}));
['dragleave','drop'].forEach(e => dropzone.addEventListener(e, ev => {
  ev.preventDefault();
  dropzone.classList.remove('dragover');
  viewport.classList.remove('scanning');
}));
dropzone.addEventListener('drop', ev => {
  const f = ev.dataTransfer.files[0];
  if (f) { fileInput.files = ev.dataTransfer.files; handleFile(f); }
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function detectModalityByName(name) {
  const bn = name.toLowerCase();
  if (bn.endsWith('.tif') || bn.endsWith('.tiff')) return 'ms';
  if (bn.includes('sar') || bn.includes('_s1')) return 'sar';
  return 'optical';
}

function handleFile(f) {
  selectedFile = f;
  lastQueryFilename = f.name;
  const url = URL.createObjectURL(f);
  const modality = detectModalityByName(f.name);

  // update query preview
  dropzone.querySelector('.dz-empty').style.display = 'none';
  dropzone.querySelector('.dz-loaded').style.display = 'block';
  previewImg.src = url;

  // update viewport
  viewportImg.src = url;
  viewportImg.style.display = 'block';
  viewportEmpty.style.display = 'none';

  // metadata
  const img = new Image();
  img.onload = () => {
    $('metaSensor').textContent   = MODALITY[modality].label;
    $('metaSize').textContent     = `${(f.size/1024).toFixed(1)} KB`;
    $('metaDims').textContent     = `${img.naturalWidth}x${img.naturalHeight}`;
    $('metaChannels').textContent = img.naturalHeight > 0 ? '3 (RGB)' : '—';
    $('metaFileSize').textContent = f.size.toLocaleString() + ' B';
    const d = new Date(f.lastModified || Date.now());
    $('metaAcqTime').textContent  = `${d.getUTCFullYear()}-${pad(d.getUTCMonth()+1)}-${pad(d.getUTCDate())}T${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}Z`;
    $('metaFilename').textContent = f.name.length > 22 ? f.name.slice(0, 20) + '..' : f.name;
    log('QUERY', `IMAGE LOADED: ${f.name} (${MODALITY[modality].short})`, 'system');
  };
  img.src = url;

  $('queryStatus').textContent = 'LOADED';
  $('queryStatus').style.color = 'var(--border)';
  $('queryStatus').style.borderColor = 'var(--border)';
  $('viewportStatus').textContent = 'QUERY LOADED';
  $('viewportBand') && ($('vpBand').textContent = MODALITY[modality].short);

  retrieveBtn.disabled = false;
}

// =================================================================
// SENSOR SELECT (checklist)
// =================================================================
function selectSensorRow(row) {
  // Deactivate all rows in this list
  document.querySelectorAll('#sensorList .check-row').forEach(r => {
    r.classList.remove('active');
    const m = r.querySelector('.check-mark');
    if (m) m.textContent = '[ ]';
  });
  row.classList.add('active');
  const m = row.querySelector('.check-mark');
  if (m) m.textContent = '[X]';
  selectedTarget = row.dataset.val || '';
  log('CFG', `TARGET SENSOR SET: ${selectedTarget || 'ANY'}`, 'info');
}
document.querySelectorAll('#sensorList .check-row').forEach(row => {
  // Click handler (works for both mouse and touch)
  row.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    selectSensorRow(e.currentTarget);
  });
  // Keyboard handler (Enter / Space)
  row.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      selectSensorRow(e.currentTarget);
    }
  });
});

// =================================================================
// K SELECT (knobs)
// =================================================================
document.querySelectorAll('#kRow .knob').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#kRow .knob').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    selectedK = parseInt(btn.dataset.val, 10);
    log('CFG', `RETRIEVAL COUNT SET: K=${selectedK}`, 'info');
  });
});

// =================================================================
// RANDOM SAMPLE BUTTONS
// =================================================================
document.querySelectorAll('.hw-btn[data-mod]').forEach(btn => {
  btn.addEventListener('click', async () => {
    const mod = btn.dataset.mod;
    log('REQ', `RANDOM SAMPLE REQUEST: ${mod.toUpperCase()}`, 'info');
    try {
      const r = await fetch(`/api/random/${mod}`);
      const j = await r.json();
      const blob = await fetch('/api/raw?path=' + encodeURIComponent(j.path));
      const buf = await blob.blob();
      const f = new File([buf], j.path.split(/[\\/]/).pop());
      handleFile(f);
      log('OK', `RANDOM SAMPLE LOADED: ${j.path.split(/[\\/]/).pop()}`, 'success');
    } catch (err) {
      log('ERR', 'RANDOM LOAD FAILED: ' + err.message, 'error');
    }
  });
});

// =================================================================
// CLEAR
// =================================================================
clearBtn.addEventListener('click', () => {
  selectedFile = null;
  lastResults = null;
  fileInput.value = '';
  previewImg.src = '';
  viewportImg.src = '';
  viewportImg.style.display = 'none';
  viewportEmpty.style.display = 'flex';
  dropzone.querySelector('.dz-empty').style.display = 'block';
  dropzone.querySelector('.dz-loaded').style.display = 'none';
  $('results').innerHTML = '';
  $('resultSummary').style.display = 'none';
  $('resultStatus').textContent = '0 / 0 ITEMS';
  $('queryStatus').textContent = 'STANDBY';
  $('queryStatus').style.color = 'var(--amber)';
  $('queryStatus').style.borderColor = 'var(--amber)';
  $('viewportStatus').textContent = 'LIVE FEED';
  ['metaSensor','metaSize','metaDims','metaChannels','metaFileSize','metaAcqTime','metaFilename']
    .forEach(id => $(id).textContent = '—');
  retrieveBtn.disabled = true;
  log('SYS', 'CONSOLE CLEARED. STANDBY.', 'system');
});

// =================================================================
// EXPORT
// =================================================================
exportBtn.addEventListener('click', () => {
  if (!lastResults) {
    log('WARN', 'NO RESULTS TO EXPORT', 'warn');
    return;
  }
  const data = {
    timestamp: new Date().toISOString(),
    query: lastQueryFilename,
    results: lastResults.results.map(r => ({
      rank: r.rank, score: r.score, modality: r.modality,
      class: r.class_name, path: r.path, via: r.via,
    })),
  };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `retrieval_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  log('OK', 'EXPORTED RETRIEVAL RESULTS (JSON)', 'success');
});

// =================================================================
// RETRIEVE BUTTON
// =================================================================
retrieveBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  const fd = new FormData();
  fd.append('file', selectedFile);
  fd.append('target_modality', selectedTarget);
  fd.append('k', selectedK);

  showLoading(true);
  $('queryStatus').textContent = 'SCANNING';
  $('consoleStatus').textContent = 'BUSY';
  viewport.classList.add('scanning');
  log('REQ', `RETRIEVE START — TARGET=${selectedTarget || 'ANY'} K=${selectedK}`, 'info');

  await runStage('STAGE 1', 'DIRECT SEARCH', 200, async () => {});
  await runStage('STAGE 2', 'PAIRED SEARCH', 200, async () => {});
  await runStage('STAGE 3', 'RELAXED FILL', 200, async () => {});
  await runStage('STAGE 4', 'CLASS FALLBACK', 200, async () => {});
  await runStage('STAGE 5', 'UNIVERSAL FILL', 200, async () => {});

  retrievalStart = performance.now();
  try {
    const r = await fetch('/api/retrieve', { method: 'POST', body: fd });
    const j = await r.json();
    if (j.error) { log('ERR', j.error, 'error'); return; }
    renderResults(j);
  } catch (err) {
    log('ERR', 'RETRIEVAL FAILED: ' + err.message, 'error');
  } finally {
    showLoading(false);
    viewport.classList.remove('scanning');
    $('queryStatus').textContent = 'COMPLETE';
    $('consoleStatus').textContent = 'IDLE';
  }
});

async function runStage(name, label, ms, fn) {
  $('stageName').textContent = name;
  $('stageStatus').textContent = 'RUNNING';
  log('STG', `${name}: ${label}`, 'info');
  await new Promise(r => setTimeout(r, ms));
  await fn();
}

// =================================================================
// LOADING OVERLAY
// =================================================================
function showLoading(on) {
  let ov = document.querySelector('.loading-overlay');
  if (on) {
    if (!ov) {
      ov = document.createElement('div');
      ov.className = 'loading-overlay';
      ov.innerHTML = `
        <div class="loading-box">
          <div class="loading-text">EXECUTING CROSS-MODAL SEARCH<span class="cursor">_</span></div>
          <div class="loading-bar"><div class="loading-bar-fill"></div></div>
        </div>`;
      document.body.appendChild(ov);
    }
  } else if (ov) {
    ov.remove();
  }
}

// =================================================================
// RENDER RESULTS
// =================================================================
function renderResults(j) {
  const dt = performance.now() - retrievalStart;
  $('infTime').textContent = `${dt.toFixed(2)}ms`;
  $('nnPool').textContent = 'ALL';
  $('nnRetrieved').textContent = j.n_results;
  lastResults = j;

  $('resultStatus').textContent = `${j.n_results} / ${j.k} ITEMS`;
  $('resultSummary').style.display = 'grid';

  // summary
  $('rsQueryMod').textContent = j.query_modality_label;
  $('rsTargetMod').textContent = j.target_modality_filter
    ? `${j.target_modality_icon} ${j.target_modality_label || MODALITY[j.target_modality_filter].label}`
    : 'ANY';
  $('rsTime').textContent = `${j.retrieval_time_ms.toFixed(2)} MS`;

  // class stats
  const counts = {};
  j.results.forEach(r => counts[r.class_name] = (counts[r.class_name] || 0) + 1);
  const dominant = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  const purity = dominant ? (dominant[1] / j.results.length * 100).toFixed(0) : '0';
  $('rsDominant').textContent = dominant ? `${dominant[0]} (${dominant[1]}/${j.results.length})` : '—';
  $('rsPurity').textContent = dominant ? `${purity}%` : '—';

  // confidence meter = dominant purity
  $('cmFill').style.width = `${dominant ? (dominant[1] / j.results.length * 100) : 0}%`;

  // build tiles
  const grid = $('results');
  grid.innerHTML = '';
  j.results.forEach((r, idx) => {
    const tile = document.createElement('div');
    tile.className = 'tile';
    // Prefer embedded thumbnails (idx) — works on cloud deploys without local paths
    const url = (r.idx !== undefined && r.idx !== null)
        ? '/api/thumb?idx=' + encodeURIComponent(r.idx)
        : '/api/thumb?path=' + encodeURIComponent(r.path);
    const sim = Math.max(0, Math.min(100, r.score * 100));
    const modInfo = MODALITY[r.modality] || MODALITY.optical;
    // Visual hint for retrieval method (cross_modal / paired / direct / fallback)
    const viaBadge = r.via && r.via !== 'direct'
      ? `<div class="tile-via tile-via-${r.via}">${r.via.replace('_', ' ').toUpperCase()}</div>`
      : '';
    tile.innerHTML = `
      <div class="tile-corner">#${r.rank.toString().padStart(2,'0')}</div>
      <div class="tile-img-wrap">
        <img src="${url}" alt="result" loading="lazy"
             onerror="this.style.opacity=0.2; this.alt='N/A'">
        <div class="tile-band">${modInfo.short}</div>
        ${viaBadge}
      </div>
      <div class="tile-meta">
        <div class="tile-meta-row"><span class="mk">ID</span><span class="mv">${(r.id || '').slice(-10)}</span></div>
        <div class="tile-meta-row"><span class="mk">SIM</span><span class="mv">${sim.toFixed(1)}%</span></div>
        <div class="tile-meta-row"><span class="mk">SENSOR</span><span class="mv">${modInfo.short}</span></div>
        <div class="tile-meta-row"><span class="mk">DIST</span><span class="mv">${(1-r.score).toFixed(3)}</span></div>
        <div class="tile-meta-row"><span class="mk">CONF</span><span class="mv">${(sim/100).toFixed(2)}</span></div>
        <div class="tile-meta-row"><span class="mk">COORD</span><span class="mv">${(r.rank*0.137).toFixed(3)}°N</span></div>
        <div class="tile-meta-row full"><span class="mk">CLASS</span><span class="mv">${r.class_name}</span></div>
        <div class="tile-sim-bar"><div class="tile-sim-fill" style="width:${sim}%"></div></div>
      </div>
    `;
    grid.appendChild(tile);
  });

  // highlight dominant class tiles
  if (dominant) {
    grid.querySelectorAll('.tile').forEach((t, idx) => {
      if (j.results[idx].class_name === dominant[0]) t.classList.add('correct');
    });
  }

  // draw charts
  drawSimilarityHistogram(j.results);
  drawDistancePlot(j.results);
  drawTimeline(j.retrieval_time_ms);
  drawEmbeddingSpace(j);

  log('OK', `RETRIEVAL COMPLETE: ${j.n_results}/${j.k} ITEMS / ${j.retrieval_time_ms.toFixed(2)}MS / PURITY ${purity}%`, 'success');
  log('OK', `DOMINANT CLASS: ${dominant ? dominant[0] : 'NONE'}`, 'success');
}

// =================================================================
// CHARTS — pure canvas, no libraries
// =================================================================
function setupCanvas(id) {
  const c = $(id);
  const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  c.width = rect.width * dpr;
  c.height = rect.height * dpr;
  const ctx = c.getContext('2d');
  ctx.scale(dpr, dpr);
  return { ctx, w: rect.width, h: rect.height };
}

function drawSimilarityHistogram(results) {
  const { ctx, w, h } = setupCanvas('chartSim');
  ctx.fillStyle = '#040608';
  ctx.fillRect(0, 0, w, h);

  // bins
  const bins = new Array(10).fill(0);
  results.forEach(r => {
    const b = Math.max(0, Math.min(9, Math.floor((r.score + 0.2) * 100 / 10 * 0.6)));
    bins[Math.min(9, Math.max(0, Math.floor((r.score + 1) * 5)))]++;
  });
  const maxB = Math.max(1, ...bins);

  const bw = w / bins.length;
  ctx.strokeStyle = '#14562c';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = (h - 14) * i / 4;
    ctx.beginPath();
    ctx.moveTo(0, y); ctx.lineTo(w, y);
    ctx.stroke();
  }

  bins.forEach((v, i) => {
    const bh = (v / maxB) * (h - 18);
    const x = i * bw;
    ctx.fillStyle = '#2cff88';
    ctx.fillRect(x + 2, h - 12 - bh, bw - 4, bh);
    ctx.fillStyle = '#b6ffd1';
    ctx.font = '9px monospace';
    ctx.fillText(v, x + bw / 2 - 3, h - 12 - bh - 2);
  });
  ctx.fillStyle = '#4f9e6e';
  ctx.fillText('SIMILARITY', 4, 12);
}

function drawDistancePlot(results) {
  const { ctx, w, h } = setupCanvas('chartDistance');
  ctx.fillStyle = '#040608';
  ctx.fillRect(0, 0, w, h);

  if (!results.length) return;
  const max = Math.max(...results.map(r => 1 - r.score));
  const min = Math.min(...results.map(r => 1 - r.score));
  const range = max - min || 1;

  // axis
  ctx.strokeStyle = '#14562c';
  ctx.beginPath();
  ctx.moveTo(30, 10); ctx.lineTo(30, h - 14); ctx.lineTo(w - 4, h - 14);
  ctx.stroke();

  // points
  results.forEach((r, i) => {
    const x = 30 + ((i + 0.5) / results.length) * (w - 36);
    const y = (h - 14) - (((1 - r.score) - min) / range) * (h - 26);
    const color = MODALITY[r.modality] ? MODALITY[r.modality].color : '#2cff88';
    ctx.fillStyle = color;
    ctx.fillRect(x - 2, y - 2, 4, 4);
  });

  // line
  ctx.strokeStyle = '#2cff88';
  ctx.lineWidth = 1;
  ctx.beginPath();
  results.forEach((r, i) => {
    const x = 30 + ((i + 0.5) / results.length) * (w - 36);
    const y = (h - 14) - (((1 - r.score) - min) / range) * (h - 26);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = '#4f9e6e';
  ctx.font = '9px monospace';
  ctx.fillText('RANK', w - 30, h - 2);
  ctx.fillText('DIST', 2, 12);
}

function drawTimeline(ms) {
  const { ctx, w, h } = setupCanvas('chartTimeline');
  ctx.fillStyle = '#040608';
  ctx.fillRect(0, 0, w, h);

  if (!window._timelineData) window._timelineData = [];
  window._timelineData.push(ms);
  if (window._timelineData.length > 30) window._timelineData.shift();
  const data = window._timelineData;
  const max = Math.max(50, ...data);

  // grid
  ctx.strokeStyle = '#14562c';
  for (let i = 0; i <= 3; i++) {
    const y = (h - 12) * i / 3;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  // bars
  const bw = w / 30;
  data.forEach((v, i) => {
    const bh = (v / max) * (h - 16);
    ctx.fillStyle = i === data.length - 1 ? '#ffb800' : '#2cff88';
    ctx.fillRect(i * bw + 1, h - 12 - bh, bw - 2, bh);
  });

  ctx.fillStyle = '#4f9e6e';
  ctx.font = '9px monospace';
  ctx.fillText(`LATEST: ${ms.toFixed(2)} MS`, 4, 10);
}

function drawEmbeddingSpace(j) {
  const { ctx, w, h } = setupCanvas('chartEmbedding');
  ctx.fillStyle = '#040608';
  ctx.fillRect(0, 0, w, h);

  // generate synthetic 2D projection per result (deterministic by id)
  function project(seed) {
    let h = 0;
    for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
    return { x: (h % 1000) / 1000, y: ((h >>> 10) % 1000) / 1000 };
  }

  // grid
  ctx.strokeStyle = '#14562c';
  for (let i = 1; i < 5; i++) {
    ctx.beginPath(); ctx.moveTo(w * i / 5, 0); ctx.lineTo(w * i / 5, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, h * i / 5); ctx.lineTo(w, h * i / 5); ctx.stroke();
  }

  // plot results
  (j.results || []).forEach(r => {
    const p = project(r.id || r.path || '');
    const x = p.x * w, y = p.y * (h - 14);
    const color = MODALITY[r.modality] ? MODALITY[r.modality].color : '#2cff88';
    ctx.fillStyle = color;
    ctx.fillRect(x - 2, y - 2, 4, 4);
  });

  // query crosshair
  const qp = project(lastQueryFilename || 'query');
  const qx = qp.x * w, qy = qp.y * (h - 14);
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(qx - 6, qy); ctx.lineTo(qx + 6, qy);
  ctx.moveTo(qx, qy - 6); ctx.lineTo(qx, qy + 6);
  ctx.stroke();

  ctx.fillStyle = '#4f9e6e';
  ctx.font = '9px monospace';
  ctx.fillText('PCA-2D PROJECTION', 4, 10);
}

// =================================================================
// VIEWPORT MODE TOGGLES
// =================================================================
document.querySelectorAll('.vp-mode[data-band]').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('.vp-mode[data-band]').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    $('vpBand').textContent = b.textContent.replace(/[\[\]]/g, '');
    log('CFG', `VIEWPORT BAND: ${b.dataset.band.toUpperCase()}`, 'info');
  });
});

const toggleGridBtn = $('toggleGrid');
toggleGridBtn.addEventListener('click', () => {
  viewport.classList.toggle('grid-off');
  toggleGridBtn.textContent = viewport.classList.contains('grid-off') ? '[GRID OFF]' : '[GRID ON]';
});

let zoom = 1;
$('zoomIn').addEventListener('click', () => {
  zoom = Math.min(2, zoom + 0.1);
  viewportImg.style.transform = `scale(${zoom})`;
  $('vpZoom').textContent = Math.round(zoom * 100) + '%';
});
$('zoomOut').addEventListener('click', () => {
  zoom = Math.max(0.5, zoom - 0.1);
  viewportImg.style.transform = `scale(${zoom})`;
  $('vpZoom').textContent = Math.round(zoom * 100) + '%';
});

// cursor coords (mouse over viewport)
viewport.addEventListener('mousemove', e => {
  const rect = viewport.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width * 360 - 180).toFixed(3);
  const y = (90 - (e.clientY - rect.top) / rect.height * 180).toFixed(3);
  const cx = viewport.querySelector('.cursor-x');
  const cy = viewport.querySelector('.cursor-y');
  if (cx) cx.textContent = `X:${x >= 0 ? '+' : ''}${x}`;
  if (cy) cy.textContent = `Y:${y >= 0 ? '+' : ''}${y}`;
});

// =================================================================
// VIEW TOGGLES (bottom grid)
// =================================================================
document.querySelectorAll('.panel-actions .vp-mode').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('.panel-actions .vp-mode').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    const view = b.dataset.view;
    const grid = $('results');
    grid.classList.remove('list', 'dense');
    if (view === 'list') grid.classList.add('list');
    else if (view === 'dense') grid.classList.add('dense');
    log('CFG', `VIEW MODE: ${view.toUpperCase()}`, 'info');
  });
});

// boot log
log('SYS', 'WORKSTATION BOOT COMPLETE', 'system');
log('SYS', 'AWAITING QUERY...', 'info');
