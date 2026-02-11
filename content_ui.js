// content_ui.js — injects a floating widget into TikTok pages
(function() {
  try {
    if (!location.hostname.includes('tiktok.com')) return;
  } catch (e) {
    return;
  }

  // Avoid injecting twice
  if (document.getElementById('ext-widget-root')) return;

  const container = document.createElement('div');
  container.id = 'ext-widget-root';
  // keep container's own styles from leaking
  container.style.all = 'initial';
  const shadow = container.attachShadow({ mode: 'open' });

  const style = document.createElement('style');
  style.textContent = `
    :host { all: initial; }
    .widget {
      position: fixed;
      right: 16px;
      bottom: 80px;
      z-index: 2147483647;
      width: 260px;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji";
      box-shadow: 0 10px 30px rgba(0,0,0,0.6);
      border-radius: 12px;
      background: #161616;
      color: #fff;
      overflow: hidden;
      border: 1px solid #2a2a2a;
    }
    .header {
      padding: 10px 12px;
      font-weight: 700;
      background: #0f0f0f;
      color: #fff;
      text-align: center;
      font-size: 14px;
      letter-spacing: .2px;
      border-bottom: 1px solid #2a2a2a;
      cursor: move;
      user-select: none;
    }
    .body { padding: 10px 12px; font-size: 13px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .btn {
      padding: 8px 10px;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      background: #333;
      color: #fff;
      font-weight: 600;
    }
    .btn:disabled { opacity: .6; cursor: not-allowed; }
    .btn.secondary { background: #333; }
    #ext-stop { background: #ff2d4a; }
    .status { font-size: 12px; margin-top: 8px; color: #a0a0a0; font-style: italic; }
    .card { margin-top: 10px; border: 1px solid #2a2a2a; border-radius: 8px; padding: 8px; background: #1a1a1a; }
    .card h4 { margin: 2px 0 6px; font-size: 12px; color: #a0a0a0; text-transform: uppercase; letter-spacing: .4px; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }
    .pill.pending { background: #3a3a3a; color: #ddd; border: 1px solid #555; }
    .pill.good { background: #1f4d3d; color: #7fd8b4; border: 1px solid #3a6d5c; }
    .pill.bad { background: #4d1f1f; color: #ff8080; border: 1px solid #6d3a3a; }
    #ext-transcript { display:none; white-space:pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12px; line-height:1.4; max-height: 160px; overflow:auto; background:#1a1a1a; padding:8px; border-radius:8px; border:1px solid #2a2a2a; color: #ddd; }
    #ext-download { display:none; margin-top:6px; font-size:12px; color:#4a9eff; cursor:pointer; text-decoration:none; font-weight:600; }
    .progress-container { display: none; margin-top: 10px; }
    .progress-bar { width: 100%; height: 6px; background: #333; border-radius: 3px; overflow: hidden; }
    .progress-fill { height: 100%; background: #ff2d4a; border-radius: 3px; animation: loading 5s ease-in-out infinite; }
    @keyframes loading { 0% { width: 0%; } 50% { width: 100%; } 100% { width: 100%; } }
    .recording-indicator { display: none; width: 12px; height: 12px; background: #ff0000; border-radius: 50%; animation: blink 0.6s ease-in-out infinite; margin-right: 6px; }
    @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
  `;

  const html = document.createElement('div');
  html.className = 'widget';
  html.innerHTML = `
    <div class="header"><div style="display: flex; align-items: center; justify-content: center;"><div class="recording-indicator" id="ext-recording-indicator"></div>TikTok Hate Speech Detector</div></div>
    <div class="body">
      <div class="row">
        <button id="ext-start" class="btn">Start Detecting</button>
        <button id="ext-stop" class="btn secondary" disabled>Stop</button>
      </div>
      <div id="ext-status" class="status">Idle</div>

      <div class="card" id="ext-detect-card">
          <h4>Detection</h4>
          <div id="ext-detect-pill" class="pill">Nothing processed</div>
          <div id="ext-detect-detail" style="font-size:12px; margin-top:6px; color:#fff; display:none;"></div>
          <div id="ext-warning-box" style="display:none; margin-top:8px; background:#211616; padding:8px; border-radius:6px; color:#fff;">
            <div id="ext-warning-header" style="font-weight:700; margin-bottom:6px;">⚠️ Potential hate speech detected (0% confidence)</div>
            <div style="font-size:12px; color:#ddd; margin-bottom:6px;">Why this was flagged:</div>
            <div id="ext-warning-tags" style="display:flex; gap:6px; flex-wrap:wrap; margin-bottom:8px;"></div>
            <div id="ext-warning-caution" style="font-size:12px; color:#e6e6e6;">Please be cautious when sharing or engaging with this content.</div>
          </div>
          <div class="progress-container" id="ext-progress-container">
          <div class="progress-bar">
            <div class="progress-fill"></div>
          </div>
        </div>
      </div>

      <!-- Transcript removed per request -->
    </div>
  `;

  shadow.appendChild(style);
  shadow.appendChild(html);
  document.documentElement.appendChild(container);

  // Drag functionality
  const widget = shadow.querySelector('.widget');
  const header = shadow.querySelector('.header');
  let isDragging = false;
  let dragOffsetX = 0;
  let dragOffsetY = 0;

  header.addEventListener('mousedown', (e) => {
    isDragging = true;
    const rect = widget.getBoundingClientRect();
    dragOffsetX = e.clientX - rect.left;
    dragOffsetY = e.clientY - rect.top;
  });

  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const newX = e.clientX - dragOffsetX;
    const newY = e.clientY - dragOffsetY;
    widget.style.right = 'auto';
    widget.style.bottom = 'auto';
    widget.style.left = newX + 'px';
    widget.style.top = newY + 'px';
  });

  document.addEventListener('mouseup', () => {
    isDragging = false;
  });

  const startBtn = shadow.querySelector('#ext-start');
  const stopBtn = shadow.querySelector('#ext-stop');
  const status = shadow.querySelector('#ext-status');
  const detectPill = shadow.querySelector('#ext-detect-pill');
  const detectDetail = shadow.querySelector('#ext-detect-detail');
  const progressContainer = shadow.querySelector('#ext-progress-container');
  const recordingIndicator = shadow.querySelector('#ext-recording-indicator');

  function setStatus(s) { status.textContent = s; }

  startBtn.textContent = 'Start Detecting';
  startBtn.addEventListener('click', () => {
    // Enter waiting state until a video actually starts playing
    setStatus('Waiting for a video to play...');
    startBtn.disabled = true;
    stopBtn.disabled = false;
    if (recordingIndicator) recordingIndicator.style.display = 'none';
    // Continuous mode: content script keeps recording as you scroll through videos
    window.dispatchEvent(new CustomEvent('EXTENSION_START_CAPTURE'));
    // Reset detection UI to pending state
    if (detectPill) { detectPill.className = 'pill pending'; detectPill.textContent = 'Detecting...'; }
    if (detectDetail) { detectDetail.style.display = 'none'; detectDetail.textContent = ''; }
  });

  // Show recording indicator only when the content script reports recording actually started
  window.addEventListener('EXTENSION_RECORDING_STARTED', (ev) => {
    try {
      setStatus('Recording...');
      if (recordingIndicator) recordingIndicator.style.display = 'block';
    } catch (e) {}
  });

  // When recording stops (before/while processing), update UI accordingly
  window.addEventListener('EXTENSION_RECORDING_STOPPED', (ev) => {
    try {
      setStatus('Processing...');
      if (recordingIndicator) recordingIndicator.style.display = 'none';
      if (progressContainer) progressContainer.style.display = 'block';
    } catch (e) {}
  });

  stopBtn.addEventListener('click', () => {
    setStatus('Stopping…');
    stopBtn.disabled = true;
    startBtn.disabled = false;
    if (recordingIndicator) recordingIndicator.style.display = 'none';
    if (progressContainer) progressContainer.style.display = 'none'; // hide progress bar immediately
    window.dispatchEvent(new CustomEvent('EXTENSION_STOP_CAPTURE'));
  });
  // No select-video feature: content script records the current visible video.

  // Listen for progress messages from the capture script / background
  window.addEventListener('EXTENSION_STATUS', (ev) => {
    const { statusText } = ev.detail || {};
    if (statusText) setStatus(statusText);
  });

  // Show an explicit detecting message while transcription runs
  window.addEventListener('EXTENSION_STATUS', (ev) => {
    const { statusText } = ev.detail || {};
    if (statusText && statusText.toLowerCase().includes('detect')) {
      setStatus('Processing...');
    }
  });

  // Listen for when a recording is ready (show processing state only)
  window.addEventListener('EXTENSION_RECORDING_READY', (ev) => {
    const { b64, mime } = ev.detail || {};
    setStatus('Processing, please be patient...');
    if (recordingIndicator) recordingIndicator.style.display = 'none';
    if (progressContainer) progressContainer.style.display = 'block';
    // We don't keep the audio in the widget; user can play it in the page.
    // Optionally we could offer a download link for the raw recording here.
  });

  // Ignore transcript messages (transcript UI removed)
  window.addEventListener('EXTENSION_TRANSCRIPT', (ev) => {
    const detail = ev.detail || {};
    
    // Check if capture was interrupted
    if (detail.isInterrupted) {
      setStatus('Detection interrupted — process stopped');
      if (progressContainer) progressContainer.style.display = 'none';
      if (detectPill) { detectPill.className = 'pill pending'; detectPill.textContent = 'Interrupted'; }
      if (detectDetail) { detectDetail.style.display = 'none'; detectDetail.textContent = ''; }
      return;
    }
    
    if (detail.error) {
      setStatus('Transcription error');
      const card = shadow.querySelector('#ext-detect-card');
      if (card) card.style.display = 'block';
      if (detectPill) { detectPill.className = 'pill pending'; detectPill.textContent = 'Error'; }
      if (detectDetail) { detectDetail.style.display = 'none'; detectDetail.textContent = ''; }
      return;
    }

    // Update status and detection pill based on hateLabel/score
    setStatus('Ready');
    if (progressContainer) progressContainer.style.display = 'none';
    const label = Object.prototype.hasOwnProperty.call(detail, 'hateLabel') ? detail.hateLabel : null;
    const score = Object.prototype.hasOwnProperty.call(detail, 'hateScore') ? detail.hateScore : null;

    function pct(n) {
      if (typeof n !== 'number' || !isFinite(n)) return '';
      return Math.round(n * 100);
    }

    const card = shadow.querySelector('#ext-detect-card');
    if (!label) {
      // No label: show "Nothing processed"
      if (detectPill) { detectPill.className = 'pill'; detectPill.textContent = 'Nothing processed'; }
      if (detectDetail) { detectDetail.style.display = 'none'; detectDetail.textContent = ''; }
      return;
    }

    // Robust label mapping (accept variations + fallback to score)
    let pillClass = 'pending';
    let pillText = label || 'Unknown';
    const perc = pct(score);
    const lbl = (typeof label === 'string') ? label.toLowerCase() : '';
    const isNonHate = /\bnon[-\s]?hate\b|\bnot\s+hate\b|\bsafe\b|\bbenign\b/.test(lbl);
    const isHate = /\bhate\b|\boffens\w*\b|\babuse\w*\b/.test(lbl);
    if (isNonHate) {
      pillClass = 'good';
      pillText = (label || 'Non-Hate') + (perc !== '' ? ` (${perc}%)` : '');
    } else if (isHate) {
      pillClass = 'bad';
      pillText = (label || 'Hate') + (perc !== '' ? ` (${perc}%)` : '');
    } else if (typeof score === 'number' && isFinite(score)) {
      if (score >= 0.5) {
        pillClass = 'bad';
        pillText = 'Hate Speech' + (perc !== '' ? ` (${perc}%)` : '');
      } else {
        pillClass = 'good';
        pillText = 'Non-Hate' + (perc !== '' ? ` (${perc}%)` : '');
      }
    } else {
      pillClass = 'pending';
      pillText = 'Unknown';
    }

    if (detectPill) { detectPill.className = 'pill ' + pillClass; detectPill.textContent = pillText; }
    const parts = [];
    if (typeof score === 'number') parts.push(`score: ${score.toFixed(3)}`);
    if (detectDetail) {
      if (pillClass === 'bad' && typeof score === 'number') {
        // Details are shown in the structured warning box; keep detectDetail hidden
        // (warning box populated below)
        detectDetail.style.display = 'none';
      } else if (pillClass === 'good') {
        // Minimal, non-intrusive reassurance state
        detectDetail.innerHTML = `<div style="color:#cdebd3;">No hate speech detected</div><div style="font-size:11px;color:#bdbdbd;margin-top:4px;">Automated detection may be imperfect.</div>`;
        detectDetail.style.display = 'block';
      } else {
        detectDetail.textContent = parts.join(' \u2022 ');
        detectDetail.style.display = parts.length ? 'block' : 'none';
      }
    }

    // Populate floating warning box for bad detections
    try {
      const warnBox = shadow.querySelector('#ext-warning-box');
      const warnHeader = shadow.querySelector('#ext-warning-header');
      const warnTags = shadow.querySelector('#ext-warning-tags');
      if (warnBox && warnHeader && warnTags) {
        if (pillClass === 'bad' && typeof score === 'number') {
          const conf = pct(score);
          warnHeader.textContent = `⚠️ Potential hate speech detected (${conf}% confidence score)`;
          warnTags.innerHTML = '';
          const tags = ['Harassment', 'Targeting a group', 'Derogatory language'];
          for (const t of tags) {
            const el = document.createElement('div');
            el.textContent = t;
            el.style.padding = '4px 8px';
            el.style.background = '#2e2e2e';
            el.style.color = '#fff';
            el.style.borderRadius = '999px';
            el.style.fontSize = '12px';
            warnTags.appendChild(el);
          }
          warnBox.style.display = 'block';
        } else {
          warnBox.style.display = 'none';
        }
      }
    } catch (e) {}
  });

  // remove widget on navigation/unload
  window.addEventListener('beforeunload', () => {
    try { container.remove(); } catch (e) {}
  });
})();
