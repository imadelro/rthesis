const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusText = document.getElementById("status");
const results = document.getElementById("results");
const detectPill = document.getElementById('detectPill');
const detectDetail = document.getElementById('detectDetail');
const detectCard = document.getElementById('detectCard');

function log(msg) {
  console.log(`[popup] ${msg}`);
}

// Track whether we're currently recording (prevent showing processing UI after stop)
let _isRecording = false;

// Query background for current recording state when popup opens
try {
  chrome.runtime.sendMessage({ action: 'getRecordingState' }, (resp) => {
    try {
      if (resp && resp.isRecording) {
        statusText.textContent = 'Status: Recording...';
        startBtn.disabled = true;
        stopBtn.disabled = false;
        if (detectCard) detectCard.style.display = 'none';
        if (detectPill) { detectPill.className = 'pill pending'; detectPill.textContent = 'Detecting...'; }
      }
    } catch (e) {}
  });
} catch (e) {}

startBtn.addEventListener("click", () => {
  log('Start button clicked. Sending startCapture to active tab');
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const tab = tabs && tabs[0];
    if (!tab || !tab.id) {
      statusText.textContent = 'No active tab';
      return;
    }

    // Ensure content script is injected (the tab might have been opened before
    // the extension was loaded). Inject file if needed, then send message.
    function sendStart() {
      chrome.tabs.sendMessage(tab.id, { action: 'startCapture' }, (response) => {
        if (chrome.runtime.lastError) {
          console.error('sendMessage error:', chrome.runtime.lastError.message);
          statusText.textContent = 'Start failed: ' + chrome.runtime.lastError.message;
          return;
        }

        log('content script response: ' + JSON.stringify(response));
        if (response?.started) {
              statusText.textContent = 'Status: Waiting for video...';
          _isRecording = true;
          startBtn.disabled = true;
          stopBtn.disabled = false;
          // Reset detection pill to pending each new recording
          // Hide detection card until first classification result
          if (detectCard) detectCard.style.display = 'none';
          if (detectPill) { detectPill.className = 'pill'; detectPill.textContent = ''; }
          if (detectDetail) { detectDetail.style.display = 'none'; detectDetail.textContent = ''; }
        } else {
          statusText.textContent = 'Failed to start: ' + (response?.reason || 'unknown');
        }
      });
    }

    // Try to inject content script if the receiving end might not exist.
    if (chrome.scripting && chrome.scripting.executeScript) {
      chrome.scripting.executeScript(
        { target: { tabId: tab.id }, files: ['content_capture.js'] },
        () => {
          if (chrome.runtime.lastError) {
            console.warn('scripting.executeScript error:', chrome.runtime.lastError.message);
            // still attempt to send message in case the script exists
          }
          sendStart();
        }
      );
    } else {
      // Fallback: just try sending the message
      sendStart();
    }
  });
});

stopBtn.addEventListener("click", () => {
  log('Stop button clicked. Sending stopCapture to active tab');
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const tab = tabs && tabs[0];
      if (!tab || !tab.id) {
        statusText.textContent = 'No active tab';
        return;
      }

      function sendStop() {
        chrome.tabs.sendMessage(tab.id, { action: 'stopCapture' }, (response) => {
          if (chrome.runtime.lastError) {
            console.error('sendMessage error:', chrome.runtime.lastError.message);
            statusText.textContent = 'Stop failed: ' + chrome.runtime.lastError.message;
            return;
          }

          log('content script response: ' + JSON.stringify(response));
          if (response?.stopped) {
              statusText.textContent = 'Status: Stopped';
              startBtn.disabled = false;
              stopBtn.disabled = true;
              _isRecording = false;
              // Remove any processing/loading UI so the loading bar stops immediately
              try { const procEl = document.getElementById('processing'); if (procEl) procEl.remove(); } catch (e) {}
              try { if (results) { results.style.display = 'none'; results.innerHTML = ''; } } catch (e) {}
            } else {
              statusText.textContent = 'Failed to stop: ' + (response?.reason || 'unknown');
            }
        });
      }

      if (chrome.scripting && chrome.scripting.executeScript) {
        // ensure script present before sending stop
        chrome.scripting.executeScript(
          { target: { tabId: tab.id }, files: ['content_capture.js'] },
          () => {
            if (chrome.runtime.lastError) {
              console.warn('scripting.executeScript error:', chrome.runtime.lastError.message);
            }
            sendStop();
          }
        );
      } else {
        sendStop();
      }
    });
});

// Listen for background notifications that the recording is ready
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('[popup] runtime.onMessage', message);

  if (message.action === 'recordingReady') {
    statusText.textContent = 'Status: Processing — detecting language...';
    if (results) {
      results.style.display = 'block';
      results.innerHTML = '<h4>Processing</h4>';
      const proc = document.createElement('div');
      proc.id = 'processing';
      proc.style.textAlign = 'left';
      proc.style.fontStyle = 'italic';
      proc.style.marginTop = '6px';
      proc.textContent = 'Detecting language and classifying content';
      proc.className = 'dots';
      results.appendChild(proc);
    }
  }

  if (message.action === 'recordingStarted') {
    statusText.textContent = 'Status: Recording...';
    startBtn.disabled = true;
    stopBtn.disabled = false;
    // Reset detection UI to pending while recording
    if (detectCard) detectCard.style.display = 'none';
    if (detectPill) { detectPill.className = 'pill pending'; detectPill.textContent = 'Detecting...'; }
    if (detectDetail) { detectDetail.style.display = 'none'; detectDetail.textContent = ''; }
    if (results) {
      results.style.display = 'block';
      results.innerHTML = '<h4>Recording</h4>';
      const proc = document.createElement('div');
      proc.id = 'processing';
      proc.style.textAlign = 'left';
      proc.style.fontStyle = 'italic';
      proc.style.marginTop = '6px';
      proc.textContent = 'Recording audio...';
      proc.className = 'dots';
      results.appendChild(proc);
    }
  }

  if (message.action === 'transcriptionReady') {
    console.log('[popup] transcriptionReady (transcript hidden)', message);
    // remove processing indicator
  const procEl = document.getElementById('processing'); if (procEl) procEl.remove();
    
    // Check if capture was interrupted
    if (message.isInterrupted) {
      statusText.textContent = 'Status: Detection interrupted — process stopped';
      if (detectCard) detectCard.style.display = 'block';
      // Hide any processing UI so the loading animation stops
      if (results) { results.style.display = 'none'; results.innerHTML = ''; }
      if (detectPill) { detectPill.className = 'pill pending'; detectPill.textContent = 'Interrupted'; }
      if (detectDetail) { detectDetail.style.display = 'none'; detectDetail.textContent = ''; }
      return;
    }
    
    statusText.textContent = message.error ? 'Status: Transcription error' : 'Status: Ready';
    // Detection result
    const label = Object.prototype.hasOwnProperty.call(message, 'hateLabel') ? message.hateLabel : null;
    const score = Object.prototype.hasOwnProperty.call(message, 'hateScore') ? message.hateScore : null;
  // Removed method and model display

    function pct(n) { return (typeof n === 'number' && isFinite(n)) ? Math.round(n * 100) : null; }

    if (!label) {
      // Keep card hidden if no label
      if (detectCard) detectCard.style.display = 'none';
      if (detectPill) { detectPill.className = 'pill'; detectPill.textContent = ''; }
      if (detectDetail) { detectDetail.style.display = 'none'; detectDetail.textContent = ''; }
      return;
    }
    if (detectCard) detectCard.style.display = 'block';

    // Robust label mapping: accept variations and fall back to score threshold
    console.log('[popup] received label:', label, 'score:', score);
    let pillClass = 'pending';
    let pillText = 'Unknown';
    const perc = pct(score);
    const lbl = (typeof label === 'string') ? label.toLowerCase() : '';
    if (lbl.includes('hate') || lbl.includes('offens') || lbl.includes('abuse')) {
      pillClass = 'bad';
      pillText = (label || 'Hate') + (perc !== null ? ` (${perc}%)` : '');
    } else if (lbl.includes('non') || lbl.includes('safe') || lbl.includes('benign')) {
      pillClass = 'good';
      pillText = (label || 'Non-Hate') + (perc !== null ? ` (${perc}%)` : '');
    } else if (typeof score === 'number' && isFinite(score)) {
      // If model didn't provide a readable label, infer by score (threshold 0.5)
      if (score >= 0.5) {
        pillClass = 'bad';
        pillText = 'Hate Speech' + (perc !== null ? ` (${perc}%)` : '');
      } else {
        pillClass = 'good';
        pillText = 'Non-Hate' + (perc !== null ? ` (${perc}%)` : '');
      }
    } else {
      pillClass = 'pending';
      pillText = 'Unknown';
    }

    // Update pill and detail UI
    if (detectPill) { detectPill.className = 'pill ' + pillClass; detectPill.textContent = pillText; }
    if (detectDetail) {
      if (pillClass === 'bad' && typeof score === 'number') {
        const conf = pct(score);
        detectDetail.textContent = `This video was detected to contain hate speech patterns with a confidence score of ${conf}%`;
        detectDetail.style.display = 'block';
      } else if (pillClass === 'good') {
        detectDetail.innerHTML = `<div style="color:#cdebd3;">No hate speech detected</div><div style="font-size:11px;color:#bdbdbd;margin-top:4px;">Automated detection may be imperfect.</div>`;
        detectDetail.style.display = 'block';
      } else {
        detectDetail.textContent = '';
        detectDetail.style.display = 'none';
      }
    }

    // Warning box handling: show structured warning when pillClass is bad
    try {
      const warningBox = document.getElementById('warningBox');
      const warningHeader = document.getElementById('warningHeader');
      const warningTags = document.getElementById('warningTags');
      const warningCaution = document.getElementById('warningCaution');
      if (warningBox && warningHeader && warningTags && warningCaution) {
        if (pillClass === 'bad' && typeof score === 'number') {
          const conf = pct(score);
          warningHeader.textContent = `⚠️ Potential hate speech detected (${conf}% confidence)`;
          // Use neutral, informative tags
          warningTags.innerHTML = '';
          const tags = ['Harassment', 'Targeting a group', 'Derogatory language'];
          for (const t of tags) {
            const el = document.createElement('span');
            el.textContent = t;
            el.style.padding = '4px 8px';
            el.style.background = '#2e2e2e';
            el.style.color = '#fff';
            el.style.borderRadius = '999px';
            el.style.fontSize = '12px';
            warningTags.appendChild(el);
          }
          warningBox.style.display = 'block';
        } else {
          warningBox.style.display = 'none';
        }
      }
    } catch (e) {}
  }
});

// On popup open ask background for last transcript (in case it was produced
// while popup was closed). Background replies with { transcript }.
// Removed last transcript retrieval (no transcript UI)
