// Background (service worker) orchestrates an offscreen capture page.
// Service workers cannot reliably create AudioContext or play audio, so we
// delegate actual capturing and recording to `capture.html` (offscreen).

const OFFSCREEN_URL = 'capture.html';

async function ensureOffscreen() {
  try {
    // Some Chrome/Chromium versions don't implement chrome.offscreen yet.
    // Guard against that and provide a visible-window fallback.
    if (chrome.offscreen && typeof chrome.offscreen.hasDocument === 'function') {
      const hasDoc = await chrome.offscreen.hasDocument();
      if (!hasDoc) {
        await chrome.offscreen.createDocument({
          url: OFFSCREEN_URL,
          reasons: ['AUDIO_CAPTURE'],
          justification: 'Record tab audio and allow playback during recording'
        });
        console.log('[background] Offscreen document created');
      } else {
        console.log('[background] Offscreen document already exists');
      }
      return { kind: 'offscreen' };
    }

    // Fallback: open a visible popup window that loads capture.html. This is
    // less ideal (user will see a small window) but works on browsers without
    // the offscreen API.
    console.warn('[background] chrome.offscreen unavailable — opening visible capture window as fallback');
    const url = chrome.runtime.getURL(OFFSCREEN_URL);
    const created = await chrome.windows.create({ url, type: 'popup', focused: true, width: 420, height: 320 });

    // Wait for the tab in that window to finish loading before returning.
    const tab = created.tabs && created.tabs[0];
    if (tab && tab.id) {
      await new Promise((resolve) => {
        function onUpdated(tabId, changeInfo) {
          if (tabId === tab.id && changeInfo.status === 'complete') {
            chrome.tabs.onUpdated.removeListener(onUpdated);
            resolve();
          }
        }
        chrome.tabs.onUpdated.addListener(onUpdated);

        // Failsafe: resolve after 2s even if we didn't get the event.
        setTimeout(() => {
          try { chrome.tabs.onUpdated.removeListener(onUpdated); } catch (e) {}
          resolve();
        }, 2000);
      });
    }

    return { kind: 'window', windowId: created.id };
  } catch (e) {
    console.error('[background] ensureOffscreen error', e);
    throw e;
  }
}

// Store last recording URL transiently (lost if service worker restarts)
globalThis._lastRecordingUrl = null;
// Track incoming chunk diagnostics per videoId
globalThis._chunkDiag = globalThis._chunkDiag || {};
// Track which videoIds have already had a FINAL processed to avoid duplicates
globalThis._processedFinals = globalThis._processedFinals || new Set();
// Track in-flight uploads per videoId to avoid parallel duplicate uploads; removed after completion
globalThis._inflightUploads = globalThis._inflightUploads || new Set();
// Store last classification/transcript so UI can request it later
globalThis._lastClassification = globalThis._lastClassification || null;

function sendToSenderOrBroadcast(sender, payload) {
  try {
    if (sender && sender.tab && typeof sender.tab.id === 'number') {
      chrome.tabs.sendMessage(sender.tab.id, payload, () => {
        // Intentionally ignore errors like "Receiving end does not exist"
        void chrome.runtime.lastError;
      });
    } else {
      chrome.runtime.sendMessage(payload, () => {
        // Intentionally ignore errors like "Receiving end does not exist"
        void chrome.runtime.lastError;
      });
    }
  } catch (e) {
    // swallow – fire-and-forget UI notifications
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('[background] Received message:', message);

  // Popup can request the last transcript if it missed the runtime message
  if (message && message.action === 'getLastTranscript') {
    try {
      const t = globalThis._lastTranscript || null;
      const lc = globalThis._lastClassification || null;
      sendResponse({
        transcript: t,
        hateLabel: lc ? lc.hateLabel : null,
        hateScore: lc ? lc.hateScore : null,
        method: lc ? lc.method : null,
        videoId: lc ? lc.videoId : null,
      });
    } catch (e) {
      sendResponse({ transcript: null, hateLabel: null, hateScore: null, method: null, videoId: null });
    }
    return true;
  }

  // Rebroadcast recording lifecycle messages from content scripts so UIs (popup)
  // receive them even if they aren't direct message targets.
  if (message && message.action === 'recordingStarted') {
    try { globalThis._isRecording = true; } catch (e) {}
    // Broadcast to all extension contexts (popup, options, etc.)
    try { chrome.runtime.sendMessage({ action: 'recordingStarted', videoId: message.videoId || null }); } catch (e) {}
    return;
  }
  if (message && message.action === 'recordingStopped') {
    try { globalThis._isRecording = false; } catch (e) {}
    try { chrome.runtime.sendMessage({ action: 'recordingStopped', videoId: message.videoId || null }); } catch (e) {}
    return;
  }

  // Allow popup to query current recording state when it opens
  if (message && message.action === 'getRecordingState') {
    try {
      sendResponse({ isRecording: !!globalThis._isRecording });
    } catch (e) {
      sendResponse({ isRecording: false });
    }
    return true;
  }

  if (message.action === 'start') {
    // Reset per-session state so re-running detection in a new session doesn't block uploads
    try {
      globalThis._processedFinals = new Set();
      globalThis._inflightUploads = new Set();
      globalThis._chunkDiag = {};
    } catch (e) { /* ignore */ }
    ensureOffscreen()
      .then(() => {
        // Tell the offscreen page to start capture & recording
        chrome.runtime.sendMessage({ action: 'startCapture' });
        sendResponse({ started: true });
      })
      .catch((err) => sendResponse({ started: false, error: err?.message }));

    return true; // keep channel open for async response
  }

  if (message.action === 'stop') {
    // Tell offscreen to stop. offscreen will send audioData back to runtime
    chrome.runtime.sendMessage({ action: 'stopCapture' });
    sendResponse({ stopped: true });
    return;
  }

  // (Removed) audioSegment handling: we now only send a single final recording per video

  // handle recorded chunks forwarded from offscreen
  if (message.action === 'audioData') {
    try {
      const tCaptureEnd = Date.now(); // when chunk hit background
      // --- Diagnostics: log base64 prefix and chunk counters ---
      try {
        const vid = message.videoId || 'anon';
        const diag = (globalThis._chunkDiag[vid] = globalThis._chunkDiag[vid] || { nonFinal: 0, final: 0, totalBytes: 0 });
        const isFinal = !!message.isFinal;
        const seq = typeof message.seq === 'number' ? message.seq : -1;
        const mime = message.mime || 'audio/webm';
        const b64 = message.b64 || '';
        const sizeApprox = Math.floor(b64.length * 3 / 4); // rough decoded size estimate
        diag.totalBytes += sizeApprox;
        if (isFinal) diag.final += 1; else diag.nonFinal += 1;

        // Log first 200 chars of the base64 (safe preview) and counters
        const preview = b64 ? b64.slice(0, 200) : '(no-b64)';
        console.log('[background][diag] chunk vid=', vid, 'seq=', seq, 'isFinal=', isFinal, 'mime=', mime, 'b64[:200]=', preview, 'len=', b64.length);
        if (isFinal) {
          console.log('[background][diag] FINAL received for vid=', vid, 'counts={ nonFinal:', diag.nonFinal, ', final:', diag.final, ' } totalApproxBytes=', diag.totalBytes);
        }
      } catch (e) {
        // best-effort diagnostics only
      }

      let blob = null;
      if (message.b64) {
        // content script sent base64 payload (most robust). Decode and wrap.
        try {
          const binary = atob(message.b64);
          const len = binary.length;
          const bytes = new Uint8Array(len);
          for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
          blob = new Blob([bytes], { type: message.mime || 'audio/webm' });
        } catch (e) {
          console.error('[background] failed to decode base64 audio', e);
          throw e;
        }
      } else {
        const incoming = message.data;
        // If content script sent an ArrayBuffer (may fail in some platforms), wrap it
        if (incoming && (incoming instanceof ArrayBuffer || ArrayBuffer.isView(incoming))) {
          blob = new Blob([incoming], { type: message.mime || 'audio/webm' });
        } else if (Array.isArray(incoming)) {
          // array of Blobs or chunk objects
          blob = new Blob(incoming, { type: message.mime || 'audio/webm' });
        } else if (incoming && typeof incoming === 'object' && incoming.byteLength) {
          // typed array-like
          blob = new Blob([incoming], { type: message.mime || 'audio/webm' });
        } else {
          // last resort: try to stringify
          try {
            const str = String(incoming);
            blob = new Blob([str], { type: message.mime || 'text/plain' });
          } catch (e) {
            throw new Error('unsupported audio data format');
          }
        }
      }
      // In a service worker context URL.createObjectURL may be unavailable,
      // so send the Blob directly to the popup and let the popup create a
      // URL for playback/download.
  globalThis._lastRecordingBlob = blob;
  console.log('[background] constructed blob size bytes=', blob.size, 'mime=', message.mime || 'audio/webm');

      // notify any UI (popup) that recording is ready; many runtimes have
      // issues transferring Blob objects directly via chrome.runtime.sendMessage
      // so convert the blob to a base64 string and send that instead.
      (async () => {
        try {
          const arrayBuffer = await blob.arrayBuffer();
          const uint8 = new Uint8Array(arrayBuffer);

          // convert to base64 in chunks to avoid call stack issues
          let chunkSize = 0x8000;
          let binary = '';
          for (let i = 0; i < uint8.length; i += chunkSize) {
            const slice = uint8.subarray(i, i + chunkSize);
            binary += String.fromCharCode.apply(null, slice);
          }
          const base64 = btoa(binary);

          sendToSenderOrBroadcast(sender, { action: 'recordingReady', b64: base64, mime: 'audio/webm' });
        } catch (e) {
          console.warn('[background] failed to serialize blob for messaging', e);
          sendToSenderOrBroadcast(sender, { action: 'recordingReady' });
        }
      })();
      sendResponse({ received: true });

      // --- Send to transcription server ONLY for final chunks ---
      // Reduce file spam and avoid uploading partial/short blobs; rely on final flush from content capture.
      if (message.isFinal) {
        // Check if capture was interrupted - if so, skip transcription and show interruption message
        if (message.isInterrupted) {
          console.log('[background] Capture was interrupted, skipping transcription');
          sendToSenderOrBroadcast(sender, { action: 'transcriptionReady', isInterrupted: true, hateLabel: null, hateScore: null });
          return true;
        }
        try {
          // Avoid parallel duplicate uploads for the same videoId; do NOT mark as processed yet
          const vidForDedupe = message.videoId || 'anon';
          if (globalThis._inflightUploads.has(vidForDedupe)) {
            console.warn('[background] FINAL already uploading for vid=', vidForDedupe, '— dropping');
            return true;
          }
          if (globalThis._processedFinals.has(vidForDedupe)) {
            console.warn('[background] FINAL already processed for vid=', vidForDedupe, '— dropping');
            return true;
          }
          globalThis._inflightUploads.add(vidForDedupe);
        } catch (e) { /* ignore */ }
        (async () => {
          try {
            // notify UI that transcription is starting
            try { sendToSenderOrBroadcast(sender, { action: 'transcriptionStarted' , videoId: message.videoId || null}); } catch (e) {}
            console.log('[background] Starting upload for vid=', message.videoId || 'anon', 'size=', blob.size);
            const tUploadStart = Date.now();
            const recName = `recording-${message.videoId || 'anon'}-${Date.now()}.webm`;
            const form = new FormData();
            form.append('file', blob, recName);
            if (message.videoId) form.append('videoId', message.videoId);
            form.append('final', '1');

            const resp = await fetch('http://127.0.0.1:5000/transcribe', {
              method: 'POST',
              body: form,
            });
            const tResponse = Date.now();

            if (!resp.ok) {
              const text = await resp.text();
              console.warn('[background] Transcription server error:', resp.status, text);
              chrome.runtime.sendMessage({ action: 'transcriptionReady', error: 'Transcription server error: ' + resp.status });
              return;
            }

            const data = await resp.json();
            const tJsonDone = Date.now();
            const transcript = data.transcript || data.text || '';
            console.log('[background] Received transcript:', transcript);
            if (Object.prototype.hasOwnProperty.call(data, 'hateLabel') || Object.prototype.hasOwnProperty.call(data, 'hateScore')) {
              console.log('[background] Classification:', 'label=', data.hateLabel || null, 'score=', data.hateScore ?? null, 'method=', data.method || null, 'hateModel=', data.hateModel || null);
            }
            // remember last transcript so popup can request it later
            globalThis._lastTranscript = transcript;
            const outMsg = {
              action: 'transcriptionReady',
              transcript,
              videoId: message.videoId || null,
              sequenceIndex: data.sequenceIndex,
              hateLabel: data.hateLabel || null,
              hateScore: data.hateScore || null,
              method: data.method || null,
              hateModel: data.hateModel || null
            };
            // Performance report (client-side portion)
            const perf = {
              audioCaptureToBackgroundMs: (typeof message.tCaptureStartMs === 'number') ? (tCaptureEnd - message.tCaptureStartMs) : null,
              uploadMs: tResponse - tUploadStart,
              responseJsonParseMs: tJsonDone - tResponse,
              serverTimingsMs: data.timingsMs || null,
              serverEndToEndMs: data.endToEndMs || null,
              audioDurationSec: data.duration || null,
            };
            console.log('[background][perf]', perf);
            // remember last full classification result
            try {
              globalThis._lastClassification = {
                transcript,
                hateLabel: outMsg.hateLabel,
                hateScore: outMsg.hateScore,
                method: outMsg.method,
                videoId: outMsg.videoId
              };
            } catch (e) { /* ignore */ }
            sendToSenderOrBroadcast(sender, outMsg);
            try { globalThis._processedFinals.add(message.videoId || 'anon'); } catch (e) {}
          } catch (e) {
            console.warn('[background] Failed to call transcription server:', e);
            sendToSenderOrBroadcast(sender, { action: 'transcriptionReady', error: String(e), videoId: message.videoId || null });
          }
          finally {
            try { globalThis._inflightUploads.delete(message.videoId || 'anon'); } catch (e) {}
          }
        })();
      } else {
        console.log('[background] Skipping server upload for non-final chunk (reducing file spam)');
      }
    } catch (e) {
      console.error('[background] audioData handling error', e);
      sendResponse({ received: false, error: e.message });
    }

    return true;
  }
});
