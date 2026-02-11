// content_capture.js — continuous per-video recording (no Web Audio)
// Continuously detects the active TikTok <video>, records its audio in chunks,
// and finalizes when switching or stopping. Chunks are sent immediately; the
// last chunk is flagged isFinal for per-video merge/transcription server-side.

let mediaRecorder = null;
let stopFn = null;
let singleShotCleanup = null;
let stoppingWithTail = false; // guard against multiple tail stops
let isInterrupted = false; // flag when stop button is pressed
const TAIL_CAPTURE_MS = 700; // extra time to allow final audio frames to flush

// Continuous detection state and tuning
let isDetecting = false;
let currentVideoEl = null;
let currentVideoId = null;
let recordedVideoIds = new Set(); // avoid re-recording same video id in one session
let lastRecordedVideoId = null; // last video id we actually recorded
let seenDifferentSinceLastRecord = true; // true once we've observed a different video after the last record
let activePollTimer = null;
let markNextChunkFinal = false;
let chunkSeq = 0;
let _autoVidCounter = 0; // unique id counter for elements lacking stable src/duration

// Recording strategy: Avoid uploading timeslice fragments. Record continuous stream
// without a timeslice and send a single, final Blob when stopping. This ensures
// the resulting WebM has a proper EBML header and is transcodable by ffmpeg.
const CHUNK_MS = null; // no periodic timeslice; collect one self-contained blob
const VISIBILITY_THRESHOLD = 0.35; // relax to 35% of viewport to catch letterboxed videos
const PROGRESS_DELTA = 0.05; // ~50ms advancement to count as progressing
const POLL_MS = 500; // faster polling improves detection responsiveness
const lastTime = new WeakMap();

// Debug controls for selection logging (disabled by default)
const DEBUG_PICK = false;
function logPick(...args) { if (DEBUG_PICK) console.log(...args); }
function warnPick(...args) { if (DEBUG_PICK) console.warn(...args); }

function pickCurrentVideo() {
	const vids = Array.from(document.querySelectorAll('video'));
	if (!vids.length) { console.warn('[content] no <video> elements found'); return null; }
	const vpW = window.innerWidth || 0;
	const vpH = window.innerHeight || 0;
	const viewportArea = Math.max(1, vpW * vpH);

	// Collect candidates with metrics
	const candidates = [];
	for (const v of vids) {
		try {
			const rect = v.getBoundingClientRect();
			const interW = Math.max(0, Math.min(rect.right, vpW) - Math.max(rect.left, 0));
			const interH = Math.max(0, Math.min(rect.bottom, vpH) - Math.max(rect.top, 0));
			const interArea = interW * interH;
			const ratio = interArea / viewportArea;
			const prevT = lastTime.get(v) ?? v.currentTime;
			const nowT = v.currentTime || 0;
			const dt = Math.abs(nowT - (prevT || 0));
			lastTime.set(v, nowT);
			const playingish = !v.paused && !v.ended && (v.readyState >= 2) && (v.playbackRate || 0) >= 0.9;
			const progressing = dt >= PROGRESS_DELTA && playingish;
			candidates.push({ el: v, ratio, dt, playingish, progressing });
		} catch (e) {}
	}
	// Sort by visibility desc
	candidates.sort((a, b) => b.ratio - a.ratio);

	// Prefer visible + progressing
	const prog = candidates.filter(c => c.ratio >= VISIBILITY_THRESHOLD && c.progressing);
	if (prog.length) {
		const chosen = prog[0];
		logPick('[content] pickCurrentVideo progressing ratio=', chosen.ratio.toFixed(3), 'dt=', chosen.dt.toFixed(3));
		return chosen.el;
	}
	// Then visible + playing-ish (even if dt small)
	const playish = candidates.filter(c => c.ratio >= VISIBILITY_THRESHOLD && c.playingish);
	if (playish.length) {
		const chosen = playish[0];
		logPick('[content] pickCurrentVideo playing-ish ratio=', chosen.ratio.toFixed(3), 'dt=', chosen.dt.toFixed(3));
		return chosen.el;
	}
	// Finally, top visible element as a last resort
	if (candidates.length) {
		const chosen = candidates[0];
		warnPick('[content] pickCurrentVideo fallback ratio=', chosen.ratio.toFixed(3), 'dt=', chosen.dt.toFixed(3), 'playingish=', chosen.playingish);
		return chosen.el;
	}
	warnPick('[content] pickCurrentVideo found no suitable candidate');
	return null;
}

function gracefulStopRecording() {
	try {
		if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
		if (stoppingWithTail) return; // already scheduled
		stoppingWithTail = true;
		console.log('[content] scheduling tail stop in', TAIL_CAPTURE_MS, 'ms');
		// Request current buffered data now, then allow a tail interval before final stop.
		try { mediaRecorder.requestData(); } catch (e) { /* ignore */ }
		setTimeout(() => {
			try {
				if (mediaRecorder && mediaRecorder.state !== 'inactive') {
					console.log('[content] tail stop executing');
					// mark next chunk as final so ondataavailable tags it
					markNextChunkFinal = true;
					mediaRecorder.stop();
				}
			} catch (e) { console.warn('[content] tail stop error', e); }
		}, TAIL_CAPTURE_MS);
	} catch (e) { /* ignore */ }
}

async function startCapture() {
	if (isDetecting) {
		console.warn('[content] already detecting');
		return { started: false, reason: 'already-detecting' };
	}
	// reset session state
	recordedVideoIds = new Set();
	isDetecting = true;
	isInterrupted = false; // reset interrupt flag for new capture session
	const loop = async () => {
		if (!isDetecting) return;
		try {
			const candidate = pickCurrentVideo();
			if (candidate) {
				const vidId = getOrCreateVideoId(candidate);
				// If we see a different video than the last recorded, mark that fact
				if (vidId !== lastRecordedVideoId) seenDifferentSinceLastRecord = true;

				if (currentVideoEl === candidate && mediaRecorder && mediaRecorder.state !== 'inactive') {
					// already recording this video
				} else {
					// If this is the same as the last recorded video and we haven't
					// seen a different video since, skip re-recording.
					if (vidId === lastRecordedVideoId && !seenDifferentSinceLastRecord) {
						// Ensure any recorder for a different video is stopped so we move on
						if (mediaRecorder && mediaRecorder.state !== 'inactive' && currentVideoEl !== candidate) {
							gracefulStopRecording();
						}
						// Do not start a new recording for this repeated video
					} else {
						// Start recording for this (new or allowed) video
						if (mediaRecorder && mediaRecorder.state !== 'inactive') gracefulStopRecording();
						const ok = await startRecordingFor(candidate);
						if (ok) {
							recordedVideoIds.add(vidId);
							lastRecordedVideoId = vidId;
							seenDifferentSinceLastRecord = false; // reset until we see a different video
						}
					}
				}
			} else {
				// No candidate found (scrolled away) — stop any active recording
				if (mediaRecorder && mediaRecorder.state !== 'inactive') gracefulStopRecording();
			}
		} catch (e) {}
	};
	await loop();
	activePollTimer = setInterval(loop, POLL_MS);
	return { started: true, mode: 'continuous' };
}

function getOrCreateVideoId(sourceEl) {
	// Build a lightweight signature from currentSrc/src and duration to detect content changes
	const src = sourceEl.currentSrc || sourceEl.src || '';
	const dur = isFinite(sourceEl.duration) && sourceEl.duration > 0 ? Math.round(sourceEl.duration * 10) : 0; // deciseconds
	const sigNow = `${src}|${dur}`;
	const sigPrev = sourceEl.getAttribute('data-ext-vid-sig');
	let vidId = sourceEl.getAttribute('data-ext-vid-id');

	// If we don't have an id yet, or if the signature changed (same DOM <video> reused for a new clip), then mint a new id
	if (!vidId || sigPrev !== sigNow) {
		let newId;
		if (src) {
			// Stable id based on src hash + duration hint
			let h = 0; for (let i = 0; i < src.length; i++) { h = ((h << 5) - h) + src.charCodeAt(i); h |= 0; }
			newId = 'vid-' + Math.abs(h).toString(36) + (dur ? '-' + dur : '');
		} else {
			// Generate a per-element unique id when src/duration are not yet available
			_autoVidCounter += 1;
			newId = 'vid-auto-' + Date.now().toString(36) + '-' + _autoVidCounter.toString(36);
		}
		vidId = newId;
		try {
			sourceEl.setAttribute('data-ext-vid-id', vidId);
			sourceEl.setAttribute('data-ext-vid-sig', sigNow);
		} catch (e) {}
	}
	return vidId;
}

async function startRecordingFor(sourceEl) {
	const vidId = getOrCreateVideoId(sourceEl);
	currentVideoId = vidId;
	currentVideoEl = sourceEl;

	const stream = sourceEl.captureStream ? sourceEl.captureStream() : null;
	if (!stream) {
		console.warn('[content] captureStream unavailable on <video>');
		try { chrome.runtime.sendMessage({ action: 'captureError', reason: 'captureStream-unavailable' }); } catch (e) {}
		return false;
	}

	const audioTracks = stream.getAudioTracks();
	if (!audioTracks || audioTracks.length === 0) {
		console.warn('[content] video has no audio tracks');
		try { chrome.runtime.sendMessage({ action: 'captureError', reason: 'no-audio-tracks' }); } catch (e) {}
		return false;
	}

	// Audio-only recording: drop video track to avoid VP8 encoding overhead.
	// This reduces file size, upload time, and server processing.
	let recorderStream = new MediaStream(stream.getAudioTracks());

	// Prefer explicit Opus audio MIME; fallback to generic audio/webm if needed.
	let preferredMime = 'audio/webm; codecs=opus';
	if (!MediaRecorder.isTypeSupported(preferredMime)) {
		preferredMime = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '';
	}
	try {
		mediaRecorder = preferredMime ? new MediaRecorder(recorderStream, { mimeType: preferredMime }) : new MediaRecorder(recorderStream);
	} catch (e) {
		try { mediaRecorder = new MediaRecorder(recorderStream); } catch (err) { console.error('[content] MediaRecorder failed', err); return false; }
	}

	chunkSeq = 0;
	markNextChunkFinal = false;
	stoppingWithTail = false;

	let primeTimer = null;
	let gotAnyData = false;
	let fallbackTimer = null;

	// Collect all parts and only send a single final Blob on stop.
	const allParts = [];
	mediaRecorder.ondataavailable = (ev) => {
		try {
			if (!ev.data || ev.data.size === 0) return;
			gotAnyData = true;
			allParts.push(ev.data);
		} catch (e) { console.warn('[content] ondataavailable error', e); }
	};

	function sendBlob(blob, seq, isFinal) {
		if (!blob || blob.size === 0) return;
		const fr = new FileReader();
		fr.onload = () => {
			try {
				const dataUrl = String(fr.result || '');
				// Data URL can contain a comma in the codecs list (e.g., vp8,opus). Always slice after 'base64,' or the last comma.
				let b64 = '';
				const tag = 'base64,';
				const idx = dataUrl.indexOf(tag);
				if (idx !== -1) {
					b64 = dataUrl.substring(idx + tag.length);
				} else {
					const lastComma = dataUrl.lastIndexOf(',');
					b64 = lastComma >= 0 ? dataUrl.slice(lastComma + 1) : dataUrl;
				}
				const tCaptureStartMs = Date.now();
				chrome.runtime.sendMessage({
					action: 'audioData',
					b64,
					mime: blob.type || preferredMime || 'audio/webm',
					videoId: currentVideoId,
					isFinal,
					seq,
					tCaptureStartMs,
					isInterrupted: isInterrupted, // pass interrupt flag to background
				});
			} catch (e) { console.error('[content] send chunk failed', e); }
		};
		fr.onerror = (err) => console.error('[content] FileReader chunk error', err);
		fr.readAsDataURL(blob);
	}

	mediaRecorder.onstop = () => {
		console.log('[content] recorder stopped for', currentVideoId);
		mediaRecorder = null;
		try { if (primeTimer) clearTimeout(primeTimer); } catch (e) {}
		try { if (fallbackTimer) clearTimeout(fallbackTimer); } catch (e) {}
		// Emit a single final blob assembled from all collected parts (if any)
		try {
			if (allParts.length) {
				const finalBlob = new Blob(allParts, { type: preferredMime || 'audio/webm' });
				sendBlob(finalBlob, chunkSeq++, true);
			}
		} catch (e) { console.warn('[content] failed to assemble final blob', e); }
		// No Web Audio nodes to clean up in this mode
		try { if (typeof singleShotCleanup === 'function') singleShotCleanup(); } catch (e) {}
		singleShotCleanup = null;
	};

	try { CHUNK_MS ? mediaRecorder.start(CHUNK_MS) : mediaRecorder.start(); } catch (e) { try { CHUNK_MS ? mediaRecorder.start(CHUNK_MS) : mediaRecorder.start(); } catch (err) { console.error('[content] mediaRecorder.start failed', err); return false; } }
	console.log('[content] MediaRecorder started (audio-only) for', vidId, 'chunkMs=', CHUNK_MS, 'mime=', mediaRecorder.mimeType);
	// Notify in-page UI and any popup that recording has actually started
	try {
		window.dispatchEvent(new CustomEvent('EXTENSION_RECORDING_STARTED', { detail: { videoId: vidId } }));
		chrome.runtime.sendMessage({ action: 'recordingStarted', videoId: vidId });
	} catch (e) { /* best-effort */ }

	// Auto-stop when the video ends or after a reasonable max duration
	try {
		const onEndedOnce = () => { try { if (mediaRecorder && mediaRecorder.state !== 'inactive') gracefulStopRecording(); } catch (e) {} };
		sourceEl.addEventListener('ended', onEndedOnce, { once: true });
		const expectedMs = (isFinite(sourceEl.duration) && sourceEl.duration > 0) ? sourceEl.duration * 1000 : 120000;
		const maxMs = Math.min(Math.max(expectedMs, 5000), 5 * 60 * 1000);
		const timer = setTimeout(onEndedOnce, maxMs);
		// Prime an early flush so very short stays still produce an early part (we'll still send only final)
		primeTimer = setTimeout(() => {
			try { if (mediaRecorder && mediaRecorder.state !== 'inactive' && !gotAnyData) mediaRecorder.requestData(); } catch (e) {}
		}, 1500);

		// Fallback: if no data after a few seconds, recreate recorder audio-only
		fallbackTimer = setTimeout(() => {
			try {
				if (gotAnyData || !mediaRecorder || mediaRecorder.state !== 'recording') return;
				console.warn('[content] no data yet; recreating recorder with audio-only MIME');
				mediaRecorder.ondataavailable = null;
				try { mediaRecorder.stop(); } catch (e) {}
				// Rebuild audio-only MediaRecorder
				const audioOnlyStream = new MediaStream(stream.getAudioTracks());
				let fallbackMime = 'audio/webm; codecs=opus';
				if (!MediaRecorder.isTypeSupported(fallbackMime)) {
					fallbackMime = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '';
				}
				try {
					mediaRecorder = fallbackMime ? new MediaRecorder(audioOnlyStream, { mimeType: fallbackMime }) : new MediaRecorder(audioOnlyStream);
				} catch (e2) {
					mediaRecorder = new MediaRecorder(audioOnlyStream);
				}
				// Reset state and handlers
				chunkSeq = 0; markNextChunkFinal = false; stoppingWithTail = false; gotAnyData = false;
				const aoParts = [];
				mediaRecorder.ondataavailable = (ev) => {
					try {
						if (!ev.data || ev.data.size === 0) return;
						gotAnyData = true;
						aoParts.push(ev.data);
					} catch (e) { console.warn('[content] ondataavailable error (fallback)', e); }
				};
				mediaRecorder.onstop = () => {
					console.log('[content] recorder stopped for', currentVideoId);
					// Notify UI that recording stopped (processing will follow)
					try { window.dispatchEvent(new CustomEvent('EXTENSION_RECORDING_STOPPED', { detail: { videoId: currentVideoId } })); } catch (e) {}
					mediaRecorder = null;
					try { if (primeTimer) clearTimeout(primeTimer); } catch (e) {}
					try { if (fallbackTimer) clearTimeout(fallbackTimer); } catch (e) {}
					try {
						if (aoParts.length) {
							const finalBlob = new Blob(aoParts, { type: 'audio/webm' });
							sendBlob(finalBlob, chunkSeq++, true);
						}
					} catch (e) {}
					try { if (typeof singleShotCleanup === 'function') singleShotCleanup(); } catch (e) {}
					singleShotCleanup = null;
				};
				try { CHUNK_MS ? mediaRecorder.start(CHUNK_MS) : mediaRecorder.start(); } catch (e3) { try { CHUNK_MS ? mediaRecorder.start(CHUNK_MS) : mediaRecorder.start(); } catch (e4) { console.error('[content] fallback mediaRecorder.start failed', e4); } }
			} catch (e) { /* ignore */ }
		}, 3000);

		singleShotCleanup = () => {
			try { sourceEl.removeEventListener('ended', onEndedOnce); } catch (e) {}
			try { clearTimeout(timer); } catch (e) {}
			try { if (primeTimer) clearTimeout(primeTimer); } catch (e) {}
		};
	} catch (e) {}

	return true;
}

function stopCapture() {
	isDetecting = false;
	isInterrupted = true; // mark as interrupted so we don't transcribe
	if (activePollTimer) { try { clearInterval(activePollTimer); } catch (e) {} activePollTimer = null; }
	if (mediaRecorder && mediaRecorder.state !== 'inactive') { gracefulStopRecording(); return { stopped: true }; }
	return { stopped: true };
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
	if (!msg || !msg.action) return;
	if (msg.action === 'startCapture') { startCapture().then(sendResponse).catch(e => sendResponse({ started: false, error: String(e) })); return true; }
	if (msg.action === 'stopCapture') { sendResponse(stopCapture()); return; }
	if (msg.action === 'recordingReady') {
		try {
			window.dispatchEvent(new CustomEvent('EXTENSION_STATUS', { detail: { statusText: 'Processing — detecting language...' } }));
			window.dispatchEvent(new CustomEvent('EXTENSION_RECORDING_READY', { detail: { b64: msg.b64, mime: msg.mime } }));
		} catch (e) {}
		return;
	}
	if (msg.action === 'transcriptionReady') {
		try {
			if (msg.error) {
				window.dispatchEvent(new CustomEvent('EXTENSION_STATUS', { detail: { statusText: 'Transcription error' } }));
				window.dispatchEvent(new CustomEvent('EXTENSION_TRANSCRIPT', { detail: { error: msg.error, videoId: msg.videoId || null, hateLabel: (Object.prototype.hasOwnProperty.call(msg, 'hateLabel') ? msg.hateLabel : null), hateScore: (Object.prototype.hasOwnProperty.call(msg, 'hateScore') ? msg.hateScore : null), method: msg.method || null, hateModel: (Object.prototype.hasOwnProperty.call(msg, 'hateModel') ? msg.hateModel : null) } }));
			} else if (msg.isInterrupted) {
				window.dispatchEvent(new CustomEvent('EXTENSION_STATUS', { detail: { statusText: 'Detection interrupted — process stopped' } }));
				window.dispatchEvent(new CustomEvent('EXTENSION_TRANSCRIPT', { detail: {
					isInterrupted: true,
					videoId: msg.videoId || null,
					hateLabel: null,
					hateScore: null
				} }));
			} else {
				window.dispatchEvent(new CustomEvent('EXTENSION_STATUS', { detail: { statusText: 'Ready' } }));
				// Preserve falsy numeric scores (e.g., 0.0) and include label if present
				window.dispatchEvent(new CustomEvent('EXTENSION_TRANSCRIPT', { detail: {
					transcript: msg.transcript,
					videoId: msg.videoId || null,
					hateLabel: (Object.prototype.hasOwnProperty.call(msg, 'hateLabel') ? msg.hateLabel : null),
					hateScore: (Object.prototype.hasOwnProperty.call(msg, 'hateScore') ? msg.hateScore : null),
					method: msg.method || null,
					hateModel: (Object.prototype.hasOwnProperty.call(msg, 'hateModel') ? msg.hateModel : null)
				} }));
			}
		} catch (e) {}
		return;
	}
	if (msg.action === 'transcriptionStarted') {
		try { window.dispatchEvent(new CustomEvent('EXTENSION_STATUS', { detail: { statusText: 'Detecting language...' } })); } catch (e) {}
		return;
	}
});

// Optional manual API
window.__contentCapture = { startCapture, stopCapture };

// In-page UI integration (if content_ui.js dispatches these)
window.addEventListener('EXTENSION_START_CAPTURE', () => { try { startCapture(); window.dispatchEvent(new CustomEvent('EXTENSION_STATUS', { detail: { statusText: 'Recording (auto)' } })); } catch (e) {} });
window.addEventListener('EXTENSION_STOP_CAPTURE', () => { const res = stopCapture(); window.dispatchEvent(new CustomEvent('EXTENSION_STATUS', { detail: { statusText: res.stopped ? 'Stopped' : 'Stop failed' } })); });

