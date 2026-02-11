// Offscreen recorder: capture tab audio, play it to the user (connect to
// AudioContext destination), record with MediaRecorder, and send chunks back
// to the background via chrome.runtime.sendMessage({ action: 'audioData', data }).

class Recorder {
    constructor(onChunksAvailable) {
        this.chunks = [];
        this.active = false;
        this.callback = onChunksAvailable;
        this.context = null; // created lazily
        this.source = null;
        this.recorder = null;
    }

    start(stream) {
        if (this.active) {
            throw new Error('recorder is already running');
        }

        // create AudioContext lazily
        if (!this.context) {
            this.context = new (window.AudioContext || window.webkitAudioContext)();
        }

        // Reconnect the stream to actual output so the user hears the audio
        this.source = this.context.createMediaStreamSource(stream);
        this.source.connect(this.context.destination);

        try {
            this.recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        } catch (e) {
            // fallback without mimeType
            this.recorder = new MediaRecorder(stream);
        }

        this.recorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                this.chunks.push(event.data);
            }
        };

        this.recorder.onstop = () => {
            try {
                // stop the capture track to free resources
                const tracks = stream.getAudioTracks();
                if (tracks && tracks[0]) tracks[0].stop();
            } catch (e) {
                console.warn('Error stopping audio track:', e);
            }

            this.callback([...this.chunks]);

            // clear chunks asynchronously
            setTimeout(() => {
                this.chunks = [];
            }, 0);

            this.active = false;
        };

        this.active = true;
        // Start without timeslice to collect full buffer, or pass a timeslice like 3000
        this.recorder.start();
        console.log('[capture] Recorder started');
    }

    stop() {
        if (!this.active) {
            throw new Error('recorder is already stop');
        }
        this.recorder.stop();
    }
}

let recorder = null;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'startCapture') {
        startAudioCapture();
    } else if (request.action === 'stopCapture') {
        stopAudioCapture();
    }
});

function startAudioCapture() {
    chrome.tabCapture.capture({ audio: true, video: false }, (stream) => {
        if (!stream) {
            console.error('[capture] Audio capture failed:', chrome.runtime.lastError);
            return;
        }

        console.log('[capture] Audio capture started');

        recorder = new Recorder((chunks) => {
            console.log('[capture] Sending audioData, chunks:', chunks.length);
            chrome.runtime.sendMessage({ action: 'audioData', data: chunks });
        });

        try {
            recorder.start(stream);
        } catch (e) {
            console.error('[capture] Recorder start failed:', e);
            try {
                // ensure stream tracks stopped on failure
                const tracks = stream.getAudioTracks();
                if (tracks && tracks[0]) tracks[0].stop();
            } catch (err) {}
        }
    });
}

function stopAudioCapture() {
    if (recorder && recorder.active) {
        recorder.stop();
        console.log('[capture] Stop requested');
    } else {
        console.warn('[capture] No active recorder to stop');
    }
}
