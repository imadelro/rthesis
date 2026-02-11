chrome.runtime.onMessage.addListener(async (message) => {
  if (message.action === "gotStreamId" && message.streamId) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          mandatory: {
            chromeMediaSource: "tab",
            chromeMediaSourceId: message.streamId
          }
        }
      });

      console.log("Audio stream captured:", stream);
      // Here you can route this audio to Web Audio API, Whisper, etc.
    } catch (err) {
      console.error("Error accessing audio stream:", err);
    }
  }
});
