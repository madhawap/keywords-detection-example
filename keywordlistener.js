// Load a transfer-learning keyword model (saved by trainkeyword.html) from
// IndexedDB and listen for those keywords via the microphone. The library's
// listen() already handles debouncing through probabilityThreshold and
// suppressionTimeMillis, so the UI logic here stays very small.

const BG_NOISE = '_background_noise_';

// Only fire a detection when this confident. Raise to reduce false positives,
// lower to catch quieter utterances.
const PROBABILITY_THRESHOLD = 0.75;

// Minimum gap (ms) between consecutive detections of the same word. The
// library enforces this internally — prevents one spoken word counting twice.
const SUPPRESSION_MS = 1000;

// How long (ms) the green highlight stays on after a detection.
const HIGHLIGHT_MS = 600;

const LOCAL_MODEL_URL =
  new URL('vendor/speech-commands-model/model.json', location.href).href;
const LOCAL_METADATA_URL =
  new URL('vendor/speech-commands-model/metadata.json', location.href).href;

let baseRecognizer;
let transfer;
let keywordLabels = []; // the non-noise labels, in transfer.wordLabels() order.
let counts = {};         // label -> number of detections.
let highlightTimer;

function setStatus(msg) { document.getElementById('status').textContent = msg; }

// Build one label box per keyword (skipping the background-noise class).
function renderLabels() {
  const container = document.getElementById('labels');
  container.innerHTML = '';
  counts = {};
  keywordLabels.forEach((label, i) => {
    counts[label] = 0;
    const div = document.createElement('div');
    div.className = 'label';
    div.id = 'label-' + i;
    div.innerHTML = `${label}<span class="count" id="count-${i}">0</span>`;
    container.appendChild(div);
  });
  document.getElementById('known').innerHTML =
    `Loaded vocabulary: ${keywordLabels.map(l => `<code>${l}</code>`).join(' ')}`;
}

function resetCounts() {
  keywordLabels.forEach((label, i) => {
    counts[label] = 0;
    document.getElementById('count-' + i).textContent = '0';
  });
}

function flash(labelIdx) {
  keywordLabels.forEach((_, i) => {
    document.getElementById('label-' + i).classList.toggle('active', i === labelIdx);
  });
  clearTimeout(highlightTimer);
  highlightTimer = setTimeout(() => {
    document.getElementById('label-' + labelIdx).classList.remove('active');
  }, HIGHLIGHT_MS);
}

async function loadModel() {
  const modelFile = document.getElementById('modelFile').files[0];
  const weightsFile = document.getElementById('weightsFile').files[0];
  const metadataFile = document.getElementById('metadataFile').files[0];

  if (!modelFile || !weightsFile || !metadataFile) {
    setStatus('Please pick all three files (.json, .bin, .metadata.json).');
    return;
  }

  let metadata;
  try {
    metadata = JSON.parse(await metadataFile.text());
  } catch (err) {
    setStatus(`Metadata JSON is invalid: ${err.message}`);
    return;
  }
  if (!Array.isArray(metadata.wordLabels) || metadata.wordLabels.length === 0) {
    setStatus('Metadata file is missing "wordLabels".');
    return;
  }
  const name = metadata.modelName || 'loaded-model';

  setStatus('Loading pretrained base model…');
  if (!baseRecognizer) {
    baseRecognizer = speechCommands.create(
      'BROWSER_FFT', undefined, LOCAL_MODEL_URL, LOCAL_METADATA_URL);
    await baseRecognizer.ensureModelLoaded();
  }

  if (transfer && transfer.isListening()) await transfer.stopListening();

  transfer = baseRecognizer.createTransfer(name);
  try {
    // tf.io.browserFiles handles model topology + weights from user-picked files.
    await transfer.load(tf.io.browserFiles([modelFile, weightsFile]));
  } catch (err) {
    setStatus(`Failed to load model files: ${err.message}`);
    return;
  }
  // The library's load(handler) doesn't populate the vocabulary when given a
  // custom IOHandler, so inject it ourselves from the metadata sidecar.
  transfer.words = metadata.wordLabels;

  keywordLabels = transfer.wordLabels().filter(l => l !== BG_NOISE);
  renderLabels();

  document.getElementById('listen').disabled = false;
  document.getElementById('reset').disabled = false;
  setStatus(`Loaded "${name}". Press Listen to start detection.`);
}

async function toggleListen() {
  if (transfer.isListening()) {
    await transfer.stopListening();
    document.getElementById('listen').textContent = 'Listen';
    setStatus('Stopped.');
    return;
  }

  document.getElementById('listen').textContent = 'Stop';
  setStatus('Listening…');

  // The library calls our callback at most once per detection (respecting the
  // suppression window), and only when a non-noise class clears the threshold.
  await transfer.listen(
    async result => {
      const scores = Array.from(result.scores);
      const labels = transfer.wordLabels();
      let topIdx = 0;
      for (let i = 1; i < scores.length; i++) {
        if (scores[i] > scores[topIdx]) topIdx = i;
      }
      const topLabel = labels[topIdx];
      if (topLabel === BG_NOISE) return;

      const keywordIdx = keywordLabels.indexOf(topLabel);
      if (keywordIdx === -1) return;

      counts[topLabel]++;
      document.getElementById('count-' + keywordIdx).textContent = counts[topLabel];
      flash(keywordIdx);
    },
    {
      probabilityThreshold: PROBABILITY_THRESHOLD,
      invokeCallbackOnNoiseAndUnknown: false,
      overlapFactor: 0.5,
      suppressionTimeMillis: SUPPRESSION_MS,
    },
  );
}
