// Transfer-learning keyword trainer.
// Uses speechCommands.createTransfer() to build a classifier on top of the
// pretrained 18-word BROWSER_FFT model — so we get strong speech features for
// free and only need to train a small head with ~15-25 recordings per keyword.

const BG_NOISE = '_background_noise_'; // special label the library recognises.

// Local model files so this works offline. speech-commands requires absolute URLs.
const LOCAL_MODEL_URL =
  new URL('vendor/speech-commands-model/model.json', location.href).href;
const LOCAL_METADATA_URL =
  new URL('vendor/speech-commands-model/metadata.json', location.href).href;

let baseRecognizer; // wraps the pretrained 18-word model.
let transfer;       // our custom transfer recognizer (contains its own examples/head).
let trained = false;
let lastEpochMsg = '';

function setStatus(msg) { document.getElementById('status').textContent = msg; }

function toggleButtons(enable) {
  document.querySelectorAll('button').forEach(b => b.disabled = !enable);
  if (enable && !trained) document.getElementById('save').disabled = true;
}

// Render example counts using the transfer recognizer as the source of truth.
function refreshCounts() {
  const counts = transfer.countExamples();
  const labels = Object.keys(counts);
  if (labels.length === 0) {
    document.getElementById('counts').textContent = 'No examples yet.';
    return;
  }
  const parts = labels
    .sort()
    .map(l => `${l === BG_NOISE ? 'noise' : l}: ${counts[l]}`);
  document.getElementById('counts').textContent = parts.join('  |  ');
}

// Record one 1-second example for the typed keyword name.
// collectExample() captures exactly one spectrogram window — equivalent to the
// "say the word once" pattern the user asked for.
async function recordKeyword(idx) {
  const nameInput = document.getElementById('kw' + idx);
  const name = (nameInput.value || '').trim();
  if (!name) { setStatus(`Type a name for keyword ${idx} first.`); return; }
  if (name === BG_NOISE) { setStatus('That name is reserved — use the Noise button.'); return; }

  const btn = document.getElementById('rec' + idx);
  btn.classList.add('recording');
  setStatus(`Recording "${name}" — say it now (1 second window)…`);
  try {
    await transfer.collectExample(name);
  } catch (err) {
    setStatus(`Failed to record: ${err.message}`);
    return;
  } finally {
    btn.classList.remove('recording');
  }
  setStatus(`Captured one example of "${name}".`);
  refreshCounts();
  // Lock the model name once we have data, so save/load names stay consistent.
  document.getElementById('modelName').disabled = true;
}

async function recordNoise() {
  const btn = document.getElementById('recN');
  btn.classList.add('recording');
  setStatus('Recording background noise — stay quiet or let typical sounds play…');
  try {
    await transfer.collectExample(BG_NOISE);
  } catch (err) {
    setStatus(`Failed to record noise: ${err.message}`);
    return;
  } finally {
    btn.classList.remove('recording');
  }
  setStatus('Captured one noise example.');
  refreshCounts();
  document.getElementById('modelName').disabled = true;
}

async function train() {
  const counts = transfer.countExamples();
  const labels = Object.keys(counts);
  const keywordLabels = labels.filter(l => l !== BG_NOISE);
  if (keywordLabels.length < 2) {
    setStatus('Record examples for at least 2 different keywords before training.');
    return;
  }
  if (!counts[BG_NOISE]) {
    setStatus('Please record a few noise examples too (press Record Noise).');
    return;
  }

  const epochsInput = parseInt(document.getElementById('epochs').value, 10);
  const epochs = Number.isFinite(epochsInput) && epochsInput > 0 ? epochsInput : 25;

  toggleButtons(false);
  setStatus('Training…');
  try {
    await transfer.train({
      epochs,
      callback: {
        onEpochEnd: (epoch, logs) => {
          lastEpochMsg =
            `Epoch ${epoch + 1}/${epochs}  —  ` +
            `acc: ${(logs.acc * 100).toFixed(1)}%  loss: ${logs.loss.toFixed(3)}`;
          setStatus(lastEpochMsg);
        },
      },
    });
  } catch (err) {
    setStatus(`Training failed: ${err.message}`);
    toggleButtons(true);
    return;
  }
  trained = true;
  toggleButtons(true);
  document.getElementById('save').disabled = false;
  setStatus(lastEpochMsg + '   — done. Click Save Model next.');
}

// Save to disk as three files:
//   <name>.json              - TF.js model topology + weights manifest
//   <name>.weights.bin       - TF.js model weights
//   <name>.metadata.json     - vocabulary (wordLabels + modelName)
//
// The library's save(url) only writes the first two; we write the third
// ourselves because the library normally keeps the vocabulary in localStorage,
// which isn't portable across machines.
async function saveModel() {
  if (!trained) { setStatus('Train the model before saving.'); return; }
  const name = (document.getElementById('modelName').value || 'my-keywords').trim();

  try {
    await transfer.save('downloads://' + name);
  } catch (err) {
    setStatus(`Model save failed: ${err.message}`);
    return;
  }

  downloadJSON(name + '.metadata.json', {
    modelName: name,
    wordLabels: transfer.wordLabels(),
  });

  setStatus(
    `Saved three files to Downloads: ${name}.json, ${name}.weights.bin, ` +
    `${name}.metadata.json. Open keywordlistener.html and pick all three files.`
  );
}

function downloadJSON(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(a.href);
}

// Throw away current examples and transfer model, keep the base recognizer.
function clearAll() {
  const name = (document.getElementById('modelName').value || 'my-keywords').trim();
  transfer = baseRecognizer.createTransfer(name);
  trained = false;
  document.getElementById('save').disabled = true;
  document.getElementById('modelName').disabled = false;
  refreshCounts();
  setStatus('Session reset. Record fresh examples to continue.');
}

async function app() {
  setStatus('Loading pretrained speech-commands model…');
  baseRecognizer = speechCommands.create(
    'BROWSER_FFT', undefined, LOCAL_MODEL_URL, LOCAL_METADATA_URL);
  await baseRecognizer.ensureModelLoaded();

  const name = (document.getElementById('modelName').value || 'my-keywords').trim();
  transfer = baseRecognizer.createTransfer(name);
  refreshCounts();
  setStatus('Ready. Click Record next to each keyword to capture examples.');
}

app();
