// Element references
const form = document.getElementById("crop-form");
const submitBtn = document.getElementById("submit-btn");
const errorMsg = document.getElementById("error-msg");
const resultSection = document.getElementById("result");
const resultMain = document.getElementById("result-main");
const resultTop3 = document.getElementById("result-top3");

const FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'];

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  // 1. collect form values
  const payload = {};
  for (const feature of FEATURES) {
    const raw = document.getElementById(feature).value;
    if (raw === '') {
      showError(`Please fill in ${feature}.`);
      return;
    }
    payload[feature] = parseFloat(raw);
  }

  // 2. UI: hide previous result/error, disable button
  resultSection.classList.add('hidden');
  errorMsg.classList.add('hidden');
  submitBtn.disabled = true;
  submitBtn.textContent = 'Predicting…';

  try {
    // 3. call the API
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    // 4. handle non-2xx responses
    if (!response.ok) {
      showError(formatApiError(data));
      return;
    }

    // 5. success — render the result
    renderResult(data);
  } catch (err) {
    // network-level error (server down, no internet, etc.)
    showError('Could not reach the server. Is it running?');
    console.error(err);
  } finally {
    // 6. always restore the button
    submitBtn.disabled = false;
    submitBtn.textContent = 'Recommend Crop';
  }
});

// --- helpers ---

function formatApiError(data) {
    // FastAPI 422 response: { detail: [ { msg, loc, ... }, ... ] }
    if (data && Array.isArray(data.detail)) {
        return data.detail.map((d) => d.msg).join("; ");
    }
    if (data && typeof data.detail === "string") {
        return data.detail;
    }
    return "Something went wrong. Please check your inputs.";
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.remove('hidden');
}

function renderResult(data) {
  // main crop + confidence
  resultMain.innerHTML = `
        <span class="result-crop">${data.predicted_crop}</span>
        <span class="result-confidence">
            ${(data.confidence * 100).toFixed(1)}% confidence
        </span>
    `;

  // top-3 list
  resultTop3.innerHTML = data.top_3
    .map(
      (item) => `
            <li>
                <span class="crop-name">${item.crop}</span>
                <span class="prob">${(item.probability * 100).toFixed(1)}%</span>
            </li>`
    )
    .join('');

  resultSection.classList.remove('hidden');
}
