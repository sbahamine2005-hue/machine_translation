const modelSelect  = document.getElementById('modelSelect');
const langTabs     = document.querySelectorAll('.target-tab');
const inputText    = document.getElementById('inputText');
const outputText   = document.getElementById('outputText');
const translateBtn = document.getElementById('translateBtn');
const charCount    = document.getElementById('charCount');
const langBadge    = document.getElementById('langBadge');

let selectedModel    = modelSelect.value;
let selectedLanguage = 'Arabic';

modelSelect.addEventListener('change', (e) => { selectedModel = e.target.value; });

langTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    langTabs.forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    selectedLanguage = tab.getAttribute('data-lang');
    langBadge.textContent = 'EN → ' + (selectedLanguage === 'Arabic' ? 'AR' : 'FR');
  });
});

inputText.addEventListener('input', () => {
  const n = inputText.value.length;
  charCount.textContent = n === 0 ? '0 characters' : n + ' character' + (n !== 1 ? 's' : '');
  if (!inputText.value.trim()) {
    outputText.value = '';
    outputText.classList.remove('filled');
    langBadge.classList.remove('visible');
  }
});

translateBtn.addEventListener('click', triggerTranslation);

function triggerTranslation() {
  const text = inputText.value.trim();
  if (!text) { outputText.value = ''; return; }
  sendToFastAPI(text, selectedModel, selectedLanguage);
}

async function sendToFastAPI(text, model, language) {
  const API_URL = "http://127.0.0.1:8000/translation";

  translateBtn.innerHTML = '<span style="letter-spacing:2px;font-size:12px;opacity:.7">···</span>';
  translateBtn.disabled = true;
  document.body.classList.add('translating');
  outputText.classList.remove('filled');

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text,
        languge_target: language,   // Matches Pydantic model exactly
        model: model                // Matches Pydantic model exactly
      })
    });

    if (response.ok) {
      const data = await response.json();
      outputText.value = data.translated_text;
      outputText.classList.add('filled');
      langBadge.textContent = 'EN → ' + (language === 'Arabic' ? 'AR' : 'FR');
      langBadge.classList.add('visible');
    } else {
      outputText.value = 'Server error. Please try again.';
    }
  } catch (err) {
    console.error(err);
    outputText.value = 'Could not reach the translation server.';
  } finally {
    translateBtn.innerHTML = 'Translate <span class="btn-arrow">→</span>';
    translateBtn.disabled = false;
    document.body.classList.remove('translating');
  }
}