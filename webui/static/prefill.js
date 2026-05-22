/* ────────────────────────────────────────────────────────────────────
   prefill.js — logic for the Prefill Answer Sheets page
   ──────────────────────────────────────────────────────────────────── */

// ── Tab switching ────────────────────────────────────────────────────

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.style.display = 'none');
        btn.classList.add('active');
        document.getElementById('tab-' + btn.dataset.tab).style.display = '';
    });
});

document.querySelectorAll('.subtab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.subtab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.subtab-pane').forEach(p => p.style.display = 'none');
        btn.classList.add('active');
        document.getElementById('subtab-' + btn.dataset.subtab).style.display = '';
    });
});

// ── Helpers ──────────────────────────────────────────────────────────

function showError(el, msg) {
    el.textContent = msg;
    el.style.display = msg ? '' : 'none';
}

function setLoading(btn, loading) {
    btn.disabled = loading;
    btn.textContent = loading ? 'Generating…' : 'Generate & Download';
}

function triggerDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
}

function hasDesktopApi() {
    return Boolean(window.pywebview && window.pywebview.api);
}

function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error || new Error('Failed to read download data.'));
        reader.readAsDataURL(blob);
    });
}

async function saveDesktopBlob(blob, filename, errorEl) {
    const data = await blobToBase64(blob);
    const result = await window.pywebview.api.save_download_base64(filename, data);
    if (!result.ok && !result.cancelled) showError(errorEl, result.message || 'Download failed.');
}

async function saveDesktopUrl(downloadUrl, filename, errorEl) {
    const result = await window.pywebview.api.save_download_url(downloadUrl, filename || 'download');
    if (!result.ok && !result.cancelled) showError(errorEl, result.message || 'Download failed.');
}

async function postFormAndDownload(url, formData, submitBtn, errorEl, hintEl) {
    showError(errorEl, '');
    setLoading(submitBtn, true);
    try {
        const res = await fetch(url, { method: 'POST', body: formData });
        if (!res.ok) {
            let detail = `Server error ${res.status}`;
            try {
                const body = await res.json();
                detail = body.detail || detail;
            } catch (_) {}
            showError(errorEl, detail);
            return;
        }
        const contentType = res.headers.get('Content-Type') || '';
        if (contentType.includes('application/json')) {
            const body = await res.json();
            if (body.download_url) {
                if (hintEl && body.filename) {
                    hintEl.textContent = `Last download from server: ${body.filename}`;
                    hintEl.style.display = '';
                }
                if (hasDesktopApi()) {
                    await saveDesktopUrl(body.download_url, body.filename || 'download', errorEl);
                    return;
                }
                window.location.href = body.download_url;
                return;
            }
        }
        const disposition = res.headers.get('Content-Disposition') || '';
        const match = disposition.match(/filename="([^"]+)"/);
        const filename = match ? match[1] : 'download';
        if (hintEl) {
            hintEl.textContent = `Last download from server: ${filename}`;
            hintEl.style.display = '';
        }
        const blob = await res.blob();
        if (hasDesktopApi()) {
            await saveDesktopBlob(blob, filename, errorEl);
            return;
        }
        triggerDownload(blob, filename);
    } catch (err) {
        showError(errorEl, `Request failed: ${err.message}`);
    } finally {
        setLoading(submitBtn, false);
        submitBtn.textContent = 'Generate & Download';
    }
}

// ── Single sheet ─────────────────────────────────────────────────────

const singleForm = document.getElementById('single-form');
const singleSubmit = document.getElementById('single-submit');
const singleError = document.getElementById('single-error');

singleForm.addEventListener('submit', async e => {
    e.preventDefault();
    showError(singleError, '');

    const studentName = singleForm.querySelector('[name=student_name]').value.trim();
    const schoolName  = singleForm.querySelector('[name=school_name]').value.trim();
    const examName    = singleForm.querySelector('[name=exam_name]').value.trim();
    const candidateNo = singleForm.querySelector('[name=candidate_number]').value.trim();
    const outputFmt   = singleForm.querySelector('[name=output_format]:checked').value;
    const realismPreset = singleForm.querySelector('[name=realism_preset]').value;

    if (!studentName || !schoolName || !examName || !candidateNo) {
        showError(singleError, 'All fields are required.');
        return;
    }
    if (!/^\d{10}$/.test(candidateNo)) {
        showError(singleError, 'Candidate number must be exactly 10 digits.');
        singleForm.querySelector('[name=candidate_number]').classList.add('invalid');
        return;
    }
    singleForm.querySelector('[name=candidate_number]').classList.remove('invalid');

    const fd = new FormData();
    fd.append('student_name', studentName);
    fd.append('school_name', schoolName);
    fd.append('exam_name', examName);
    fd.append('candidate_number', candidateNo);
    fd.append('output_format', outputFmt);
    fd.append('realism_preset', realismPreset);

    const singleLastDownload = document.getElementById('single-last-download');
    await postFormAndDownload('/api/v1/prefill/single', fd, singleSubmit, singleError, singleLastDownload);
});

// ── Live preset comparison gallery ──────────────────────────────────

const PRESET_DEFINITIONS = [
    { id: 'none', label: 'None', description: 'Clean reference output' },
    { id: 'subtle', label: 'Subtle', description: 'Mild scanner noise + slight skew' },
    { id: 'moderate', label: 'Moderate', description: 'Hole-punch shadows, vignette, smudges' },
    { id: 'adversarial', label: 'Adversarial', description: 'Occlusions, scribbles, marker dog-ears' },
];

const presetPreview = document.getElementById('preset-preview');
const presetPreviewGrid = document.getElementById('preset-preview-grid');
const presetPreviewBtn = document.getElementById('single-preview-btn');
const presetPreviewRefresh = document.getElementById('preset-preview-refresh');

const renderPresetGallery = () => {
    if (!presetPreviewGrid) return;
    const candidate = singleForm.querySelector('[name=candidate_number]').value.trim() || '9010690012';
    const safeCandidate = /^\d{10}$/.test(candidate) ? candidate : '9010690012';
    const cacheBust = Date.now();
    presetPreviewGrid.innerHTML = '';
    PRESET_DEFINITIONS.forEach(({ id, label, description }) => {
        const url = `/api/v1/prefill/sample?preset=${id}&candidate_number=${safeCandidate}&_=${cacheBust}`;
        const card = document.createElement('figure');
        card.className = 'preset-preview-card';
        card.innerHTML = `
            <img alt="${label} preview" loading="lazy" src="${url}" />
            <figcaption>
                <strong>${label}</strong>
                <span class="muted small">${description}</span>
            </figcaption>
        `;
        presetPreviewGrid.appendChild(card);
    });
};

if (presetPreviewBtn && presetPreview) {
    presetPreviewBtn.addEventListener('click', () => {
        presetPreview.open = true;
        renderPresetGallery();
        presetPreview.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
}
if (presetPreview) {
    presetPreview.addEventListener('toggle', () => {
        if (presetPreview.open && presetPreviewGrid && !presetPreviewGrid.children.length) {
            renderPresetGallery();
        }
    });
}
if (presetPreviewRefresh) {
    presetPreviewRefresh.addEventListener('click', renderPresetGallery);
}

// Clear invalid styling on input
singleForm.querySelector('[name=candidate_number]').addEventListener('input', function () {
    this.classList.remove('invalid');
    showError(singleError, '');
});

// ── Batch — manual table ─────────────────────────────────────────────

const batchBody = document.getElementById('batch-body');

function makeRow(data = {}) {
    const tr = document.createElement('tr');
    const cols = ['student_name', 'school_name', 'exam_name', 'candidate_number', 'output_file'];
    cols.forEach(col => {
        const td = document.createElement('td');
        const input = document.createElement('input');
        input.type = 'text';
        input.dataset.col = col;
        input.value = data[col] || '';
        input.style.width = '100%';
        if (col === 'candidate_number') {
            input.inputMode = 'numeric';
            input.maxLength = 10;
            input.style.width = '8rem';
        }
        td.appendChild(input);
        tr.appendChild(td);
    });
    const actionTd = document.createElement('td');
    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'btn small danger';
    removeBtn.textContent = '✕';
    removeBtn.addEventListener('click', () => tr.remove());
    actionTd.appendChild(removeBtn);
    tr.appendChild(actionTd);
    return tr;
}

function addRow(data = {}) {
    batchBody.appendChild(makeRow(data));
}

// Start with one empty row
addRow();

document.getElementById('add-row-btn').addEventListener('click', () => addRow());

document.getElementById('clear-rows-btn').addEventListener('click', () => {
    if (batchBody.children.length === 0) return;
    if (!confirm('Clear all rows?')) return;
    batchBody.innerHTML = '';
    addRow();
});

function getTableRows() {
    const rows = [];
    batchBody.querySelectorAll('tr').forEach(tr => {
        const row = {};
        tr.querySelectorAll('input[data-col]').forEach(inp => {
            row[inp.dataset.col] = inp.value.trim();
        });
        rows.push(row);
    });
    return rows;
}

function rowsToCsvText(rows) {
    const headers = ['student_name', 'school_name', 'exam_name', 'candidate_number', 'output_file'];
    const lines = [headers.join(',')];
    rows.forEach(row => {
        lines.push(headers.map(h => {
            const v = row[h] || '';
            return v.includes(',') || v.includes('"') ? `"${v.replace(/"/g, '""')}"` : v;
        }).join(','));
    });
    return lines.join('\n');
}

// ── Batch — CSV upload & preview ─────────────────────────────────────

const csvUpload = document.getElementById('csv-upload');
const csvPreview = document.getElementById('csv-preview');
const csvPreviewCount = document.getElementById('csv-preview-count');
const csvPreviewHead = document.getElementById('csv-preview-head');
const csvPreviewBody = document.getElementById('csv-preview-body');

function parseCsv(text) {
    const lines = text.trim().split(/\r?\n/);
    if (lines.length < 1) return { headers: [], rows: [] };
    const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
    const rows = lines.slice(1).map(line => {
        // Simple CSV split (handles quoted fields)
        const vals = [];
        let cur = '', inQuote = false;
        for (let i = 0; i < line.length; i++) {
            const ch = line[i];
            if (ch === '"') { inQuote = !inQuote; }
            else if (ch === ',' && !inQuote) { vals.push(cur.trim()); cur = ''; }
            else { cur += ch; }
        }
        vals.push(cur.trim());
        const obj = {};
        headers.forEach((h, i) => { obj[h] = vals[i] || ''; });
        return obj;
    }).filter(r => Object.values(r).some(v => v));
    return { headers, rows };
}

let uploadedCsvText = null;

csvUpload.addEventListener('change', () => {
    const file = csvUpload.files[0];
    if (!file) { uploadedCsvText = null; csvPreview.style.display = 'none'; return; }
    const reader = new FileReader();
    reader.onload = e => {
        uploadedCsvText = e.target.result;
        const { headers, rows } = parseCsv(uploadedCsvText);
        csvPreviewHead.innerHTML = headers.map(h => `<th>${h}</th>`).join('');
        csvPreviewBody.innerHTML = rows.slice(0, 10).map(row =>
            `<tr>${headers.map(h => `<td>${row[h] || ''}</td>`).join('')}</tr>`
        ).join('');
        csvPreviewCount.textContent = `${rows.length} row(s) parsed${rows.length > 10 ? ' (showing first 10)' : ''}.`;
        csvPreview.style.display = '';
    };
    reader.readAsText(file);
});

// ── Batch submit ──────────────────────────────────────────────────────

const batchSubmit = document.getElementById('batch-submit');
const batchError  = document.getElementById('batch-error');
const batchOutputMode = document.getElementById('batch-output-mode');
const batchRealismPreset = document.getElementById('batch-realism-preset');

batchSubmit.addEventListener('click', async () => {
    showError(batchError, '');
    const activeSubtab = document.querySelector('.subtab-btn.active').dataset.subtab;
    const fd = new FormData();
    fd.append('output_mode', batchOutputMode.value);
    fd.append('realism_preset', batchRealismPreset.value);

    if (activeSubtab === 'manual') {
        const rows = getTableRows().filter(r =>
            r.student_name || r.school_name || r.exam_name || r.candidate_number
        );
        if (rows.length === 0) {
            showError(batchError, 'Add at least one row with data.');
            return;
        }
        for (const row of rows) {
            if (!row.student_name || !row.school_name || !row.exam_name || !row.candidate_number) {
                showError(batchError, 'All rows must have student name, school name, exam name, and candidate number.');
                return;
            }
            if (!/^\d{10}$/.test(row.candidate_number)) {
                showError(batchError, `Invalid candidate number "${row.candidate_number}" — must be exactly 10 digits.`);
                return;
            }
        }
        fd.append('csv_text', rowsToCsvText(rows));
    } else {
        if (!uploadedCsvText) {
            showError(batchError, 'Please upload a CSV file first.');
            return;
        }
        fd.append('csv_text', uploadedCsvText);
    }

    await postFormAndDownload('/api/v1/prefill/batch', fd, batchSubmit, batchError);
});
