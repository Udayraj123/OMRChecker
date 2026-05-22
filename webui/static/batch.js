const panel = document.querySelector("[data-batch-id]")
const batchId = panel ? panel.dataset.batchId : null
const STATUS_CLASSES = ["status-created", "status-queued", "status-running", "status-done", "status-failed", "status-cancelled"]

const show = (element, message, kind = "info") => {
    if (!element) return
    element.hidden = false
    element.textContent = message
    element.classList.remove("error", "success")
    if (kind === "error") element.classList.add("error")
    if (kind === "success") element.classList.add("success")
}

const apiUrl = (path) => `/api/v1/batches/${batchId}${path}`

const jsonFetch = async (url, options = {}) => {
    const response = await fetch(url, {
        cache: "no-store",
        ...options,
    })
    const text = await response.text()
    const data = text ? JSON.parse(text) : {}
    if (!response.ok) {
        throw new Error(data.detail || `Request failed (${response.status})`)
    }
    return data
}

const getSelectedRotation = () => {
    const select = document.getElementById("rotation-degrees")
    return select ? Number(select.value || 0) : 0
}

const applyPreviewRotation = () => {
    const degrees = getSelectedRotation()
    document.querySelectorAll(".sheet-preview-image").forEach((image) => {
        image.style.setProperty("--preview-rotation", `${degrees}deg`)
    })
}

const fileBrowserState = {
    files: [],
    filterQuery: "",
    selectedFileName: null,
}

// Tracks which filenames are currently rendered in the file navigator so we
// can do incremental DOM appends instead of a full rebuild when files are only
// being added (the common case while a PDF is being split).
let _renderedFileNames = []

const _updateActiveStates = (list, selectedFileName) => {
    for (const row of list.querySelectorAll("[data-file-row]")) {
        const btn = row.querySelector("[data-file-select]")
        if (!btn) continue
        const isActive = btn.dataset.fileSelect === selectedFileName
        row.classList.toggle("active", isActive)
        btn.classList.toggle("active", isActive)
        btn.setAttribute("aria-selected", isActive ? "true" : "false")
    }
}

const getFileBrowserElements = () => ({
    shell: document.querySelector("[data-file-browser]"),
    list: document.getElementById("file-list"),
    totalCount: document.getElementById("file-count"),
    filteredCount: document.getElementById("file-browser-count"),
    search: document.getElementById("file-search"),
    previewContent: document.getElementById("file-preview-content"),
    previewEmpty: document.getElementById("file-preview-empty"),
    previewName: document.getElementById("selected-file-name"),
    previewSize: document.getElementById("selected-file-size"),
    previewOpen: document.getElementById("selected-file-open"),
    previewDelete: document.getElementById("selected-file-delete"),
    previewLink: document.getElementById("selected-file-preview-link"),
    previewImage: document.getElementById("selected-file-preview-image"),
})

const getFilePreviewUrl = (filename) => apiUrl(`/files/${encodeURIComponent(filename)}/preview`)

const formatFileSize = (sizeBytes) => `${(Number(sizeBytes || 0) / 1024).toFixed(1)} KB`

const getVisibleFiles = () => {
    const query = fileBrowserState.filterQuery.trim().toLowerCase()
    if (!query) return fileBrowserState.files
    return fileBrowserState.files.filter((file) => file.name.toLowerCase().includes(query))
}

const resolveSelectedFile = (files, preferredName = fileBrowserState.selectedFileName) => {
    if (!Array.isArray(files) || files.length === 0) return null
    if (preferredName) {
        const preferred = files.find((file) => file.name === preferredName)
        if (preferred) return preferred
    }
    return files[0]
}

const getNeighborFileName = (fileName) => {
    const index = fileBrowserState.files.findIndex((file) => file.name === fileName)
    if (index === -1) return fileBrowserState.selectedFileName
    return fileBrowserState.files[index + 1]?.name || fileBrowserState.files[index - 1]?.name || null
}

const focusFileButton = (fileName) => {
    if (!fileName) return
    const button = Array.from(document.querySelectorAll("[data-file-select]")).find(
        (element) => element.dataset.fileSelect === fileName,
    )
    if (!button) return
    button.focus({ preventScroll: true })
    button.scrollIntoView({ block: "nearest", inline: "nearest" })
}

const openFilePreview = (fileName) => {
    if (!fileName) return
    window.open(getFilePreviewUrl(fileName), "_blank", "noopener")
}

const _buildFileListItemHtml = (file, isActive, index) => `
    <li class="file-list-row${isActive ? " active" : ""}" data-file-row data-file-name="${escapeHtml(file.name)}" data-file-size="${file.size_bytes}" data-file-index="${index}">
        <button
            class="file-list-select${isActive ? " active" : ""}"
            type="button"
            data-file-select="${escapeHtml(file.name)}"
            role="option"
            aria-selected="${isActive ? "true" : "false"}">
            <span class="mono file-list-name">${escapeHtml(file.name)}</span>
            <span class="muted small file-list-size">${formatFileSize(file.size_bytes)}</span>
        </button>
        <button class="btn danger small file-list-delete" type="button" data-delete-file="${escapeHtml(file.name)}" aria-label="Remove ${escapeHtml(file.name)}">Remove</button>
    </li>
`

const renderFileNavigator = (visibleFiles, selectedFileName) => {
    const elements = getFileBrowserElements()
    if (!elements.list) return

    if (elements.totalCount) {
        elements.totalCount.textContent = `(${fileBrowserState.files.length})`
    }
    if (elements.filteredCount) {
        elements.filteredCount.textContent = `Showing ${visibleFiles.length} of ${fileBrowserState.files.length}`
    }

    if (visibleFiles.length === 0) {
        _renderedFileNames = []
        const query = fileBrowserState.filterQuery.trim()
        elements.list.innerHTML = `<li class="file-list-empty">${query ? `No files match "${escapeHtml(query)}".` : "No files uploaded yet."}</li>`
        return
    }

    // Incremental append: when the visible list is the same set plus new items
    // at the end (the common case during PDF split progress polling), avoid a
    // full DOM rebuild -- just append the new rows and update active states.
    // This keeps the main thread unblocked for large page counts.
    const prev = _renderedFileNames
    const canAppend =
        prev.length > 0 &&
        visibleFiles.length > prev.length &&
        prev.every((name, i) => visibleFiles[i]?.name === name)

    if (canAppend) {
        const template = document.createElement("template")
        const fragment = document.createDocumentFragment()
        for (let i = prev.length; i < visibleFiles.length; i++) {
            template.innerHTML = _buildFileListItemHtml(visibleFiles[i], visibleFiles[i].name === selectedFileName, i).trim()
            fragment.appendChild(template.content.firstChild)
        }
        elements.list.appendChild(fragment)
        _renderedFileNames = visibleFiles.map((f) => f.name)
        _updateActiveStates(elements.list, selectedFileName)
        return
    }

    // Full rebuild (first render, filter change, delete, reorder, etc.)
    _renderedFileNames = visibleFiles.map((f) => f.name)
    elements.list.innerHTML = visibleFiles
        .map((file, index) => _buildFileListItemHtml(file, file.name === selectedFileName, index))
        .join("")
}

const renderFilePreview = (selectedFile) => {
    const elements = getFileBrowserElements()
    if (!elements.previewContent || !elements.previewEmpty) return

    if (!selectedFile && fileBrowserState.files.length > 0) {
        selectedFile = fileBrowserState.files[0]
        fileBrowserState.selectedFileName = selectedFile.name
    }

    if (!selectedFile) {
        elements.previewContent.hidden = true
        elements.previewEmpty.hidden = false
        const heading = elements.previewEmpty.querySelector("h3")
        const body = elements.previewEmpty.querySelector("p")
        if (fileBrowserState.files.length === 0) {
            if (heading) heading.textContent = "No files uploaded yet."
            if (body) body.textContent = "Upload images or PDFs to start browsing scans here."
        } else {
            if (heading) heading.textContent = "No matching files"
            if (body) {
                const query = fileBrowserState.filterQuery.trim()
                body.textContent = query
                    ? `Clear “${query}” to show all ${fileBrowserState.files.length} file(s).`
                    : "Adjust the filter to show files."
            }
        }
        if (elements.previewDelete) {
            elements.previewDelete.disabled = true
            delete elements.previewDelete.dataset.deleteFile
        }
        if (elements.previewOpen) {
            elements.previewOpen.setAttribute("href", "#")
        }
        if (elements.previewLink) {
            elements.previewLink.setAttribute("href", "#")
        }
        if (elements.previewImage) {
            elements.previewImage.removeAttribute("src")
            elements.previewImage.removeAttribute("alt")
        }
        return
    }

    elements.previewContent.hidden = false
    elements.previewEmpty.hidden = true
    if (elements.previewName) elements.previewName.textContent = selectedFile.name
    if (elements.previewSize) elements.previewSize.textContent = formatFileSize(selectedFile.size_bytes)

    const previewUrl = getFilePreviewUrl(selectedFile.name)
    if (elements.previewOpen) {
        elements.previewOpen.href = previewUrl
    }
    if (elements.previewDelete) {
        elements.previewDelete.disabled = false
        elements.previewDelete.dataset.deleteFile = selectedFile.name
        elements.previewDelete.setAttribute("aria-label", `Remove ${selectedFile.name}`)
    }
    if (elements.previewLink) {
        elements.previewLink.href = previewUrl
    }
    if (elements.previewImage) {
        elements.previewImage.src = previewUrl
        elements.previewImage.alt = `Preview of ${selectedFile.name}`
        elements.previewImage.style.setProperty("--preview-rotation", `${getSelectedRotation()}deg`)
    }
}

const renderFileBrowser = ({ focusSelected = false } = {}) => {
    const visibleFiles = getVisibleFiles()
    const selectedFile = resolveSelectedFile(visibleFiles)
    fileBrowserState.selectedFileName = selectedFile ? selectedFile.name : null
    renderFileNavigator(visibleFiles, fileBrowserState.selectedFileName)
    renderFilePreview(selectedFile)
    applyPreviewRotation()
    if (focusSelected && selectedFile) {
        focusFileButton(selectedFile.name)
    }
}

const bootstrapFileBrowser = () => {
    const elements = getFileBrowserElements()
    if (!elements.list) return

    const selectButtons = Array.from(elements.list.querySelectorAll("[data-file-select]"))
    const rows = Array.from(elements.list.querySelectorAll("[data-file-row]"))
    fileBrowserState.files = (selectButtons.length > 0 ? selectButtons : rows)
        .map((button) => {
            const row = button.closest ? button.closest("[data-file-row]") : button
            const name = button.dataset?.fileSelect || row?.dataset?.fileName
            if (!name) return null
            return {
                name,
                size_bytes: Number(row?.dataset?.fileSize || row?.querySelector?.("[data-file-size]")?.dataset?.fileSize || 0),
            }
        })
        .filter(Boolean)
    fileBrowserState.selectedFileName =
        elements.shell?.dataset.selectedFile ||
        elements.list.querySelector(".file-list-select.active")?.dataset.fileSelect ||
        fileBrowserState.files[0]?.name ||
        null
    const search = elements.search
    if (search) {
        fileBrowserState.filterQuery = search.value || ""
        search.addEventListener("input", handleFileBrowserSearch)
    }
    renderFileBrowser()
}

const handleFileBrowserSearch = (event) => {
    const input = event.target.closest("#file-search")
    if (!input) return
    fileBrowserState.filterQuery = input.value || ""
    renderFileBrowser()
}

const handleFileBrowserSelect = (event) => {
    const button = event.target.closest("[data-file-select]")
    if (!button) return
    event.preventDefault()
    fileBrowserState.selectedFileName = button.dataset.fileSelect || null
    renderFileBrowser({ focusSelected: true })
}

const handleFileBrowserKeyDown = (event) => {
    const button = event.target.closest("[data-file-select]")
    const inBrowser = Boolean(event.target.closest("[data-file-browser]")) || event.target === document.body
    if (!inBrowser && !button) return

    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        if (!inBrowser) return
        const visibleFiles = getVisibleFiles()
        if (!visibleFiles.length) return
        event.preventDefault()
        const currentName = button?.dataset.fileSelect || fileBrowserState.selectedFileName || visibleFiles[0].name
        const currentIndex = visibleFiles.findIndex((file) => file.name === currentName)
        const baseIndex = currentIndex === -1 ? 0 : currentIndex
        const nextIndex = Math.min(
            visibleFiles.length - 1,
            Math.max(0, baseIndex + (event.key === "ArrowDown" ? 1 : -1)),
        )
        fileBrowserState.selectedFileName = visibleFiles[nextIndex].name
        renderFileBrowser({ focusSelected: true })
        return
    }

    if (button && event.key === "Enter") {
        event.preventDefault()
        openFilePreview(button.dataset.fileSelect)
        return
    }

    if (button && (event.key === " " || event.key === "Spacebar")) {
        event.preventDefault()
        fileBrowserState.selectedFileName = button.dataset.fileSelect || null
        renderFileBrowser({ focusSelected: true })
        return
    }

    if (event.target === document.body && event.key === "Enter") {
        const selected = resolveSelectedFile(getVisibleFiles())
        if (!selected) return
        event.preventDefault()
        openFilePreview(selected.name)
    }
}

const updateStatus = (status, lastError, preprocessFailures = []) => {
    const pill = document.getElementById("status-pill")
    if (pill) {
        STATUS_CLASSES.forEach((cls) => pill.classList.remove(cls))
        pill.classList.add(`status-${status}`)
        pill.textContent = status
    }
    const errorEl = document.getElementById("last-error")
    if (errorEl) {
        if (lastError) {
            errorEl.hidden = false
            errorEl.textContent = lastError
        } else {
            errorEl.hidden = true
            errorEl.textContent = ""
        }
    }
    const warnEl = document.getElementById("preprocess-warning")
    if (warnEl) {
        if (Array.isArray(preprocessFailures) && preprocessFailures.length > 0) {
            warnEl.hidden = false
            warnEl.textContent = `Preprocessor could not locate markers in ${preprocessFailures.length} file(s): ${preprocessFailures.join(", ")}`
        } else {
            warnEl.hidden = true
            warnEl.textContent = ""
        }
    }
    const runBtn = document.getElementById("run-btn")
    if (runBtn) {
        runBtn.disabled = status === "queued" || status === "running"
        runBtn.textContent = status === "running" ? "Running..." : "Run OMR"
    }
    const stopBtn = document.getElementById("stop-btn")
    if (stopBtn) {
        stopBtn.disabled = !(status === "queued" || status === "running")
    }
    const restartBtn = document.getElementById("restart-btn")
    if (restartBtn) {
        restartBtn.disabled = status === "queued" || status === "running"
    }

    // Show/hide progress panel
    const progressEl = document.getElementById("run-progress")
    if (progressEl) {
        const showPanel = status === "running" || status === "queued" || status === "done" || status === "failed" || status === "cancelled"
        progressEl.hidden = !showPanel
    }
}

const _fmtSeconds = (s) => {
    if (s == null) return "—"
    if (s < 60) return `${Math.round(s)}s`
    const m = Math.floor(s / 60), sec = Math.round(s % 60)
    return sec > 0 ? `${m}m ${sec}s` : `${m}m`
}

// One-time fetch of system GPU/worker info, shown permanently in the status block
let _systemInfoLoaded = false
const _loadSystemInfo = async () => {
    if (_systemInfoLoaded) return
    _systemInfoLoaded = true
    try {
        const info = await fetch("/api/v1/system/info").then(r => r.json())
        const gpuEl = document.getElementById("gpu-badge")
        if (gpuEl) {
            if (info.gpu_available) {
                gpuEl.textContent = "\u26a1 GPU active"
                gpuEl.classList.add("gpu-active")
            } else {
                gpuEl.textContent = "CPU only"
                gpuEl.title = info.gpu_status
                gpuEl.classList.add("gpu-inactive")
            }
        }
    } catch (_) { /* system/info unavailable — skip silently */ }
}

const updateProgress = (data) => {
    const processed = data.processed_files ?? 0
    const total = data.total_files ?? 0
    const pct = total > 0 ? Math.round(processed / total * 100) : 0
    const isDone = data.status === "done" || data.status === "failed" || data.status === "cancelled"

    const bar = document.getElementById("run-progress-bar")
    if (bar) {
        bar.style.width = `${isDone ? 100 : pct}%`
        bar.classList.toggle("run-progress-bar--done", isDone)
    }

    const countEl = document.getElementById("run-stat-count")
    if (countEl) countEl.textContent = `${processed} / ${total} (${isDone ? 100 : pct}%)`

    const elapsedEl = document.getElementById("run-stat-elapsed")
    if (elapsedEl) elapsedEl.textContent = data.elapsed_s != null ? `${_fmtSeconds(data.elapsed_s)} elapsed` : ""

    const rateEl = document.getElementById("run-stat-rate")
    if (rateEl) rateEl.textContent = data.rate_per_min != null ? `${data.rate_per_min}/min` : (isDone ? "" : "—")

    const etaEl = document.getElementById("run-stat-eta")
    if (etaEl) {
        if (isDone) {
            etaEl.textContent = data.status === "done" ? "\u2713 Completed" : (data.status === "cancelled" ? "Cancelled" : "Failed")
        } else {
            etaEl.textContent = data.eta_s != null ? `ETA ${_fmtSeconds(data.eta_s)}` : (processed > 0 ? "calculating\u2026" : "")
        }
    }

    const fileEl = document.getElementById("run-stat-file")
    if (fileEl) fileEl.textContent = (!isDone && data.latest_processed_file) ? `last: ${data.latest_processed_file}` : ""
}

const pollStatus = async () => {
    try {
        const data = await jsonFetch(apiUrl("/status"))
        updateStatus(data.status, data.last_error, data.preprocess_failures)
        const pipelineBadge = document.getElementById("pipeline-badge")
        if (pipelineBadge) {
            const showBadge = !!data.pipelined_run && (data.status === "running" || data.status === "queued" || data.status === "done")
            pipelineBadge.hidden = !showBadge
        }
        if (data.status === "running") {
            updateProgress(data)
            setTimeout(pollStatus, 1500)
        } else if (data.status === "queued") {
            setTimeout(pollStatus, 2000)
        } else if (data.status === "done" || data.status === "cancelled" || data.status === "failed") {
            updateProgress(data)
            await refreshResults()
        } else {
            // "created" or any future status — keep polling so transitions are
            // picked up without a page reload.
            setTimeout(pollStatus, 2000)
        }
    } catch (error) {
        console.error(error)
        // Retry after a transient network error or 500 so the progress bar
        // does not silently stop updating.
        setTimeout(pollStatus, 3000)
    }
}

const refreshFiles = async (preferredSelection = fileBrowserState.selectedFileName) => {
    try {
        const files = await jsonFetch(apiUrl("/files"))
        fileBrowserState.files = files
        const resolvedPreferred =
            preferredSelection ||
            fileBrowserState.selectedFileName ||
            files[0]?.name ||
            null
        fileBrowserState.selectedFileName = resolvedPreferred
        renderFileBrowser()
    } catch (error) {
        console.error(error)
    }
}

const handleRotationChange = async (event) => {
    const select = event.target.closest("#rotation-degrees")
    if (!select) return
    const feedback = document.getElementById("rotation-feedback")
    try {
        applyPreviewRotation()
        const data = await jsonFetch(apiUrl("/rotation"), {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ rotation_degrees: Number(select.value) }),
        })
        select.value = String(data.rotation_degrees)
        applyPreviewRotation()
        if (feedback) {
            show(feedback, `Rotation saved: ${data.rotation_degrees} degrees.`, "success")
        }
    } catch (error) {
        if (feedback) show(feedback, error.message, "error")
    }
}

const refreshResults = async () => {
    try {
        const data = await jsonFetch(apiUrl("/results"))
        const container = document.getElementById("results-container")
        if (!container) return
        renderResults(container, data)
    } catch (error) {
        console.error(error)
    }
}

const getResponseColumns = (columns) => {
    return (columns || []).filter((col) => !["file_id", "input_path", "output_path", "score", "status", "error_reason"].includes(col))
}

const getResultCell = (row, col) => {
    if (col === "file_id") return row.file_id || ""
    if (col === "input_path") return row.input_path || ""
    if (col === "output_path") return row.output_path || ""
    if (col === "score") return row.score || ""
    if (col === "status") return row.status || ""
    if (col === "error_reason") return row.error_reason || ""
    return (row.responses && row.responses[col]) || ""
}

const getErasureRisk = (column, value) => {
    const answer = String(value || "").trim().toUpperCase()
    if (!/^q\d+/i.test(column) || answer.length <= 1) return 0
    const markedOptions = new Set(answer.replace(/[^A-Z]/g, "").split(""))
    if (markedOptions.size <= 1) return 0
    return Math.min(95, 65 + markedOptions.size * 10)
}

const getCheckedImageUrl = (row) => {
    // Derive the plain filename from file_id or input_path
    const rawName = row.file_id || (row.input_path ? row.input_path.split(/[\\/]/).pop() : null)
    if (!rawName) return null
    const filename = rawName.split(/[\\/]/).pop()
    return apiUrl(`/results/${encodeURIComponent(filename)}/checked`)
}

const getInputImageUrl = (row) => {
    const rawName = row.file_id || (row.input_path ? row.input_path.split(/[\\/]/).pop() : null)
    if (!rawName) return null
    const filename = rawName.split(/[\\/]/).pop()
    return apiUrl(`/files/${encodeURIComponent(filename)}/preview`)
}

const buildAnswerGrid = (row, responseColumns) => {
    const answers = responseColumns.map((col) => {
        const value = getResultCell(row, col) || "—"
        const erasureRisk = getErasureRisk(col, value)
        return `
            <div class="answer-cell${erasureRisk ? " answer-cell-risk" : ""}">
                <span class="answer-key">${escapeHtml(col)}</span>
                <strong>${escapeHtml(value)}</strong>
                ${erasureRisk ? `<span class="answer-risk" title="Multiple marks detected in CSV output. This can happen when an erased option remains dark enough to be read.">Erasure risk ${erasureRisk}%</span>` : ""}
            </div>
        `
    }).join("")
    return answers
}

const renderResultCard = (row, responseColumns, index) => {
    const fileId = row.file_id || `File ${index + 1}`
    const qcFlags = Array.isArray(row.qc_flags) ? row.qc_flags : []
    const qcPills = qcFlags.length
        ? `<div class="result-qc">${qcFlags.map((flag) => {
            const extra = flag === "HIGH_NR" ? ` (${Math.round((row.nr_percent || 0) * 100)}%)` : ""
            return `<span class="pill status-cancelled">QC ${escapeHtml(flag)}${extra}</span>`
        }).join("")}</div>`
        : ""

    const inputUrl = getInputImageUrl(row)
    const checkedUrl = getCheckedImageUrl(row)

    const reviewPane = `
        <div class="review-split">
            <div class="review-pane">
                <p class="review-pane-label muted small">Input scan</p>
                ${inputUrl
                    ? `<a href="${escapeHtml(inputUrl)}" target="_blank" rel="noopener" class="review-image-link">
                        <img src="${escapeHtml(inputUrl)}" class="review-image" alt="Input scan for ${escapeHtml(fileId)}" loading="lazy" />
                       </a>`
                    : `<p class="muted small">Input image not available.</p>`}
            </div>
            <div class="review-pane">
                <p class="review-pane-label muted small">OMR output <span class="muted" style="font-size:11px">(annotated by engine)</span></p>
                ${checkedUrl
                    ? `<a href="${escapeHtml(checkedUrl)}" target="_blank" rel="noopener" class="review-image-link">
                        <img src="${escapeHtml(checkedUrl)}" class="review-image review-image-checked"
                             alt="OMR output for ${escapeHtml(fileId)}"
                             loading="lazy"
                             onerror="this.closest('.review-pane').innerHTML='<p class=\\'muted small\\'>OMR output image not available yet. Run the batch first.</p>'" />
                       </a>`
                    : `<p class="muted small">OMR output not available.</p>`}
            </div>
        </div>
    `

    const viewToggle = `
        <div class="result-view-toggle" role="group" aria-label="Result view mode">
            <button class="btn small active" type="button" data-view-toggle="answers">Answers</button>
            <button class="btn small" type="button" data-view-toggle="review">Review</button>
        </div>
    `

    if (row.status === "failed") {
        return `
            <article class="result-card" data-result-card="${index}" ${index === 0 ? "" : "hidden"}>
                <div class="flex-between result-card-head">
                    <div>
                        <h3>${escapeHtml(fileId)}</h3>
                        ${row.input_path ? `<p class="muted small mono">${escapeHtml(row.input_path)}</p>` : ""}
                        ${qcPills}
                        ${row.error_reason ? `<p class="error small">${escapeHtml(row.error_reason)}</p>` : ""}
                    </div>
                    <span class="pill status-failed">Failed</span>
                </div>
                ${viewToggle}
                <div data-view-panel="answers">
                    ${row.output_path ? `<p class="muted small mono result-output-path">Output: ${escapeHtml(row.output_path)}</p>` : ""}
                </div>
                <div data-view-panel="review" hidden>
                    ${reviewPane}
                </div>
            </article>
        `
    }

    const answers = buildAnswerGrid(row, responseColumns)
    return `
        <article class="result-card" data-result-card="${index}" ${index === 0 ? "" : "hidden"}>
            <div class="flex-between result-card-head">
                <div>
                    <h3>${escapeHtml(fileId)}</h3>
                    ${row.input_path ? `<p class="muted small mono">${escapeHtml(row.input_path)}</p>` : ""}
                    ${qcPills}
                </div>
                ${row.score ? `<span class="pill status-done">Score ${escapeHtml(row.score)}</span>` : ""}
            </div>
            ${viewToggle}
            <div data-view-panel="answers">
                <div class="answer-grid">${answers}</div>
                <p class="muted small result-risk-note">Erasure risk is a review heuristic from the CSV output. Multiple marks on one question can indicate a student erased one answer and selected another, but the residual mark still read as filled.</p>
                ${row.output_path ? `<p class="muted small mono result-output-path">Output: ${escapeHtml(row.output_path)}</p>` : ""}
            </div>
            <div data-view-panel="review" hidden>
                ${reviewPane}
                <div class="answer-grid review-answer-grid">${answers}</div>
            </div>
        </article>
    `
}

const renderResults = (container, data) => {
    if (!data.rows || data.rows.length === 0) {
        container.innerHTML = '<p class="muted" id="no-results-msg">No results yet. Run the batch to generate results.</p>'
        return
    }

    const columns = data.columns || []
    const responseColumns = getResponseColumns(columns)
    const tabs = data.rows.map((row, index) => {
        const fileId = row.file_id || `File ${index + 1}`
        const qcFlags = Array.isArray(row.qc_flags) ? row.qc_flags : []
        const qcLabel = qcFlags.length ? `<span class="muted small">QC: ${escapeHtml(qcFlags.join(", "))}</span>` : ""
        return `
            <button
                class="result-file-tab${index === 0 ? " active" : ""}"
                type="button"
                role="tab"
                aria-selected="${index === 0 ? "true" : "false"}"
                data-result-select="${index}">
                <span class="mono">${escapeHtml(fileId)}</span>
                ${qcLabel}
                ${row.status === "failed" ? '<span class="muted small">Failed</span>' : ""}
                ${row.status !== "failed" && row.score ? `<span class="muted small">Score: ${escapeHtml(row.score)}</span>` : ""}
            </button>
        `
    }).join("")
    const cards = data.rows.map((row, index) => renderResultCard(row, responseColumns, index)).join("")
    const header = columns.map((col) => `<th>${escapeHtml(col)}</th>`).join("")
    const CSV_PREVIEW_LIMIT = 100
    const allRows = data.rows
    const previewRows = allRows.slice(0, CSV_PREVIEW_LIMIT)
    const rows = previewRows.map((row) => {
        const cells = columns.map((col) => {
            const className = ["file_id", "input_path", "output_path"].includes(col) ? ' class="mono small"' : ""
            return `<td${className}>${escapeHtml(getResultCell(row, col))}</td>`
        }).join("")
        return `<tr>${cells}</tr>`
    }).join("")
    const truncationNote = allRows.length > CSV_PREVIEW_LIMIT
        ? `<p class="muted small" style="margin:6px 0 0">Showing first ${CSV_PREVIEW_LIMIT} of ${allRows.length} rows. <a href="${apiUrl('/results/download')}">Download the full CSV</a> for all results.</p>`
        : ""

    container.innerHTML = `
        <div class="results-shell">
            <div class="result-file-list" role="tablist" aria-label="Result files">${tabs}</div>
            <div class="result-file-detail">${cards}</div>
        </div>
        <details class="raw-results" open>
            <summary>Raw CSV table</summary>
            <div class="table-scroll">
                <table class="table results-table">
                    <thead><tr>${header}</tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
            ${truncationNote}
        </details>
    `
}

const handleResultSelect = (event) => {
    const button = event.target.closest("[data-result-select]")
    if (!button) return
    const container = button.closest("#results-container")
    const selected = button.dataset.resultSelect
    container.querySelectorAll("[data-result-select]").forEach((tab) => {
        const isActive = tab.dataset.resultSelect === selected
        tab.classList.toggle("active", isActive)
        tab.setAttribute("aria-selected", isActive ? "true" : "false")
    })
    container.querySelectorAll("[data-result-card]").forEach((card) => {
        card.hidden = card.dataset.resultCard !== selected
    })
}

const handleViewToggle = (event) => {
    const button = event.target.closest("[data-view-toggle]")
    if (!button) return
    const card = button.closest("[data-result-card]")
    if (!card) return
    const targetView = button.dataset.viewToggle
    card.querySelectorAll("[data-view-toggle]").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.viewToggle === targetView)
    })
    card.querySelectorAll("[data-view-panel]").forEach((panel) => {
        panel.hidden = panel.dataset.viewPanel !== targetView
    })
}

const escapeHtml = (value) => {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;")
}

const editorStates = new WeakMap()

const defaultDoc = (docName) => {
    if (docName === "template") {
        return {
            pageDimensions: [666, 515],
            bubbleDimensions: [10, 10],
            customLabels: {},
            outputColumns: [],
            fieldBlocks: {},
            preProcessors: [],
        }
    }
    if (docName === "config") {
        return {
            dimensions: {
                display_height: 515,
                display_width: 666,
                processing_height: 515,
                processing_width: 666,
            },
            outputs: {
                show_image_level: 0,
            },
        }
    }
    if (docName === "evaluation") {
        return {
            source_type: "custom",
            options: {
                questions_in_order: [],
                answers_in_order: [],
            },
            marking_schemes: {},
        }
    }
    return {}
}

const getEditorStorageKey = (box) => `omr-editor-mode:${batchId}:${box.dataset.doc}`

const createEl = (tag, className, text) => {
    const element = document.createElement(tag)
    if (className) element.className = className
    if (text !== undefined) element.textContent = text
    return element
}

const parseTextareaDoc = (box) => {
    const docName = box.dataset.doc
    const textarea = box.querySelector("[data-doc-textarea]")
    const raw = textarea.value.trim()
    if (!raw) return { ok: true, value: defaultDoc(docName) }
    try {
        const value = JSON.parse(raw)
        if (!value || typeof value !== "object" || Array.isArray(value)) {
            return { ok: false, error: `${docName}.json must be a JSON object` }
        }
        return { ok: true, value }
    } catch (error) {
        return { ok: false, error: `Invalid JSON: ${error.message}` }
    }
}

const getEditorState = (box) => {
    const existing = editorStates.get(box)
    if (existing) return existing
    const parsed = parseTextareaDoc(box)
    const state = {
        doc: parsed.ok ? parsed.value : defaultDoc(box.dataset.doc),
        mode: "code",
    }
    editorStates.set(box, state)
    return state
}

const syncTextareaFromState = (box) => {
    const state = getEditorState(box)
    const textarea = box.querySelector("[data-doc-textarea]")
    if (state.doc === null) {
        textarea.value = ""
        return
    }
    textarea.value = JSON.stringify(state.doc, null, 2)
}

const syncStateFromTextarea = (box) => {
    const parsed = parseTextareaDoc(box)
    if (!parsed.ok) return parsed
    const state = getEditorState(box)
    state.doc = parsed.value
    return parsed
}

const persistDynamicChange = (box) => {
    syncTextareaFromState(box)
    renderDynamicEditor(box)
}

const setEditorMode = (box, mode) => {
    const feedback = box.querySelector("[data-doc-feedback]")
    const codePanel = box.querySelector("[data-code-panel]")
    const dynamicPanel = box.querySelector("[data-dynamic-panel]")
    const state = getEditorState(box)

    if (mode === "ui") {
        const parsed = syncStateFromTextarea(box)
        if (!parsed.ok) {
            show(feedback, parsed.error, "error")
            return
        }
        renderDynamicEditor(box)
    } else {
        syncTextareaFromState(box)
    }

    state.mode = mode
    codePanel.hidden = mode !== "code"
    dynamicPanel.hidden = mode !== "ui"
    codePanel.style.display = mode === "code" ? "" : "none"
    dynamicPanel.style.display = mode === "ui" ? "" : "none"
    codePanel.setAttribute("aria-hidden", mode === "code" ? "false" : "true")
    dynamicPanel.setAttribute("aria-hidden", mode === "ui" ? "false" : "true")
    box.querySelectorAll("[data-doc-mode]").forEach((button) => {
        button.classList.toggle("active", button.dataset.docMode === mode)
    })
    window.localStorage.setItem(getEditorStorageKey(box), mode)
}

const makeSection = (title, description) => {
    const section = createEl("section", "dynamic-section")
    const heading = createEl("h4", null, title)
    section.appendChild(heading)
    if (description) section.appendChild(createEl("p", "muted small", description))
    return section
}

const makeField = (label, control, hint) => {
    const wrapper = createEl("label", "dynamic-field")
    wrapper.appendChild(createEl("span", null, label))
    wrapper.appendChild(control)
    if (hint) wrapper.appendChild(createEl("small", "muted", hint))
    return wrapper
}

const makeTextInput = (value, handleChange, placeholder = "") => {
    const input = document.createElement("input")
    input.value = value ?? ""
    input.placeholder = placeholder
    input.addEventListener("input", () => handleChange(input.value))
    return input
}

const makeNumberInput = (value, handleChange, options = {}) => {
    const wrap = createEl("div", "number-control")
    const input = document.createElement("input")
    input.type = "number"
    input.value = value ?? 0
    if (options.step !== undefined) input.step = String(options.step)
    if (options.min !== undefined) input.min = String(options.min)
    if (options.max !== undefined) input.max = String(options.max)
    const commit = (raw) => {
        const next = Number(raw)
        handleChange(Number.isNaN(next) ? 0 : next)
    }
    input.addEventListener("input", () => {
        if (range) range.value = input.value
        commit(input.value)
    })
    wrap.appendChild(input)

    let range = null
    if (options.slider) {
        range = document.createElement("input")
        range.type = "range"
        range.min = String(options.min ?? 0)
        range.max = String(options.max ?? 100)
        range.step = String(options.step ?? 1)
        range.value = input.value
        range.addEventListener("input", () => {
            input.value = range.value
            commit(range.value)
        })
        wrap.appendChild(range)
    }
    return wrap
}

const makeCheckbox = (value, handleChange) => {
    const input = document.createElement("input")
    input.type = "checkbox"
    input.checked = Boolean(value)
    input.addEventListener("change", () => handleChange(input.checked))
    return input
}

const makeSelect = (value, options, handleChange) => {
    const select = document.createElement("select")
    options.forEach((option) => {
        const item = document.createElement("option")
        item.value = option
        item.textContent = option
        select.appendChild(item)
    })
    select.value = value || options[0]
    select.addEventListener("change", () => handleChange(select.value))
    return select
}

const makeJsonTextarea = (value, handleChange) => {
    const textarea = document.createElement("textarea")
    textarea.rows = 5
    textarea.value = JSON.stringify(value, null, 2)
    textarea.addEventListener("change", () => {
        try {
            handleChange(JSON.parse(textarea.value || "null"))
            textarea.classList.remove("invalid")
        } catch (_error) {
            textarea.classList.add("invalid")
        }
    })
    return textarea
}

const makeTagEditor = (items, handleChange, placeholder = "Add item") => {
    const currentItems = Array.isArray(items) ? [...items] : []
    const wrap = createEl("div", "tag-editor")
    const list = createEl("div", "tag-list")
    const input = document.createElement("input")
    input.placeholder = placeholder

    const update = (nextItems) => {
        handleChange(nextItems)
    }

    currentItems.forEach((item, index) => {
        const chip = createEl("span", "tag-chip")
        const label = createEl("span", null, String(item))
        chip.appendChild(label)

        const up = createEl("button", null, "↑")
        up.type = "button"
        up.title = "Move left"
        up.disabled = index === 0
        up.addEventListener("click", () => {
            const next = [...currentItems]
            const moved = next.splice(index, 1)[0]
            next.splice(index - 1, 0, moved)
            update(next)
        })
        chip.appendChild(up)

        const down = createEl("button", null, "↓")
        down.type = "button"
        down.title = "Move right"
        down.disabled = index === currentItems.length - 1
        down.addEventListener("click", () => {
            const next = [...currentItems]
            const moved = next.splice(index, 1)[0]
            next.splice(index + 1, 0, moved)
            update(next)
        })
        chip.appendChild(down)

        const remove = createEl("button", null, "×")
        remove.type = "button"
        remove.title = "Remove"
        remove.addEventListener("click", () => {
            update(currentItems.filter((_entry, itemIndex) => itemIndex !== index))
        })
        chip.appendChild(remove)
        list.appendChild(chip)
    })

    input.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== ",") return
        event.preventDefault()
        const value = input.value.trim()
        if (!value) return
        input.value = ""
        update([...currentItems, value])
    })

    wrap.appendChild(list)
    wrap.appendChild(input)
    return wrap
}

const addPairControls = (section, label, values, handleChange, names = ["width", "height"]) => {
    const row = createEl("div", "dynamic-pair")
    const pair = Array.isArray(values) ? [...values] : [0, 0]
    names.forEach((name, index) => {
        row.appendChild(
            makeField(
                `${label} ${name}`,
                makeNumberInput(pair[index], (next) => {
                    const updated = [...pair]
                    updated[index] = next
                    handleChange(updated)
                }),
            )
        )
    })
    section.appendChild(row)
}

const updateObjectKey = (objectValue, oldKey, newKey) => {
    const safeKey = newKey.trim()
    if (!safeKey || safeKey === oldKey || objectValue[safeKey]) return oldKey
    const entries = Object.entries(objectValue)
    const rebuilt = {}
    entries.forEach(([key, value]) => {
        rebuilt[key === oldKey ? safeKey : key] = value
    })
    Object.keys(objectValue).forEach((key) => delete objectValue[key])
    Object.assign(objectValue, rebuilt)
    return safeKey
}

const makeCard = (title, actions = []) => {
    const card = createEl("article", "dynamic-card")
    const head = createEl("div", "dynamic-card-head")
    head.appendChild(createEl("h5", null, title))
    const actionWrap = createEl("div", "actions")
    actions.forEach((action) => actionWrap.appendChild(action))
    head.appendChild(actionWrap)
    card.appendChild(head)
    return card
}

const renderGenericObject = (container, objectValue, handleChange, skipKeys = new Set()) => {
    Object.entries(objectValue || {}).forEach(([key, value]) => {
        if (skipKeys.has(key)) return
        if (typeof value === "boolean") {
            container.appendChild(makeField(key, makeCheckbox(value, (next) => handleChange(key, next))))
            return
        }
        if (typeof value === "number") {
            container.appendChild(makeField(key, makeNumberInput(value, (next) => handleChange(key, next), { slider: true, min: 0, max: Math.max(100, value * 2 || 100) })))
            return
        }
        if (typeof value === "string") {
            container.appendChild(makeField(key, makeTextInput(value, (next) => handleChange(key, next))))
            return
        }
        if (Array.isArray(value) && value.every((entry) => typeof entry === "string" || typeof entry === "number")) {
            container.appendChild(makeField(key, makeTagEditor(value, (next) => handleChange(key, next))))
            return
        }
        container.appendChild(makeField(key, makeJsonTextarea(value, (next) => handleChange(key, next)), "Custom JSON"))
    })
}

const renderTemplateEditor = (box, root, doc) => {
    doc.pageDimensions = Array.isArray(doc.pageDimensions) ? doc.pageDimensions : [666, 515]
    doc.bubbleDimensions = Array.isArray(doc.bubbleDimensions) ? doc.bubbleDimensions : [10, 10]
    doc.outputColumns = Array.isArray(doc.outputColumns) ? doc.outputColumns : []
    doc.customLabels = doc.customLabels && typeof doc.customLabels === "object" ? doc.customLabels : {}
    doc.fieldBlocks = doc.fieldBlocks && typeof doc.fieldBlocks === "object" ? doc.fieldBlocks : {}
    doc.preProcessors = Array.isArray(doc.preProcessors) ? doc.preProcessors : []

    const basics = makeSection("Sheet Basics", "Normalized dimensions and output column order.")
    addPairControls(basics, "Page", doc.pageDimensions, (next) => {
        doc.pageDimensions = next
        persistDynamicChange(box)
    })
    addPairControls(basics, "Bubble", doc.bubbleDimensions, (next) => {
        doc.bubbleDimensions = next
        persistDynamicChange(box)
    })
    basics.appendChild(makeField("Output columns", makeTagEditor(doc.outputColumns, (next) => {
        doc.outputColumns = next
        persistDynamicChange(box)
    }, "q1..25")))
    root.appendChild(basics)

    const labels = makeSection("Custom Labels", "Grouped output columns such as candidate number fields.")
    Object.entries(doc.customLabels).forEach(([key, value]) => {
        const card = makeCard(key)
        card.appendChild(makeField("Name", makeTextInput(key, (next) => {
            updateObjectKey(doc.customLabels, key, next)
            persistDynamicChange(box)
        })))
        card.appendChild(makeField("Field tags", makeTagEditor(value, (next) => {
            doc.customLabels[key] = next
            persistDynamicChange(box)
        }, "cand1..10")))
        const remove = createEl("button", "btn danger small", "Remove")
        remove.type = "button"
        remove.addEventListener("click", () => {
            delete doc.customLabels[key]
            persistDynamicChange(box)
        })
        card.querySelector(".actions").appendChild(remove)
        labels.appendChild(card)
    })
    const addLabel = createEl("button", "btn small", "Add label group")
    addLabel.type = "button"
    addLabel.addEventListener("click", () => {
        let index = Object.keys(doc.customLabels).length + 1
        while (doc.customLabels[`CustomLabel${index}`]) index += 1
        doc.customLabels[`CustomLabel${index}`] = []
        persistDynamicChange(box)
    })
    labels.appendChild(addLabel)
    root.appendChild(labels)

    const blocks = makeSection("Field Blocks", "Bubble grids for questions, candidate numbers, and other sheet fields.")
    Object.entries(doc.fieldBlocks).forEach(([key, block]) => {
        const remove = createEl("button", "btn danger small", "Remove")
        remove.type = "button"
        remove.addEventListener("click", () => {
            delete doc.fieldBlocks[key]
            persistDynamicChange(box)
        })
        const card = makeCard(key, [remove])
        block.origin = Array.isArray(block.origin) ? block.origin : [0, 0]
        block.fieldLabels = Array.isArray(block.fieldLabels) ? block.fieldLabels : []
        card.appendChild(makeField("Block name", makeTextInput(key, (next) => {
            updateObjectKey(doc.fieldBlocks, key, next)
            persistDynamicChange(box)
        })))
        card.appendChild(makeField("Field type", makeSelect(block.fieldType || "QTYPE_MCQ4", ["QTYPE_MCQ4", "QTYPE_INT", "__CUSTOM__"], (next) => {
            block.fieldType = next
            persistDynamicChange(box)
        })))
        card.appendChild(makeField("Direction", makeSelect(block.direction || "vertical", ["vertical", "horizontal"], (next) => {
            block.direction = next
            persistDynamicChange(box)
        })))
        addPairControls(card, "Origin", block.origin, (next) => {
            block.origin = next
            persistDynamicChange(box)
        }, ["x", "y"])
        card.appendChild(makeField("Bubbles gap", makeNumberInput(block.bubblesGap ?? 0, (next) => {
            block.bubblesGap = next
            persistDynamicChange(box)
        }, { step: 0.1, slider: true, min: 0, max: 100 })))
        card.appendChild(makeField("Labels gap", makeNumberInput(block.labelsGap ?? 0, (next) => {
            block.labelsGap = next
            persistDynamicChange(box)
        }, { step: 0.1, slider: true, min: 0, max: 150 })))
        card.appendChild(makeField("Field labels", makeTagEditor(block.fieldLabels, (next) => {
            block.fieldLabels = next
            persistDynamicChange(box)
        }, "q1..5")))
        renderGenericObject(card, block, (childKey, next) => {
            block[childKey] = next
            persistDynamicChange(box)
        }, new Set(["origin", "fieldLabels", "fieldType", "direction", "bubblesGap", "labelsGap"]))
        blocks.appendChild(card)
    })
    const addBlock = createEl("button", "btn small", "Add field block")
    addBlock.type = "button"
    addBlock.addEventListener("click", () => {
        let index = Object.keys(doc.fieldBlocks).length + 1
        while (doc.fieldBlocks[`fieldBlock${index}`]) index += 1
        doc.fieldBlocks[`fieldBlock${index}`] = {
            origin: [0, 0],
            bubblesGap: 20,
            labelsGap: 40,
            fieldLabels: [],
            fieldType: "QTYPE_MCQ4",
        }
        persistDynamicChange(box)
    })
    blocks.appendChild(addBlock)
    root.appendChild(blocks)

    const processors = makeSection("Preprocessors", "Ordered image transforms before bubble reading.")
    doc.preProcessors.forEach((processor, index) => {
        processor.options = processor.options && typeof processor.options === "object" ? processor.options : {}
        const remove = createEl("button", "btn danger small", "Remove")
        remove.type = "button"
        remove.addEventListener("click", () => {
            doc.preProcessors.splice(index, 1)
            persistDynamicChange(box)
        })
        const up = createEl("button", "btn small", "Up")
        up.type = "button"
        up.disabled = index === 0
        up.addEventListener("click", () => {
            const moved = doc.preProcessors.splice(index, 1)[0]
            doc.preProcessors.splice(index - 1, 0, moved)
            persistDynamicChange(box)
        })
        const down = createEl("button", "btn small", "Down")
        down.type = "button"
        down.disabled = index === doc.preProcessors.length - 1
        down.addEventListener("click", () => {
            const moved = doc.preProcessors.splice(index, 1)[0]
            doc.preProcessors.splice(index + 1, 0, moved)
            persistDynamicChange(box)
        })
        const card = makeCard(`${index + 1}. ${processor.name || "Processor"}`, [up, down, remove])
        card.draggable = true
        card.dataset.arrayIndex = String(index)
        card.appendChild(makeField("Name", makeSelect(processor.name || "CropOnMarkers", ["CropOnMarkers", "CropPage", "FeatureBasedAlignment", "GaussianBlur", "Levels", "MedianBlur"], (next) => {
            processor.name = next
            persistDynamicChange(box)
        })))
        renderGenericObject(card, processor.options, (childKey, next) => {
            processor.options[childKey] = next
            persistDynamicChange(box)
        })
        card.appendChild(makeField("Options JSON", makeJsonTextarea(processor.options, (next) => {
            processor.options = next && typeof next === "object" && !Array.isArray(next) ? next : {}
            persistDynamicChange(box)
        }), "Use for nested arrays like markerCorners or referenceMarkerCenters."))
        processors.appendChild(card)
    })
    const addProcessor = createEl("button", "btn small", "Add preprocessor")
    addProcessor.type = "button"
    addProcessor.addEventListener("click", () => {
        doc.preProcessors.push({ name: "CropOnMarkers", options: { relativePath: "omr_marker.jpg" } })
        persistDynamicChange(box)
    })
    processors.appendChild(addProcessor)
    root.appendChild(processors)

    const handled = new Set(["pageDimensions", "bubbleDimensions", "customLabels", "outputColumns", "fieldBlocks", "preProcessors"])
    const advanced = makeSection("Advanced JSON", "Any custom keys not covered above.")
    renderGenericObject(advanced, doc, (key, next) => {
        doc[key] = next
        persistDynamicChange(box)
    }, handled)
    if (advanced.children.length > 2) root.appendChild(advanced)
}

const renderConfigEditor = (box, root, doc) => {
    doc.dimensions = doc.dimensions && typeof doc.dimensions === "object" ? doc.dimensions : {}
    doc.outputs = doc.outputs && typeof doc.outputs === "object" ? doc.outputs : {}
    doc.threshold_params = doc.threshold_params && typeof doc.threshold_params === "object" ? doc.threshold_params : {}
    doc.alignment_params = doc.alignment_params && typeof doc.alignment_params === "object" ? doc.alignment_params : {}

    const dimensions = makeSection("Dimensions")
    const dimensionKeys = ["display_height", "display_width", "processing_height", "processing_width"]
    dimensionKeys.forEach((key) => {
        dimensions.appendChild(makeField(key, makeNumberInput(doc.dimensions[key] ?? 0, (next) => {
            doc.dimensions[key] = next
            persistDynamicChange(box)
        }, { slider: true, min: 0, max: 3000 })))
    })
    root.appendChild(dimensions)

    const outputs = makeSection("Outputs")
    renderGenericObject(outputs, doc.outputs, (key, next) => {
        doc.outputs[key] = next
        persistDynamicChange(box)
    })
    root.appendChild(outputs)

    const thresholds = makeSection("Threshold Parameters")
    renderGenericObject(thresholds, doc.threshold_params, (key, next) => {
        doc.threshold_params[key] = next
        persistDynamicChange(box)
    })
    root.appendChild(thresholds)

    const alignment = makeSection("Alignment Parameters")
    renderGenericObject(alignment, doc.alignment_params, (key, next) => {
        doc.alignment_params[key] = next
        persistDynamicChange(box)
    })
    root.appendChild(alignment)

    const handled = new Set(["dimensions", "outputs", "threshold_params", "alignment_params"])
    const advanced = makeSection("Advanced JSON")
    renderGenericObject(advanced, doc, (key, next) => {
        doc[key] = next
        persistDynamicChange(box)
    }, handled)
    if (advanced.children.length > 1) root.appendChild(advanced)
}

const renderEvaluationEditor = (box, root, doc) => {
    doc.options = doc.options && typeof doc.options === "object" ? doc.options : {}
    doc.marking_schemes = doc.marking_schemes && typeof doc.marking_schemes === "object" ? doc.marking_schemes : {}

    const source = makeSection("Source")
    source.appendChild(makeField("source_type", makeSelect(doc.source_type || "custom", ["custom", "csv"], (next) => {
        doc.source_type = next
        persistDynamicChange(box)
    })))
    root.appendChild(source)

    const options = makeSection("Options")
    renderGenericObject(options, doc.options, (key, next) => {
        doc.options[key] = next
        persistDynamicChange(box)
    })
    root.appendChild(options)

    const schemes = makeSection("Marking Schemes")
    Object.entries(doc.marking_schemes).forEach(([key, scheme]) => {
        const remove = createEl("button", "btn danger small", "Remove")
        remove.type = "button"
        remove.addEventListener("click", () => {
            delete doc.marking_schemes[key]
            persistDynamicChange(box)
        })
        const card = makeCard(key, [remove])
        card.appendChild(makeField("Scheme name", makeTextInput(key, (next) => {
            updateObjectKey(doc.marking_schemes, key, next)
            persistDynamicChange(box)
        })))
        if (key === "DEFAULT") {
            renderGenericObject(card, scheme, (markKey, next) => {
                scheme[markKey] = next
                persistDynamicChange(box)
            })
            schemes.appendChild(card)
            return
        }
        scheme.marking = scheme.marking && typeof scheme.marking === "object" ? scheme.marking : {
            correct: 1,
            incorrect: 0,
            unmarked: 0,
        }
        if (scheme.questions !== undefined) {
            card.appendChild(makeField("Questions", makeTagEditor(scheme.questions || [], (next) => {
                scheme.questions = next
                persistDynamicChange(box)
            }, "q1..25")))
        }
        renderGenericObject(card, scheme.marking, (markKey, next) => {
            scheme.marking[markKey] = next
            persistDynamicChange(box)
        })
        renderGenericObject(card, scheme, (childKey, next) => {
            scheme[childKey] = next
            persistDynamicChange(box)
        }, new Set(["questions", "marking"]))
        schemes.appendChild(card)
    })
    const addScheme = createEl("button", "btn small", "Add marking scheme")
    addScheme.type = "button"
    addScheme.addEventListener("click", () => {
        let index = Object.keys(doc.marking_schemes).length + 1
        while (doc.marking_schemes[`SCHEME_${index}`]) index += 1
        doc.marking_schemes[`SCHEME_${index}`] = {
            questions: [],
            marking: {
                correct: 1,
                incorrect: 0,
                unmarked: 0,
            },
        }
        persistDynamicChange(box)
    })
    schemes.appendChild(addScheme)
    root.appendChild(schemes)

    const handled = new Set(["source_type", "options", "marking_schemes"])
    const advanced = makeSection("Advanced JSON")
    renderGenericObject(advanced, doc, (key, next) => {
        doc[key] = next
        persistDynamicChange(box)
    }, handled)
    if (advanced.children.length > 1) root.appendChild(advanced)
}

function renderDynamicEditor(box) {
    const root = box.querySelector("[data-dynamic-panel]")
    const state = getEditorState(box)
    const doc = state.doc
    root.innerHTML = ""

    if (doc === null) {
        root.appendChild(createEl("p", "muted", "Document cleared. Save to delete it, or switch back to UI mode from code to start a fresh document."))
        return
    }

    if (!doc || typeof doc !== "object" || Array.isArray(doc)) {
        root.appendChild(createEl("p", "error", "This editor only supports JSON objects."))
        return
    }

    if (box.dataset.doc === "template") {
        renderTemplateEditor(box, root, doc)
    } else if (box.dataset.doc === "config") {
        renderConfigEditor(box, root, doc)
    } else if (box.dataset.doc === "evaluation") {
        renderEvaluationEditor(box, root, doc)
    } else {
        renderGenericObject(root, doc, (key, next) => {
            doc[key] = next
            persistDynamicChange(box)
        })
    }
}

const refreshAssets = async () => {
    try {
        const assets = await jsonFetch(apiUrl("/assets"))
        const list = document.getElementById("asset-list")
        const empty = document.getElementById("no-assets-msg")
        const panel = document.getElementById("template-assets-panel")
        if (!panel) return

        if (!assets || assets.length === 0) {
            if (list) list.remove()
            if (!document.getElementById("no-assets-msg")) {
                const p = document.createElement("p")
                p.className = "muted"
                p.id = "no-assets-msg"
                p.textContent = "Your template does not reference any external assets."
                panel.appendChild(p)
            }
            return
        }

        if (empty) empty.remove()

        let target = list
        if (!target) {
            target = document.createElement("ul")
            target.className = "asset-list"
            target.id = "asset-list"
            panel.appendChild(target)
        }
        target.innerHTML = ""

        assets.forEach((asset) => {
            const li = document.createElement("li")
            li.dataset.assetName = asset.name

            if (asset.present) {
                const previewUrl = apiUrl(`/assets/${encodeURIComponent(asset.name)}/preview`)
                const preview = document.createElement("a")
                preview.className = "image-preview-link asset-preview-link"
                preview.href = previewUrl
                preview.target = "_blank"
                preview.rel = "noopener"

                const img = document.createElement("img")
                img.className = "image-preview-thumb asset-preview-thumb"
                img.src = previewUrl
                img.alt = `Preview of ${asset.name}`
                img.loading = "lazy"
                preview.appendChild(img)
                li.appendChild(preview)
            } else {
                const missing = document.createElement("span")
                missing.className = "image-preview-missing"
                missing.setAttribute("aria-hidden", "true")
                li.appendChild(missing)
            }

            const name = document.createElement("span")
            name.className = "mono"
            name.textContent = asset.name
            li.appendChild(name)

            const pill = document.createElement("span")
            pill.className = `pill status-${asset.present ? "done" : "failed"}`
            pill.textContent = asset.present ? "present" : "missing"
            li.appendChild(pill)

            const meta = document.createElement("span")
            meta.className = "muted small"
            if (asset.present && typeof asset.size_bytes === "number") {
                meta.textContent = `${(asset.size_bytes / 1024).toFixed(1)} KB`
            } else {
                meta.textContent = "required by template"
            }
            li.appendChild(meta)

            if (asset.present) {
                const btn = document.createElement("button")
                btn.className = "btn danger small"
                btn.type = "button"
                btn.dataset.deleteAsset = asset.name
                btn.textContent = "Remove"
                li.appendChild(btn)
            }

            target.appendChild(li)
        })
    } catch (error) {
        console.error(error)
    }
}

// Auto-start OMR: flips once per upload session when /status reports
// enough split pages and the batch is ready to run. Module-level so the
// poller (inside handleUpload), handleRun, and handleRestart can all
// reach it. Reset semantics:
//   * Cleared back to false when _isBackgroundUpload is cleared (i.e.
//     the split finished OR errored) so a fresh upload re-arms it.
//   * Cleared in handleRun / handleRestart so a manual click doesn't
//     suppress a later auto-start on a re-upload.
let _autoStartFired = false

// Top-level so the regression test can drive it directly via the
// __autoStartTriggerSelfTest hook below. Gates (all required, short-circuit):
//   1. settings.auto_start_omr_with_split === true
//   2. split is still in flight (pdf_split_total > 0)
//   3. enough pages produced (pdf_split_pages >= auto_start_omr_min_pages)
//   4. batch has template.json
//   5. batch has config.json OR auto_start_omr_require_config === false
//   6. batch isn't already running/queued
//   7. guard hasn't already fired for this upload session
const handleAutoStart = async (status) => {
    if (_autoStartFired) return
    if (status.auto_start_omr_with_split !== true) return
    const splitTotal = status.pdf_split_total ?? 0
    if (splitTotal <= 0) return
    const pages = status.pdf_split_pages ?? 0
    const minPages = status.auto_start_omr_min_pages ?? 10
    if (pages < minPages) return
    if (status.has_template !== true) return
    const requireConfig = status.auto_start_omr_require_config === true
    if (requireConfig && status.has_config !== true) return
    const runStatus = status.status || ""
    if (runStatus === "running" || runStatus === "queued") return

    // Latch BEFORE the await so a re-entrant 500ms poll tick that lands
    // while /process is in-flight can't double-fire the request.
    _autoStartFired = true

    const btn = document.getElementById("pipeline-start-btn")
    const fb = document.getElementById("pipeline-start-feedback")
    if (btn) btn.disabled = true
    if (fb) fb.textContent = `Auto-started after ${pages} pages. The 'Pipelined' badge will appear once OMR begins.`

    try {
        await jsonFetch(apiUrl("/process"), { method: "POST" })
        updateStatus("queued", null)
        setTimeout(pollStatus, 1000)
    } catch (error) {
        // Transient failure (e.g. 5xx, network): un-latch so the next
        // poll tick can retry. Manual button stays enabled as a fallback.
        _autoStartFired = false
        if (btn) btn.disabled = false
        if (fb) fb.textContent = ""
        console.warn("Auto-start /process failed; will retry on next tick:", error)
    }
}

const handleUpload = async (event) => {
    event.preventDefault()
    const feedback = document.getElementById("upload-feedback")
    const input = document.getElementById("file-input")
    if (!input.files || input.files.length === 0) {
        show(feedback, "Select at least one image first.", "error")
        return
    }

    const hasPdf = Array.from(input.files).some((f) => f.name.toLowerCase().endsWith(".pdf"))
    const progressEl = document.getElementById("upload-progress")
    const progressBar = document.getElementById("upload-progress-bar")
    const statCount = document.getElementById("upload-stat-count")
    const statPct = document.getElementById("upload-stat-pct")

    let pollInterval = null
    const _updateSplitProgress = (pages, total) => {
        const pct = total > 0 ? Math.round(pages / total * 100) : 0
        if (progressBar) progressBar.style.width = `${pct}%`
        if (statCount) statCount.textContent = `${pages} / ${total} pages`
        if (statPct) statPct.textContent = `${pct}%`
    }

    // Set when the server accepted the upload as a background task (202).
    // The poll is responsible for detecting completion and cleaning up.
    let _isBackgroundUpload = false

    if (hasPdf && progressEl) {
        _updateSplitProgress(0, 0)
        progressEl.hidden = false
        // Track the last page count we refreshed the file list at so we only
        // call refreshFiles() when meaningful new pages have been written.
        let _splitLastRefreshPages = 0
        let _splitRefreshPending = false
        // Whether the poll has seen total > 0 (split is/was in progress).
        let _splitSeenTotal = false
        pollInterval = setInterval(async () => {
            try {
                const status = await jsonFetch(apiUrl("/status"))
                const pages = status.pdf_split_pages ?? 0
                const total = status.pdf_split_total ?? 0
                if (status.pdf_split_error) {
                    // Background split failed — show error and stop polling.
                    _isBackgroundUpload = false
                    _autoStartFired = false
                    _splitSeenTotal = false
                    if (pollInterval) { clearInterval(pollInterval); pollInterval = null }
                    show(feedback, status.pdf_split_error, "error")
                    if (progressEl) progressEl.hidden = true
                } else if (total > 0) {
                    _splitSeenTotal = true
                    _updateSplitProgress(pages, total)
                    // Refresh the file list every ~20 new pages so split sheets
                    // appear progressively instead of all at once at the end.
                    if (pages - _splitLastRefreshPages >= 20 && !_splitRefreshPending) {
                        _splitLastRefreshPages = pages
                        _splitRefreshPending = true
                        refreshFiles().finally(() => { _splitRefreshPending = false })
                    }
                    // Reveal the pipelined-start affordance once at least one
                    // page exists and the batch isn't already running/queued.
                    const pipelineAction = document.getElementById("pipeline-action")
                    const runStatus = status.status || ""
                    const canStart = pages > 0 && runStatus !== "running" && runStatus !== "queued"
                    if (pipelineAction) pipelineAction.hidden = !canStart
                    // Auto-fire /process when the operator's settings
                    // allow it. handleAutoStart is idempotent via
                    // _autoStartFired so re-entry is safe.
                    handleAutoStart(status)
                } else if (_splitSeenTotal) {
                    // total went back to 0 after being active → split complete.
                    _splitSeenTotal = false
                    _isBackgroundUpload = false
                    _autoStartFired = false
                    if (pollInterval) { clearInterval(pollInterval); pollInterval = null }
                    await refreshFiles()
                    const count = fileBrowserState.files.length
                    show(feedback, `Split complete — ${count} page(s) ready.`, "success")
                    if (progressEl) {
                        _updateSplitProgress(count, count)
                        setTimeout(() => { progressEl.hidden = true }, 1200)
                    }
                    const pipelineAction = document.getElementById("pipeline-action")
                    if (pipelineAction) pipelineAction.hidden = true
                }
            } catch (_) { /* ignore poll errors during upload */ }
        }, 500)
    }

    const formData = new FormData()
    Array.from(input.files).forEach((file) => formData.append("files", file))
    try {
        const response = await fetch(apiUrl("/files"), { method: "POST", body: formData })
        const data = await response.json()
        if (!response.ok) throw new Error(data.detail || "Upload failed")
        if (response.status === 202) {
            // Background split started — poll handles progress and completion.
            _isBackgroundUpload = true
            show(feedback, "Splitting PDF pages in background\u2026", "info")
            input.value = ""
        } else {
            show(feedback, `Uploaded ${data.length} file(s).`, "success")
            input.value = ""
            await refreshFiles()
        }
    } catch (error) {
        show(feedback, error.message, "error")
    } finally {
        // For background PDF uploads the poll handles cleanup; skip it here.
        if (!_isBackgroundUpload) {
            if (pollInterval) clearInterval(pollInterval)
            if (hasPdf && progressEl) {
                _updateSplitProgress(100, 100)
                setTimeout(() => { progressEl.hidden = true }, 1200)
            }
        }
    }
}

const handleImport = async (event) => {
    event.preventDefault()
    const feedback = document.getElementById("import-feedback")
    const sourceDir = document.getElementById("source-dir").value.trim()
    if (!sourceDir) {
        show(feedback, "Provide a directory path.", "error")
        return
    }
    try {
        const data = await jsonFetch(apiUrl("/files/import"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source_dir: sourceDir, copy: true }),
        })
        show(feedback, `Imported ${data.imported.length} file(s), skipped ${data.skipped.length}.`, "success")
        await refreshFiles()
    } catch (error) {
        show(feedback, error.message, "error")
    }
}

const handleDeleteFile = async (event) => {
    const button = event.target.closest("[data-delete-file]")
    if (!button) return
    const filename = button.dataset.deleteFile
    if (!window.confirm(`Remove ${filename}?`)) return
    const preferredSelection = filename === fileBrowserState.selectedFileName
        ? getNeighborFileName(filename)
        : fileBrowserState.selectedFileName
    try {
        const response = await fetch(apiUrl(`/files/${encodeURIComponent(filename)}`), { method: "DELETE" })
        if (!response.ok && response.status !== 204) {
            const data = await response.json().catch(() => ({}))
            throw new Error(data.detail || "Delete failed")
        }
        await refreshFiles(preferredSelection)
    } catch (error) {
        window.alert(error.message)
    }
}

const handleSaveDoc = async (event) => {
    const button = event.target.closest("[data-save-doc]")
    if (!button) return
    const box = button.closest(".json-box")
    const docName = box.dataset.doc
    const textarea = box.querySelector("[data-doc-textarea]")
    const feedback = box.querySelector("[data-doc-feedback]")
    const statusEl = box.querySelector("[data-doc-status]")
    const state = getEditorState(box)
    if (state.mode === "ui") {
        if (state.doc !== null) syncTextareaFromState(box)
    } else {
        const synced = syncStateFromTextarea(box)
        if (!synced.ok) {
            show(feedback, synced.error, "error")
            return
        }
    }
    const raw = textarea.value.trim()
    let payload = null
    if (raw) {
        try {
            payload = JSON.parse(raw)
        } catch (error) {
            show(feedback, `Invalid JSON: ${error.message}`, "error")
            return
        }
    }
    try {
        const response = await fetch(apiUrl(`/${docName}`), {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        })
        const data = await response.json()
        if (!response.ok) throw new Error(data.detail || "Save failed")
        show(feedback, data.status === "deleted" ? "Deleted." : "Saved.", "success")
        if (statusEl) statusEl.textContent = raw ? "present" : "empty"
        if (docName === "template") {
            await refreshAssets()
        }
    } catch (error) {
        show(feedback, error.message, "error")
    }
}

const handleClearDoc = (event) => {
    const button = event.target.closest("[data-clear-doc]")
    if (!button) return
    const box = button.closest(".json-box")
    const state = getEditorState(box)
    const textarea = box.querySelector("[data-doc-textarea]")
    state.doc = null
    textarea.value = ""
    if (state.mode === "ui") renderDynamicEditor(box)
}

const handleEditorMode = (event) => {
    const button = event.target.closest("[data-doc-mode]")
    if (!button) return
    const box = button.closest(".json-box")
    setEditorMode(box, button.dataset.docMode)
}

const handleAssetUpload = async (event) => {
    event.preventDefault()
    const feedback = document.getElementById("asset-upload-feedback")
    const input = document.getElementById("asset-file-input")
    if (!input.files || input.files.length === 0) {
        show(feedback, "Select at least one asset file first.", "error")
        return
    }
    const formData = new FormData()
    Array.from(input.files).forEach((file) => formData.append("files", file))
    try {
        const response = await fetch(apiUrl("/assets"), { method: "POST", body: formData })
        const data = await response.json()
        if (!response.ok) throw new Error(data.detail || "Asset upload failed")
        show(feedback, `Uploaded ${data.length} asset(s).`, "success")
        input.value = ""
        await refreshAssets()
    } catch (error) {
        show(feedback, error.message, "error")
    }
}

const handleDeleteAsset = async (event) => {
    const button = event.target.closest("[data-delete-asset]")
    if (!button) return
    const name = button.dataset.deleteAsset
    if (!window.confirm(`Remove asset ${name}?`)) return
    try {
        const response = await fetch(apiUrl(`/assets/${encodeURIComponent(name)}`), { method: "DELETE" })
        if (!response.ok && response.status !== 204) {
            const data = await response.json().catch(() => ({}))
            throw new Error(data.detail || "Delete failed")
        }
        await refreshAssets()
    } catch (error) {
        window.alert(error.message)
    }
}

const handleRun = async () => {
    const errorEl = document.getElementById("last-error")
    _autoStartFired = false
    try {
        await jsonFetch(apiUrl("/process"), { method: "POST" })
        updateStatus("queued", null)
        setTimeout(pollStatus, 1000)
    } catch (error) {
        show(errorEl, error.message, "error")
    }
}

// Pipelined start: triggered from the upload-progress panel while a PDF
// split is still running. Hits the same /process endpoint but surfaces
// dedicated feedback so the operator can confirm the pipeline engaged
// (look for the green "Pipelined" badge once the run starts).
const handlePipelineStart = async () => {
    const btn = document.getElementById("pipeline-start-btn")
    const fb = document.getElementById("pipeline-start-feedback")
    const errorEl = document.getElementById("last-error")
    if (btn) btn.disabled = true
    if (fb) fb.textContent = "Starting pipelined run\u2026"
    try {
        await jsonFetch(apiUrl("/process"), { method: "POST" })
        if (fb) fb.textContent = "Pipelined run queued. Look for the green \u2018Pipelined\u2019 badge in the status bar above."
        updateStatus("queued", null)
        setTimeout(pollStatus, 1000)
    } catch (error) {
        if (btn) btn.disabled = false
        if (fb) fb.textContent = ""
        show(errorEl, error.message, "error")
    }
}

const handleStop = async () => {
    const errorEl = document.getElementById("last-error")
    try {
        const data = await jsonFetch(apiUrl("/cancel"), { method: "POST" })
        updateStatus(data.status, data.status === "running" ? "Stop requested. The current image will finish first." : "Cancelled before processing started.")
        setTimeout(pollStatus, 1000)
    } catch (error) {
        show(errorEl, error.message, "error")
    }
}

const handleRestart = async () => {
    const errorEl = document.getElementById("last-error")
    _autoStartFired = false
    try {
        await jsonFetch(apiUrl("/restart"), { method: "POST" })
        updateStatus("queued", null)
        await refreshResults()
        setTimeout(pollStatus, 1000)
    } catch (error) {
        show(errorEl, error.message, "error")
    }
}

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

const loadPresetOptions = async () => {
    const select = document.getElementById("preset-select")
    if (!select) return
    try {
        const presets = await jsonFetch("/api/v1/presets")
        presets.forEach((name) => {
            const option = document.createElement("option")
            option.value = name
            option.textContent = name.replace(/_/g, " ")
            select.appendChild(option)
        })
    } catch (_err) {
        // presets endpoint unavailable — silently skip
    }
}

const handleApplyPreset = async () => {
    const select = document.getElementById("preset-select")
    const feedback = document.getElementById("preset-feedback")
    if (!select || !select.value) {
        if (feedback) { feedback.hidden = false; feedback.textContent = "Choose a preset first." }
        return
    }
    const presetName = select.value
    try {
        // Single call — copies template, config, evaluation AND all asset files (e.g. omr_marker.jpg)
        await jsonFetch(apiUrl("/preset"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ preset_name: presetName }),
        })

        // Reload the JSON doc editors from the server so they reflect the new files
        const docNames = ["template", "config", "evaluation"]
        for (const docName of docNames) {
            const box = document.querySelector(`.json-box[data-doc="${docName}"]`)
            if (!box) continue
            try {
                const doc = await jsonFetch(apiUrl(`/documents/${docName}`))
                const textarea = box.querySelector("[data-doc-textarea]")
                if (textarea) textarea.value = doc.content ? JSON.stringify(doc.content, null, 2) : ""
                const state = getEditorState(box)
                state.doc = doc.content || null
                const statusEl = box.querySelector("[data-doc-status]")
                if (statusEl) statusEl.textContent = doc.content ? "present" : "empty"
            } catch (_err) { /* doc might not exist in this preset */ }
        }

        // Reload asset list so the newly-copied marker files show as present
        location.reload()

        if (feedback) {
            feedback.hidden = false
            feedback.textContent = `Preset "${presetName.replace(/_/g, " ")}" applied.`
            setTimeout(() => { feedback.hidden = true }, 3000)
        }
    } catch (err) {
        if (feedback) { feedback.hidden = false; feedback.textContent = `Error: ${err.message}` }
    }
}


// Collapsible panels (persisted via localStorage)
const PANEL_STATE_PREFIX = "omr.batchDetail."
const panelStateMemory = {}

const getPanelState = (stateId) => {
    const key = `${PANEL_STATE_PREFIX}${stateId}.open`
    try {
        const value = window.localStorage.getItem(key)
        if (value === null) return panelStateMemory[key] === true
        return value === "true"
    } catch (_err) {
        return panelStateMemory[key] === true
    }
}

const setPanelState = (stateId, open) => {
    const key = `${PANEL_STATE_PREFIX}${stateId}.open`
    panelStateMemory[key] = open === true
    try {
        window.localStorage.setItem(key, open ? "true" : "false")
    } catch (_err) { /* private-browsing fallback: kept in panelStateMemory */ }
}

const applyPanelState = (toggle, body, open) => {
    if (!toggle || !body) return
    toggle.setAttribute("aria-expanded", open ? "true" : "false")
    const chevron = toggle.querySelector(".panel-toggle-chevron")
    if (chevron) chevron.textContent = open ? "\u25bc" : "\u25b6"
    if (open) body.removeAttribute("hidden")
    else body.setAttribute("hidden", "")
}

const setupCollapsiblePanel = (stateId, domPrefix) => {
    const toggle = document.getElementById(`${domPrefix}-toggle`)
    const body = document.getElementById(`${domPrefix}-body`)
    if (!toggle || !body) return
    const initialOpen = getPanelState(stateId)
    applyPanelState(toggle, body, initialOpen)
    toggle.addEventListener("click", () => {
        const nextOpen = toggle.getAttribute("aria-expanded") !== "true"
        applyPanelState(toggle, body, nextOpen)
        setPanelState(stateId, nextOpen)
    })
}
document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.getElementById("upload-form")
    if (uploadForm) uploadForm.addEventListener("submit", handleUpload)
    const importForm = document.getElementById("import-form")
    if (importForm) importForm.addEventListener("submit", handleImport)
    const assetForm = document.getElementById("asset-upload-form")
    if (assetForm) assetForm.addEventListener("submit", handleAssetUpload)
    const runBtn = document.getElementById("run-btn")
    if (runBtn) runBtn.addEventListener("click", handleRun)
    const pipelineStartBtn = document.getElementById("pipeline-start-btn")
    if (pipelineStartBtn) pipelineStartBtn.addEventListener("click", handlePipelineStart)
    const stopBtn = document.getElementById("stop-btn")
    if (stopBtn) stopBtn.addEventListener("click", handleStop)
    const restartBtn = document.getElementById("restart-btn")
    if (restartBtn) restartBtn.addEventListener("click", handleRestart)
    const rotationForm = document.getElementById("rotation-form")
    if (rotationForm) rotationForm.addEventListener("change", handleRotationChange)
    bootstrapFileBrowser()
    setupCollapsiblePanel("jsonDocsPanel", "json-docs")
    applyPreviewRotation()
    loadPresetOptions()
    const applyPresetBtn = document.getElementById("apply-preset-btn")
    if (applyPresetBtn) applyPresetBtn.addEventListener("click", handleApplyPreset)
    document.addEventListener("click", (event) => {
        handleFileBrowserSelect(event)
        handleDeleteFile(event)
        handleDeleteAsset(event)
        handleResultSelect(event)
        handleViewToggle(event)
        handleEditorMode(event)
        handleSaveDoc(event)
        handleClearDoc(event)
    })
    document.addEventListener("keydown", handleFileBrowserKeyDown)

    document.querySelectorAll(".json-box").forEach((box) => {
        const savedMode = window.localStorage.getItem(getEditorStorageKey(box))
        if (savedMode === "ui") {
            setEditorMode(box, "ui")
        }
    })

    const initialStatus = document.getElementById("status-pill")
    if (initialStatus) {
        const current = initialStatus.textContent.trim()
        if (current === "queued" || current === "running") {
            setTimeout(pollStatus, 1000)
        } else if (current === "done" || current === "failed" || current === "cancelled") {
            // Populate progress strip with stored final stats
            setTimeout(pollStatus, 0)
        }
    }
    _loadSystemInfo()
})

// ---------------------------------------------------------------------------
// Ad-hoc browser-console self-test for handleAutoStart. Runs only when
// window.__OMR_TEST_MODE === true so production page loads never see it.
// Drives handleAutoStart against synthetic status payloads and writes a
// pass/fail summary to console. NOT a substitute for the Python regression
// test in webui/tests/test_auto_start_trigger.py — that one covers the
// status-payload contract; this one covers the JS gating logic.
// ---------------------------------------------------------------------------
const __autoStartTriggerSelfTest = async () => {
    if (typeof window === "undefined" || window.__OMR_TEST_MODE !== true) return
    const results = []
    const baseStatus = {
        auto_start_omr_with_split: true,
        auto_start_omr_min_pages: 10,
        auto_start_omr_require_config: false,
        pdf_split_total: 20,
        pdf_split_pages: 15,
        has_template: true,
        has_config: true,
        status: "created",
    }
    const record = (name, fired, expected) => {
        const ok = fired === expected
        results.push({ name, ok, fired, expected })
    }
    // Stub jsonFetch so the self-test doesn't actually POST /process.
    const origFetch = window.fetch
    let calls = 0
    window.fetch = async () => { calls += 1; return new Response("{}", { status: 200 }) }
    try {
        // Gate 1: disabled
        _autoStartFired = false
        calls = 0
        await handleAutoStart({ ...baseStatus, auto_start_omr_with_split: false })
        record("gate:disabled", calls, 0)
        // Gate 2: split not in flight
        _autoStartFired = false
        calls = 0
        await handleAutoStart({ ...baseStatus, pdf_split_total: 0 })
        record("gate:no_split", calls, 0)
        // Gate 3: not enough pages
        _autoStartFired = false
        calls = 0
        await handleAutoStart({ ...baseStatus, pdf_split_pages: 5 })
        record("gate:few_pages", calls, 0)
        // Gate 4: no template
        _autoStartFired = false
        calls = 0
        await handleAutoStart({ ...baseStatus, has_template: false })
        record("gate:no_template", calls, 0)
        // Gate 5: require_config but no config
        _autoStartFired = false
        calls = 0
        await handleAutoStart({ ...baseStatus, has_config: false, auto_start_omr_require_config: true })
        record("gate:require_config", calls, 0)
        // Gate 6: already running
        _autoStartFired = false
        calls = 0
        await handleAutoStart({ ...baseStatus, status: "running" })
        record("gate:running", calls, 0)
        // Happy path — should fire exactly once
        _autoStartFired = false
        calls = 0
        await handleAutoStart(baseStatus)
        record("happy_path", calls, 1)
        // Gate 7: re-entry is suppressed by _autoStartFired
        calls = 0
        await handleAutoStart(baseStatus)
        record("guard:idempotent", calls, 0)
    } finally {
        window.fetch = origFetch
        _autoStartFired = false
    }
    const failed = results.filter((r) => !r.ok)
    console.log(`[autoStartSelfTest] ${results.length - failed.length}/${results.length} pass`)
    if (failed.length > 0) console.warn("[autoStartSelfTest] failures:", failed)
    return { passed: results.length - failed.length, total: results.length, failures: failed }
}
