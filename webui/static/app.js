const showFeedback = (element, message, kind = "info") => {
    if (!element) return
    element.hidden = false
    element.textContent = message
    element.classList.remove("error", "success")
    if (kind === "error") element.classList.add("error")
    if (kind === "success") element.classList.add("success")
}

const postJson = async (url, body) => {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    })
    const text = await response.text()
    const data = text ? JSON.parse(text) : {}
    if (!response.ok) {
        const detail = data.detail || `Request failed (${response.status})`
        throw new Error(detail)
    }
    return data
}

const deleteJson = async (url) => {
    const response = await fetch(url, { method: "DELETE" })
    if (!response.ok && response.status !== 204) {
        const data = await response.json().catch(() => ({}))
        throw new Error(data.detail || `Request failed (${response.status})`)
    }
}

const handleCreateBatch = async (event) => {
    event.preventDefault()
    const form = event.currentTarget
    const feedback = document.getElementById("create-feedback")
    const input = form.querySelector("#batch-name")
    const name = input.value.trim()
    if (!name) return

    try {
        const batch = await postJson("/api/v1/batches", { name })
        showFeedback(feedback, `Created ${batch.name}. Redirecting...`, "success")
        window.location.href = `/batches/${batch.id}`
    } catch (error) {
        showFeedback(feedback, error.message, "error")
    }
}

const handleDeleteBatch = async (event) => {
    const button = event.target.closest("[data-delete-batch]")
    if (!button) return
    const batchId = button.dataset.deleteBatch
    if (!window.confirm("Delete this batch and all its files?")) return
    try {
        await deleteJson(`/api/v1/batches/${batchId}`)
        window.location.reload()
    } catch (error) {
        window.alert(error.message)
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const createForm = document.getElementById("create-batch-form")
    if (createForm) createForm.addEventListener("submit", handleCreateBatch)
    document.addEventListener("click", handleDeleteBatch)
})

// ── Log panel ─────────────────────────────────────────────────────────────

;(function initLogPanel() {
    const panel     = document.getElementById("log-panel")
    const header    = document.getElementById("log-panel-header")
    const output    = document.getElementById("log-output")
    const badge     = document.getElementById("log-badge")
    const clearBtn  = document.getElementById("log-clear-btn")
    const autoBtn   = document.getElementById("log-autoscroll-btn")
    if (!panel || !output) return

    const MAX_LINES = 2000
    let lineCount   = 0
    let errorCount  = 0
    let warnCount   = 0
    let autoScroll  = true

    const updateBadge = () => {
        badge.textContent = lineCount > 999 ? "999+" : String(lineCount)
        badge.classList.remove("has-error", "has-warn")
        if (errorCount > 0) badge.classList.add("has-error")
        else if (warnCount > 0) badge.classList.add("has-warn")
    }

    const classForLine = (text) => {
        if (/\bERROR\b|\bCRITICAL\b/.test(text))   return "log-line-error"
        if (/\bWARNING\b|\bWARN\b/.test(text))      return "log-line-warn"
        if (/\bINFO\b/.test(text))                  return "log-line-info"
        return "log-line-debug"
    }

    const appendLine = (text) => {
        // Trim oldest lines if over the cap
        while (output.childElementCount >= MAX_LINES) {
            const removed = output.firstElementChild
            if (removed) {
                if (removed.classList.contains("log-line-error")) errorCount = Math.max(0, errorCount - 1)
                if (removed.classList.contains("log-line-warn"))  warnCount  = Math.max(0, warnCount  - 1)
                lineCount = Math.max(0, lineCount - 1)
                output.removeChild(removed)
            }
        }

        const cls = classForLine(text)
        const span = document.createElement("span")
        span.className = cls
        span.textContent = text + "\n"
        output.appendChild(span)
        lineCount++
        if (cls === "log-line-error") errorCount++
        if (cls === "log-line-warn")  warnCount++
        updateBadge()

        if (autoScroll) output.scrollTop = output.scrollHeight

        // Auto-open panel on first error/warning if collapsed
        if ((cls === "log-line-error" || cls === "log-line-warn") && panel.classList.contains("collapsed")) {
            togglePanel(true)
        }
    }

    const togglePanel = (forceOpen) => {
        const willCollapse = forceOpen === true ? false : !panel.classList.contains("collapsed")
        panel.classList.toggle("collapsed", willCollapse)
        document.body.classList.toggle("log-panel-open",      !willCollapse)
        document.body.classList.toggle("log-panel-collapsed",  willCollapse)
        if (!willCollapse && autoScroll) output.scrollTop = output.scrollHeight
    }

    header.addEventListener("click", () => togglePanel())

    clearBtn.addEventListener("click", () => {
        output.innerHTML = ""
        lineCount = errorCount = warnCount = 0
        updateBadge()
    })

    autoBtn.addEventListener("click", () => {
        autoScroll = !autoScroll
        autoBtn.style.opacity = autoScroll ? "1" : "0.45"
        autoBtn.setAttribute("aria-pressed", String(autoScroll))
        if (autoScroll) output.scrollTop = output.scrollHeight
    })

    // Poll for new log entries every second.
    // Plain HTTP GET works reliably in all browsers and in WebView2, unlike
    // SSE streaming responses which can be silently buffered by WebView2.
    let _logSeq = -1

    const pollLogs = async () => {
        try {
            const res = await fetch(`/api/v1/logs/poll?since=${_logSeq}`)
            if (res.ok) {
                const data = await res.json()
                for (const entry of data.entries) appendLine(entry.msg)
                if (data.latest_seq > _logSeq) _logSeq = data.latest_seq
            }
        } catch (_) { /* server not ready yet — retry next tick */ }
    }

    pollLogs()                          // immediate first poll (gets all history)
    setInterval(pollLogs, 1000)         // then every 1 s

    // Start collapsed but visible
    document.body.classList.add("log-panel-collapsed")
    updateBadge()
})()
