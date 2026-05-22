/* settings.js — client logic for the /settings page. */
(() => {
    "use strict"

    const SETTINGS_URL = "/api/v1/settings"
    const META_URL = "/api/v1/settings/meta"

    /** @type {Record<string, unknown>} */
    let originalValues = {}
    /** @type {{descriptions: Record<string,string>, defaults: Record<string,unknown>, mutable_keys: string[]}} */
    let meta = { descriptions: {}, defaults: {}, mutable_keys: [] }

    const form = document.getElementById("settings-form")
    const feedback = document.getElementById("settings-feedback")
    const dirtyBadge = document.getElementById("settings-dirty")
    const saveBtn = document.getElementById("settings-save")
    const reloadBtn = document.getElementById("settings-reload")

    // ────────────────────────────────────────────────────────────────────────
    // Feedback helpers
    // ────────────────────────────────────────────────────────────────────────

    const showFeedback = (message, kind = "info") => {
        if (!feedback) return
        feedback.hidden = false
        feedback.textContent = message
        feedback.classList.remove("error", "success")
        if (kind === "error") feedback.classList.add("error")
        if (kind === "success") feedback.classList.add("success")
    }

    const clearFeedback = () => {
        if (!feedback) return
        feedback.hidden = true
        feedback.textContent = ""
        feedback.classList.remove("error", "success")
    }

    // ────────────────────────────────────────────────────────────────────────
    // Type coercion (form <-> JSON)
    // ────────────────────────────────────────────────────────────────────────

    /** Read the current value of an input element as JSON-shaped data. */
    const readInputValue = (input) => {
        const type = input.dataset.type || "str"
        if (input.type === "checkbox") return Boolean(input.checked)
        const raw = input.value
        if (type === "int") {
            if (raw === "" || raw === null) return null
            const n = Number(raw)
            return Number.isFinite(n) ? Math.trunc(n) : null
        }
        if (type === "float") {
            if (raw === "" || raw === null) return null
            const n = Number(raw)
            return Number.isFinite(n) ? n : null
        }
        // str fields: treat empty string as null so default_preset="" disables
        if (raw === "") return null
        return raw
    }

    /** Write a JSON value into a form input, coercing as needed. */
    const writeInputValue = (input, value) => {
        if (input.type === "checkbox") {
            input.checked = Boolean(value)
            return
        }
        if (value === null || value === undefined) {
            input.value = ""
            return
        }
        input.value = String(value)
    }

    /** Find the input element for a given settings key. */
    const inputForKey = (key) => form.querySelector(`[data-key="${CSS.escape(key)}"]`)

    /** Collect current form values as a flat object. */
    const collectCurrent = () => {
        const out = {}
        for (const input of form.querySelectorAll("[data-key]")) {
            out[input.dataset.key] = readInputValue(input)
        }
        return out
    }

    /** Compare current form state vs original snapshot; return diff. */
    const collectChanged = () => {
        const current = collectCurrent()
        const diff = {}
        for (const [k, v] of Object.entries(current)) {
            const original = originalValues[k]
            // Treat null/undefined as equivalent so empty optional strings don't churn.
            const a = v === undefined ? null : v
            const b = original === undefined ? null : original
            if (a !== b) diff[k] = v
        }
        return diff
    }

    const hasDirtyEdits = () => Object.keys(collectChanged()).length > 0

    const refreshDirtyBadge = () => {
        if (!dirtyBadge) return
        dirtyBadge.hidden = !hasDirtyEdits()
    }

    // ────────────────────────────────────────────────────────────────────────
    // Description injection
    // ────────────────────────────────────────────────────────────────────────

    const applyDescriptions = () => {
        for (const node of form.querySelectorAll("[data-description]")) {
            const key = node.dataset.description
            const text = meta.descriptions && meta.descriptions[key]
            node.textContent = text ? text : ""
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Data loading
    // ────────────────────────────────────────────────────────────────────────

    const populateFromValues = (values) => {
        for (const input of form.querySelectorAll("[data-key]")) {
            const key = input.dataset.key
            if (key in values) writeInputValue(input, values[key])
        }
    }

    const loadAll = async () => {
        clearFeedback()
        try {
            const [valuesResp, metaResp] = await Promise.all([
                fetch(SETTINGS_URL, { cache: "no-store" }),
                fetch(META_URL, { cache: "no-store" }),
            ])

            if (!valuesResp.ok) {
                throw new Error(`Failed to load settings: HTTP ${valuesResp.status}`)
            }
            if (!metaResp.ok) {
                throw new Error(`Failed to load settings metadata: HTTP ${metaResp.status}`)
            }

            const values = await valuesResp.json()
            const metaJson = await metaResp.json()

            originalValues = { ...values }
            meta = {
                descriptions: metaJson.descriptions || {},
                defaults: metaJson.defaults || {},
                mutable_keys: metaJson.mutable_keys || [],
            }

            populateFromValues(values)
            applyDescriptions()
            refreshDirtyBadge()
        } catch (err) {
            showFeedback(`Could not load settings: ${err.message}`, "error")
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Save / Reset / Reload handlers
    // ────────────────────────────────────────────────────────────────────────

    const extractErrorDetail = async (response) => {
        try {
            const data = await response.json()
            if (!data) return `HTTP ${response.status}`
            if (typeof data.detail === "string") return data.detail
            if (Array.isArray(data.detail)) {
                return data.detail
                    .map((d) => {
                        const loc = Array.isArray(d.loc) ? d.loc.join(".") : ""
                        const msg = d.msg || JSON.stringify(d)
                        return loc ? `${loc}: ${msg}` : msg
                    })
                    .join("; ")
            }
            if (data.detail) return JSON.stringify(data.detail)
            return JSON.stringify(data)
        } catch {
            return `HTTP ${response.status}`
        }
    }

    const handleSave = async (event) => {
        if (event) event.preventDefault()
        clearFeedback()

        const diff = collectChanged()
        const changeCount = Object.keys(diff).length
        if (changeCount === 0) {
            showFeedback("No changes to save.", "info")
            return
        }

        saveBtn.disabled = true
        try {
            const response = await fetch(SETTINGS_URL, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                cache: "no-store",
                body: JSON.stringify(diff),
            })

            if (!response.ok) {
                const detail = await extractErrorDetail(response)
                showFeedback(`Save failed (${response.status}): ${detail}`, "error")
                return
            }

            const updated = await response.json()
            originalValues = { ...originalValues, ...updated }
            // Mirror back the canonical values in case the server normalised them.
            populateFromValues(updated)
            refreshDirtyBadge()
            const noun = changeCount === 1 ? "change" : "changes"
            showFeedback(
                `Settings saved. ${changeCount} ${noun} applied immediately.`,
                "success",
            )
        } catch (err) {
            showFeedback(`Save failed: ${err.message}`, "error")
        } finally {
            saveBtn.disabled = false
        }
    }

    const handleReset = (event) => {
        const button = event.target.closest("[data-reset]")
        if (!button) return
        const key = button.dataset.reset
        const input = inputForKey(key)
        if (!input) return
        const defaultValue = meta.defaults ? meta.defaults[key] : undefined
        writeInputValue(input, defaultValue)
        refreshDirtyBadge()
    }

    const handleReload = async () => {
        if (hasDirtyEdits()) {
            const ok = window.confirm(
                "You have unsaved changes. Reload from disk and discard them?",
            )
            if (!ok) return
        }
        await loadAll()
        showFeedback("Reloaded settings from disk.", "info")
    }

    const handleInputChange = () => {
        refreshDirtyBadge()
        // Clear any stale "saved" feedback so the user knows their edits aren't yet applied.
        if (feedback && feedback.classList.contains("success")) clearFeedback()
    }

    // ────────────────────────────────────────────────────────────────────────
    // Bootstrap
    // ────────────────────────────────────────────────────────────────────────

    const init = async () => {
        if (!form) return
        form.addEventListener("submit", handleSave)
        form.addEventListener("click", handleReset)
        form.addEventListener("input", handleInputChange)
        form.addEventListener("change", handleInputChange)
        if (reloadBtn) reloadBtn.addEventListener("click", handleReload)
        await loadAll()
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init)
    } else {
        init()
    }
})()
