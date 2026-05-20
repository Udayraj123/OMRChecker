/* generate_csv.js — client-side logic for the Generate Test CSV page */
(function () {
    "use strict";

    // ---------------------------------------------------------------------------
    // Name generation helpers
    // ---------------------------------------------------------------------------
    const FIRST_NAMES = [
        "Aaron","Abigail","Adam","Adrian","Aisha","Alex","Alicia","Aliyah","Amanda","Amber",
        "Amelia","Andre","Andrew","Angela","Ann","Anthony","Ashley","Ayesha","Barbara","Benjamin",
        "Brandon","Brianna","Caleb","Cameron","Carlos","Carmen","Chantel","Chelsea","Christian","Christine",
        "Christopher","Cindy","Claire","Clayton","Cody","Crystal","Damian","Daniel","Danielle","David",
        "Diana","Dominic","Dylan","Eduardo","Elena","Elizabeth","Emily","Emma","Eric","Ethan",
        "Faith","Fatima","Fernando","Frank","Gabriel","George","Grace","Hannah","Hector","Henry",
        "Imani","Isaiah","Jacob","Jade","James","Janet","Jasmine","Jason","Jennifer","Jessica",
        "Joel","Jonathan","Jordan","Jose","Joshua","Juan","Julian","Kayla","Kevin","Kiran",
        "Kyle","Laura","Lauren","Leah","Leonardo","Leslie","Liam","Lisa","Logan","Lucas",
        "Luis","Madison","Marcus","Maria","Mason","Matthew","Maya","Mia","Michael","Michelle",
        "Miguel","Mira","Nathan","Natalie","Nicholas","Nicole","Noah","Nora","Olivia","Omar",
        "Patrick","Paul","Peter","Rachel","Rebecca","Richard","Robert","Ryan","Samantha","Samuel",
        "Sandra","Sara","Sarah","Shawn","Sofia","Stephanie","Steven","Susan","Taylor","Thomas",
        "Timothy","Tyler","Victoria","Vincent","Whitney","William","Yasmine","Zachary","Zoe","Zara"
    ];
    const LAST_NAMES = [
        "Adams","Alexander","Allen","Anderson","Andrews","Archer","Armstrong","Atkins","Austin","Bailey",
        "Baker","Banks","Barnes","Bell","Bennett","Bishop","Black","Blake","Boyd","Brooks",
        "Brown","Bryan","Burke","Burns","Butler","Campbell","Carr","Carter","Chambers","Chapman",
        "Charles","Clarke","Coleman","Collins","Cook","Cooper","Cox","Crawford","Cruz","Davis",
        "Dean","Dixon","Edwards","Ellis","Evans","Ferguson","Fisher","Fleming","Fletcher","Ford",
        "Foster","Fox","Francis","Fraser","Freeman","Garcia","Gibson","Gill","Gordon","Graham",
        "Grant","Gray","Green","Griffin","Hall","Hamilton","Harris","Harrison","Hart","Harvey",
        "Hayes","Henderson","Henry","Hill","Holmes","Howard","Hughes","Hunter","Jackson","James",
        "Jenkins","Johnson","Jones","Joseph","Kelly","Kennedy","King","Knight","Lambert","Lawrence",
        "Lewis","Long","Lopez","Martin","Martinez","Mason","Mathews","Mitchell","Moore","Morgan",
        "Morris","Morrison","Murray","Nelson","Newton","Nichols","Noel","Oliver","Owens","Palmer",
        "Parker","Patterson","Payne","Perry","Peters","Phillips","Pierre","Porter","Powell","Price",
        "Ramkissoon","Reid","Richards","Richardson","Roberts","Robinson","Rogers","Ross","Russell","Sanchez",
        "Sanders","Scott","Shaw","Singh","Smith","Spencer","Stewart","Sullivan","Taylor","Thomas",
        "Thompson","Torres","Turner","Walker","Ward","Watson","White","Williams","Wilson","Wood"
    ];

    let _nameIdx = 0;
    const _usedNames = new Set();

    function resetNames() {
        _nameIdx = 0;
        _usedNames.clear();
    }

    function randomName() {
        // Try a fresh combination; fall back to indexed if we exhaust uniqueness
        for (let attempt = 0; attempt < 10; attempt++) {
            const f = FIRST_NAMES[Math.floor(Math.random() * FIRST_NAMES.length)];
            const l = LAST_NAMES[Math.floor(Math.random() * LAST_NAMES.length)];
            const name = `${f} ${l}`;
            if (!_usedNames.has(name)) {
                _usedNames.add(name);
                return name;
            }
        }
        // Fallback: append index to guarantee uniqueness
        _nameIdx++;
        return `${FIRST_NAMES[_nameIdx % FIRST_NAMES.length]} ${LAST_NAMES[_nameIdx % LAST_NAMES.length]} ${_nameIdx}`;
    }

    // ---------------------------------------------------------------------------
    // CSV generation (runs in the browser — no server round-trip needed)
    // ---------------------------------------------------------------------------

    function buildCsvContent(count, schoolName, examName, candidateStart, nameStyle) {
        const rows = [["student_name", "school_name", "exam_name", "candidate_number", "output_file"]];
        resetNames();
        let candNum = BigInt(candidateStart);
        for (let i = 1; i <= count; i++) {
            const name = nameStyle === "random" ? randomName() : `Student ${i}`;
            const cand = String(candNum).padStart(10, "0");
            // output_file: slug of student name
            const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
            rows.push([name, schoolName, examName, cand, `${slug}.png`]);
            candNum++;
        }
        return rows.map(r => r.map(escapeCsv).join(",")).join("\r\n");
    }

    function escapeCsv(value) {
        const s = String(value ?? "");
        if (s.includes(",") || s.includes('"') || s.includes("\n") || s.includes("\r")) {
            return `"${s.replace(/"/g, '""')}"`;
        }
        return s;
    }

    function triggerDownload(csvContent, filename) {
        const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // ---------------------------------------------------------------------------
    // Preview rendering
    // ---------------------------------------------------------------------------

    const PREVIEW_ROWS = 5;

    function renderPreview(count, schoolName, examName, candidateStart, nameStyle) {
        const previewEl = document.getElementById("gen-preview");
        const labelEl = document.getElementById("preview-label");
        const tbody = document.getElementById("preview-body");

        resetNames();
        let candNum = BigInt(candidateStart);
        const names = nameStyle === "random" ? randomName : (i) => `Student ${i}`;

        tbody.innerHTML = "";
        const shown = Math.min(count, PREVIEW_ROWS);
        for (let i = 1; i <= shown; i++) {
            const name = nameStyle === "random" ? randomName() : `Student ${i}`;
            const cand = String(candNum).padStart(10, "0");
            const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
            const tr = document.createElement("tr");
            [name, schoolName, examName, cand, `${slug}.png`].forEach(val => {
                const td = document.createElement("td");
                td.textContent = val;
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
            candNum++;
        }

        if (count > PREVIEW_ROWS) {
            const tr = document.createElement("tr");
            const td = document.createElement("td");
            td.colSpan = 5;
            td.className = "muted small";
            td.style.textAlign = "center";
            td.textContent = `… and ${count - PREVIEW_ROWS} more rows`;
            tr.appendChild(td);
            tbody.appendChild(tr);
        }

        labelEl.textContent = `(first ${Math.min(count, PREVIEW_ROWS)} of ${count} rows)`;
        previewEl.style.display = "";
    }

    // ---------------------------------------------------------------------------
    // Validation helpers
    // ---------------------------------------------------------------------------

    function showError(msg) {
        const el = document.getElementById("gen-error");
        el.textContent = msg;
        el.style.display = msg ? "" : "none";
    }

    function getFormValues() {
        const count = parseInt(document.getElementById("g-count").value, 10);
        const schoolName = document.getElementById("g-school").value.trim();
        const examName = document.getElementById("g-exam").value.trim();
        const candidateStart = document.getElementById("g-start").value.trim();
        const nameStyle = document.querySelector('input[name="name_style"]:checked')?.value ?? "numbered";

        const errors = [];
        if (!Number.isInteger(count) || count < 1 || count > 10000) {
            errors.push("Number of students must be between 1 and 10,000.");
        }
        if (!schoolName) errors.push("School name is required.");
        if (!examName) errors.push("Exam name is required.");
        if (!/^\d{10}$/.test(candidateStart)) {
            errors.push("Candidate number start must be exactly 10 digits.");
        }
        return { count, schoolName, examName, candidateStart, nameStyle, errors };
    }

    // ---------------------------------------------------------------------------
    // Wire up form
    // ---------------------------------------------------------------------------

    document.addEventListener("DOMContentLoaded", () => {
        const form = document.getElementById("gen-form");
        const submitBtn = document.getElementById("gen-submit");

        form.addEventListener("submit", (e) => {
            e.preventDefault();
            showError("");

            const { count, schoolName, examName, candidateStart, nameStyle, errors } = getFormValues();
            if (errors.length) {
                showError(errors.join(" "));
                return;
            }

            submitBtn.disabled = true;
            submitBtn.textContent = "Generating…";

            // Use setTimeout(0) so the browser can repaint "Generating…" before the
            // synchronous CSV build blocks the main thread for large counts.
            setTimeout(() => {
                try {
                    const csv = buildCsvContent(count, schoolName, examName, candidateStart, nameStyle);
                    const ts = new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");
                    triggerDownload(csv, `test_students_${count}_${ts}.csv`);
                    renderPreview(count, schoolName, examName, candidateStart, nameStyle);
                } catch (err) {
                    showError(`Failed to generate CSV: ${err.message}`);
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.textContent = "Generate & Download CSV";
                }
            }, 0);
        });
    });
})();
