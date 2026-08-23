const SAMPLE_JD =
  "We seek a Senior Software Engineer with 5+ years experience. Key skills: Python, SQL, machine learning, AWS, Docker, Kubernetes, REST APIs, and CI/CD. The candidate should have strong leadership and communication skills and work with cross-functional teams.";

const form = document.getElementById("score-form");
const sampleBtn = document.getElementById("sample-btn");
const submitBtn = document.getElementById("submit-btn");
const errorEl = document.getElementById("form-error");
const results = document.getElementById("results");
const empty = document.getElementById("empty-state");
const body = document.getElementById("result-body");

const fileInput = form.resume;
const textInput = form.resume_text;

fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files.length) textInput.value = "";
});
textInput.addEventListener("input", () => {
  if (textInput.value.trim()) fileInput.value = "";
});

sampleBtn.addEventListener("click", () => {
  form.job_description.value = SAMPLE_JD;
});

function fitLabel(score) {
  if (score < 40) return "No fit";
  if (score < 65) return "Potential fit";
  return "Good fit";
}

function setRing(score) {
  const circ = 2 * Math.PI * 52;
  const offset = circ * (1 - Math.max(0, Math.min(score, 100)) / 100);
  document.getElementById("ring-value").style.strokeDashoffset = String(offset);
}

function chips(el, items, emptyText) {
  el.innerHTML = "";
  if (!items || !items.length) {
    el.innerHTML = `<span class="chip">${emptyText}</span>`;
    return;
  }
  for (const item of items) {
    const span = document.createElement("span");
    span.className = "chip";
    span.textContent = item;
    el.appendChild(span);
  }
}

function list(el, items) {
  el.innerHTML = "";
  for (const item of items || []) {
    const li = document.createElement("li");
    li.textContent = item;
    el.appendChild(li);
  }
  if (!el.children.length) el.innerHTML = "<li>None listed.</li>";
}

function render(data) {
  empty.hidden = true;
  body.hidden = false;
  results.dataset.empty = "false";
  const score = data.composite_score ?? 0;
  document.getElementById("score-value").textContent = score.toFixed(1);
  document.getElementById("fit-label").textContent = data.blocked ? "Blocked by guardrails" : fitLabel(score);
  const preview = (data.extraction?.text || "").replace(/\s+/g, " ").trim();
  const snippet = preview ? preview.slice(0, 88) + (preview.length > 88 ? "…" : "") : "";
  document.getElementById("score-meta").textContent = data.blocked
    ? "The pipeline stopped before scoring."
    : `Scored this run: “${snippet}” (${data.extraction?.char_count || preview.length} chars) · ${data.semantic_match?.backend || "vectors"} · ${data.llm_backend}`;

  const semantic = data.semantic_match;
  document.getElementById("semantic-meta").textContent = semantic
    ? `Calibrated similarity (not raw cosine×100). Hits below the embedding floor score 0. Document ${semantic.document_score.toFixed(1)} · requirement coverage ${semantic.requirement_coverage.toFixed(1)} (mean of per-requirement calibrated max-hits)`
    : "Semantic matcher did not run.";
  const align = document.getElementById("alignments");
  align.innerHTML = "";
  for (const item of semantic?.alignments || []) {
    const li = document.createElement("li");
    li.textContent = `${item.score.toFixed(0)} · JD: ${item.requirement} ↔ Resume: ${item.resume_span}`;
    align.appendChild(li);
  }
  if (!align.children.length) align.innerHTML = "<li>No requirement chunks to align.</li>";
  setRing(data.blocked ? 0 : score);

  const dims = data.critique
    ? [data.critique.grammar, data.critique.formatting, data.critique.relevance, data.critique.impact]
    : [];
  const bars = document.getElementById("dimension-bars");
  bars.innerHTML = "";
  for (const dim of dims) {
    const row = document.createElement("div");
    row.className = "bar";
    row.innerHTML = `<span>${dim.name}</span><i><span style="width:${dim.score}%"></span></i><strong>${Math.round(dim.score)}</strong>`;
    bars.appendChild(row);
  }

  const semanticReqs = semantic?.matched_requirements?.length || semantic?.gap_requirements?.length;
  const matched = semanticReqs ? semantic.matched_requirements : data.ats?.matched_skills;
  const gaps = semanticReqs ? semantic.gap_requirements : data.ats?.missing_skills;
  document.getElementById("ats-meta").textContent = semanticReqs
    ? "JD phrases from this run. Stronger means calibrated score ≥ 25; weaker includes misses and weak word-overlap."
    : (data.ats
      ? `Skill coverage ${data.ats.required_skill_coverage.toFixed(1)}% · keyword similarity ${data.ats.keyword_similarity.toFixed(1)}%`
      : "Requirement evidence unavailable.");
  chips(document.getElementById("matched-chips"), matched, "No stronger alignments");
  chips(document.getElementById("missing-chips"), gaps, "No weaker alignments");

  document.getElementById("summary").textContent = data.critique?.summary || "No critique generated.";
  list(document.getElementById("strengths"), data.critique?.strengths);
  list(document.getElementById("gaps"), data.critique?.gaps);
  list(document.getElementById("rewrites"), data.critique?.rewrite_suggestions);

  const g = document.getElementById("guardrails");
  g.innerHTML = "";
  for (const finding of data.guardrails || []) {
    const li = document.createElement("li");
    li.className = `sev-${finding.severity}`;
    li.textContent =
      finding.code === "prompt_injection"
        ? `${finding.code}: ${finding.message} (does not freeze scores or chips; scoring still used the resume and JD.)`
        : `${finding.code}: ${finding.message}`;
    g.appendChild(li);
  }
  if (!g.children.length) g.innerHTML = "<li>No guardrail findings.</li>";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorEl.hidden = true;
  submitBtn.disabled = true;
  submitBtn.textContent = "Scoring…";
  const payload = new FormData();
  payload.set("job_description", form.job_description.value);
  payload.set("candidate_id", form.candidate_id.value || "anonymous");
  const pasted = (form.resume_text.value || "").trim();
  const file = form.resume.files && form.resume.files[0];
  if (pasted) payload.set("resume_text", pasted);
  else if (file && file.size) payload.set("resume", file);
  try {
    const response = await fetch("/analyze", { method: "POST", body: payload });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Analysis failed");
    render(data);
  } catch (err) {
    errorEl.hidden = false;
    errorEl.textContent = err.message || String(err);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Get score";
  }
});
