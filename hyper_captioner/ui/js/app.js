// Hyper Captioner - Interactive CyberDeck UI Controller
document.addEventListener("DOMContentLoaded", () => {
  // State
  let datasetPath = "";
  let galleryImages = [];
  let currentReviewIndex = 0;
  let batchPollInterval = null;
  let lockedTagsSet = new Set();
  let presetsData = {};

  // DOM Elements
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  const datasetPathInput = document.getElementById("dataset-path");
  const browseBtn = document.getElementById("browse-btn");
  const scanBtn = document.getElementById("scan-btn");

  const statTotal = document.getElementById("stat-total");
  const statCaptioned = document.getElementById("stat-captioned");
  const statMissing = document.getElementById("stat-missing");
  const galleryContainer = document.getElementById("gallery-container");
  const galleryCount = document.getElementById("gallery-count");

  const vramTelemetry = document.getElementById("vram-telemetry");
  const vramDetails = document.getElementById("vram-details");
  const batchBadge = document.getElementById("batch-badge");

  // Captioning Controls
  const presetSelect = document.getElementById("preset-select");
  const captionModeSelect = document.getElementById("caption-mode-select");
  const captionFormatSelect = document.getElementById("caption-format-select");
  const triggerPlacementSelect = document.getElementById("trigger-placement-select");
  const loraStrategySelect = document.getElementById("lora-strategy-select");
  const triggerWordInput = document.getElementById("trigger-word-input");
  const refDescInput = document.getElementById("ref-desc-input");
  const underscoresChk = document.getElementById("underscores-chk");
  const backupChk = document.getElementById("backup-chk");
  const overwriteChk = document.getElementById("overwrite-chk");

  const previewBtn = document.getElementById("preview-5-btn");
  const startBatchBtn = document.getElementById("start-batch-btn");
  const cancelBatchBtn = document.getElementById("cancel-batch-btn");
  const batchProgressBar = document.getElementById("batch-progress-bar");
  const batchCounter = document.getElementById("batch-counter");
  const batchPercent = document.getElementById("batch-percent");
  const batchLogContainer = document.getElementById("batch-log-container");

  // Review Elements
  const reviewImg = document.getElementById("review-img");
  const reviewFilename = document.getElementById("review-filename");
  const reviewIndex = document.getElementById("review-index");
  const reviewCaptionText = document.getElementById("review-caption-text");
  const reviewChipsContainer = document.getElementById("review-chips-container");
  const reviewPrevBtn = document.getElementById("review-prev-btn");
  const reviewNextBtn = document.getElementById("review-next-btn");
  const reviewSaveBtn = document.getElementById("review-save-btn");
  const reviewRegenBtn = document.getElementById("review-regen-btn");
  const reviewCopyBtn = document.getElementById("review-copy-btn");
  const reviewStatusBadge = document.getElementById("review-status-badge");
  const reviewValidationBadge = document.getElementById("review-validation-badge");

  // Vocabulary Elements
  const vocabSearchInput = document.getElementById("vocab-search-input");
  const vocabSearchBtn = document.getElementById("vocab-search-btn");
  const vocabReloadBtn = document.getElementById("vocab-reload-btn");
  const normOldTag = document.getElementById("norm-old-tag");
  const normNewTag = document.getElementById("norm-new-tag");
  const normApplyBtn = document.getElementById("norm-apply-btn");
  const normDeleteBtn = document.getElementById("norm-delete-btn");
  const vocabTbody = document.getElementById("vocab-tbody");

  // Export Elements
  const exportBtn = document.getElementById("export-btn");
  const exportFormatSelect = document.getElementById("export-format-select");
  const exportFilenameInput = document.getElementById("export-filename-input");
  const exportMsg = document.getElementById("export-msg");
  const backupBtn = document.getElementById("backup-btn");
  const backupMsg = document.getElementById("backup-msg");

  // Modal
  const previewModal = document.getElementById("preview-modal");
  const previewModalBody = document.getElementById("preview-modal-body");
  const modalCloseBtn = document.getElementById("modal-close-btn");
  const modalCancelBtn = document.getElementById("modal-cancel-btn");
  const modalAcceptBtn = document.getElementById("modal-accept-btn");

  // -----------------------------------------------------------
  // Tab Switching
  // -----------------------------------------------------------
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-tab");
      tabBtns.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(target).classList.add("active");

      if (target === "vocabulary-tab" && datasetPath) {
        loadVocabulary();
      }
    });
  });

  function switchTab(tabId) {
    const btn = document.querySelector(`[data-tab="${tabId}"]`);
    if (btn) btn.click();
  }

  // -----------------------------------------------------------
  // Hardware & Telemetry Poller
  // -----------------------------------------------------------
  async function pollHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      if (data.vram && data.vram.device_name) {
        vramTelemetry.innerHTML = `<span style="color:var(--cyan);">${data.vram.device_name}:</span> ${data.vram.free_gib}GB Free / ${data.vram.total_gib}GB Total`;
        vramDetails.innerHTML = `
          <strong>Device:</strong> ${data.vram.device_name}<br>
          <strong>Total VRAM:</strong> ${data.vram.total_gib} GiB<br>
          <strong>Free VRAM:</strong> ${data.vram.free_gib} GiB<br>
          <strong>Allocated:</strong> ${data.vram.allocated_gib} GiB | <strong>Reserved:</strong> ${data.vram.reserved_gib} GiB
        `;
      } else {
        vramTelemetry.innerHTML = `<span style="color:var(--amber);">CPU Mode</span>`;
        vramDetails.innerHTML = `Running in CPU inference mode.`;
      }

      if (data.batch_active) {
        batchBadge.style.display = "inline-flex";
        if (!batchPollInterval) startBatchPolling();
      } else {
        batchBadge.style.display = "none";
      }
    } catch (e) {
      // Offline / connecting
    }
  }
  setInterval(pollHealth, 4000);
  pollHealth();

  // -----------------------------------------------------------
  // Presets Loader
  // -----------------------------------------------------------
  async function loadPresets() {
    try {
      const res = await fetch("/api/presets");
      const data = await res.json();
      presetsData = data.details || {};
      presetSelect.innerHTML = "";
      data.presets.forEach(pName => {
        const opt = document.createElement("option");
        opt.value = pName;
        opt.textContent = pName;
        presetSelect.appendChild(opt);
      });
    } catch (e) {
      console.error("Failed to load presets:", e);
    }
  }
  loadPresets();

  presetSelect.addEventListener("change", () => {
    const p = presetsData[presetSelect.value];
    if (p) {
      if (p.caption_mode) captionModeSelect.value = p.caption_mode;
      if (p.caption_format && captionFormatSelect) captionFormatSelect.value = p.caption_format;
      if (p.trigger_placement && triggerPlacementSelect) triggerPlacementSelect.value = p.trigger_placement;
      if (p.lora_strategy) loraStrategySelect.value = p.lora_strategy;
      if (p.character) {
        triggerWordInput.value = p.character.trigger_word || "";
        refDescInput.value = p.character.reference_description || "";
      }
      if (p.keep_underscores !== undefined) underscoresChk.checked = p.keep_underscores;
    }
  });

  // -----------------------------------------------------------
  // Dataset Browse & Scan
  // -----------------------------------------------------------
  browseBtn.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/browse", { method: "POST" });
      const data = await res.json();
      if (data.path) {
        datasetPathInput.value = data.path;
        datasetPath = data.path;
        triggerScan();
      }
    } catch (e) {
      console.error(e);
    }
  });

  scanBtn.addEventListener("click", () => {
    datasetPath = datasetPathInput.value.trim();
    if (datasetPath) triggerScan();
  });

  async function triggerScan() {
    datasetPath = datasetPathInput.value.trim();
    if (!datasetPath) return;

    scanBtn.disabled = true;
    scanBtn.textContent = "Scanning...";

    try {
      const res = await fetch("/api/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_path: datasetPath, verify: false })
      });
      const data = await res.json();

      statTotal.textContent = data.total_images || 0;
      statCaptioned.textContent = data.with_captions_count || 0;
      statMissing.textContent = data.missing_captions_count || 0;

      await loadGallery();
    } catch (e) {
      alert("Scan failed: " + e.message);
    } finally {
      scanBtn.disabled = false;
      scanBtn.textContent = "Scan Dataset";
    }
  }

  async function loadGallery() {
    if (!datasetPath) return;
    try {
      const res = await fetch(`/api/gallery?path=${encodeURIComponent(datasetPath)}`);
      const data = await res.json();
      galleryImages = data.images || [];
      galleryCount.textContent = `${galleryImages.length} Images`;

      galleryContainer.innerHTML = "";
      if (galleryImages.length === 0) {
        galleryContainer.innerHTML = `<div style="grid-column:1/-1; text-align:center; color:var(--text-dim); padding:40px;">No supported images found in this folder.</div>`;
        return;
      }

      galleryImages.forEach((img, idx) => {
        const item = document.createElement("div");
        item.className = "gallery-item";
        item.innerHTML = `
          <img src="/api/image?path=${encodeURIComponent(img.path)}" loading="lazy" alt="${img.filename}">
          <div class="${img.has_caption ? 'badge-captioned' : 'badge-missing'}"></div>
        `;
        item.addEventListener("click", () => {
          currentReviewIndex = idx;
          loadReviewItem(idx);
          switchTab("review-tab");
        });
        galleryContainer.appendChild(item);
      });
    } catch (e) {
      console.error("Failed to load gallery:", e);
    }
  }

  // -----------------------------------------------------------
  // Review Workstation
  // -----------------------------------------------------------
  async function loadReviewItem(index) {
    if (!galleryImages || galleryImages.length === 0) return;
    if (index < 0) index = 0;
    if (index >= galleryImages.length) index = galleryImages.length - 1;
    currentReviewIndex = index;

    const item = galleryImages[index];
    reviewFilename.textContent = item.filename;
    reviewIndex.textContent = `${index + 1} / ${galleryImages.length}`;
    reviewImg.src = `/api/image?path=${encodeURIComponent(item.path)}`;
    reviewStatusBadge.textContent = item.has_caption ? "Captioned" : "Missing Caption";
    reviewStatusBadge.style.color = item.has_caption ? "var(--emerald)" : "var(--rose)";

    try {
      const res = await fetch(`/api/review/item?image_path=${encodeURIComponent(item.path)}`);
      const data = await res.json();
      reviewCaptionText.value = data.caption || "";

      if (reviewValidationBadge) {
        const vStatus = (data.validation_status || "valid").toLowerCase();
        if (vStatus === "repaired") {
          reviewValidationBadge.textContent = "REPAIRED ⚠️";
          reviewValidationBadge.style.color = "var(--amber)";
          reviewValidationBadge.style.borderColor = "var(--amber)";
        } else if (vStatus === "rejected") {
          reviewValidationBadge.textContent = "REJECTED 🔄";
          reviewValidationBadge.style.color = "var(--rose)";
          reviewValidationBadge.style.borderColor = "var(--rose)";
        } else {
          reviewValidationBadge.textContent = "VALID ✓";
          reviewValidationBadge.style.color = "var(--emerald)";
          reviewValidationBadge.style.borderColor = "var(--emerald)";
        }
      }

      const tokenList = (data.traceability && data.traceability.length > 0)
        ? data.traceability
        : ((data.tokens && data.tokens.length > 0)
            ? data.tokens
            : (data.tags || []));
      renderProvenanceChips(tokenList);
    } catch (e) {
      console.error("Failed to load review item:", e);
    }
  }

  function renderProvenanceChips(tokens) {
    reviewChipsContainer.innerHTML = "";
    if (!tokens || tokens.length === 0) {
      reviewChipsContainer.innerHTML = `<span style="color:var(--text-dim); font-size:11px;">No individual tokens parsed.</span>`;
      return;
    }

    tokens.forEach(t => {
      const chip = document.createElement("div");
      const key = t.text.toLowerCase();
      if (t.locked) {
        lockedTagsSet.add(key);
      }
      const isLocked = lockedTagsSet.has(key);
      const cat = t.category || t.primary_category || (t.categories && t.categories[0]) || "token";

      chip.className = `tag-chip ${isLocked ? 'locked' : ''}`;
      chip.innerHTML = `
        <span>${t.text}</span>
        <span class="chip-source" style="margin-left:4px;">[${cat}]</span>
        <button class="chip-lock-btn" title="Toggle Lock">${isLocked ? '🔒' : '🔓'}</button>
      `;

      const lockBtn = chip.querySelector(".chip-lock-btn");
      lockBtn.addEventListener("click", () => {
        if (lockedTagsSet.has(key)) {
          lockedTagsSet.delete(key);
          chip.classList.remove("locked");
          lockBtn.textContent = "🔓";
        } else {
          lockedTagsSet.add(key);
          chip.classList.add("locked");
          lockBtn.textContent = "🔒";
        }
      });

      reviewChipsContainer.appendChild(chip);
    });
  }

  reviewPrevBtn.addEventListener("click", () => {
    if (currentReviewIndex > 0) loadReviewItem(currentReviewIndex - 1);
  });

  reviewNextBtn.addEventListener("click", () => {
    if (currentReviewIndex < galleryImages.length - 1) loadReviewItem(currentReviewIndex + 1);
  });

  reviewSaveBtn.addEventListener("click", async () => {
    if (!galleryImages[currentReviewIndex]) return;
    const item = galleryImages[currentReviewIndex];
    const newCap = reviewCaptionText.value.trim();

    try {
      const res = await fetch("/api/review/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_path: item.path,
          caption: newCap,
          locked_tags: Array.from(lockedTagsSet)
        })
      });
      const data = await res.json();
      reviewStatusBadge.textContent = "Saved ✓";
      reviewStatusBadge.style.color = "var(--emerald)";
      item.has_caption = true;
    } catch (e) {
      alert("Failed to save caption: " + e.message);
    }
  });

  reviewRegenBtn.addEventListener("click", async () => {
    if (!galleryImages[currentReviewIndex]) return;
    const item = galleryImages[currentReviewIndex];

    reviewRegenBtn.disabled = true;
    reviewRegenBtn.textContent = "Regenerating...";

    try {
      const res = await fetch("/api/review/regenerate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_path: item.path,
          preset_name: presetSelect.value,
          caption_mode: captionModeSelect.value,
          caption_format: captionFormatSelect ? captionFormatSelect.value : "tags",
          trigger_placement: triggerPlacementSelect ? triggerPlacementSelect.value : "prepend",
          trigger_word: triggerWordInput.value.trim(),
          keep_underscores: underscoresChk.checked,
          locked_tags: Array.from(lockedTagsSet)
        })
      });
      const data = await res.json();
      reviewCaptionText.value = data.caption || "";

      if (reviewValidationBadge) {
        const vStatus = (data.validation_status || "valid").toLowerCase();
        if (vStatus === "repaired") {
          reviewValidationBadge.textContent = "REPAIRED ⚠️";
          reviewValidationBadge.style.color = "var(--amber)";
          reviewValidationBadge.style.borderColor = "var(--amber)";
        } else if (vStatus === "rejected") {
          reviewValidationBadge.textContent = "REJECTED 🔄";
          reviewValidationBadge.style.color = "var(--rose)";
          reviewValidationBadge.style.borderColor = "var(--rose)";
        } else {
          reviewValidationBadge.textContent = "VALID ✓";
          reviewValidationBadge.style.color = "var(--emerald)";
          reviewValidationBadge.style.borderColor = "var(--emerald)";
        }
      }

      const tokenList = (data.tokens && data.tokens.length > 0)
        ? data.tokens
        : ((data.traceability && data.traceability.length > 0)
            ? data.traceability
            : (data.tags || []));
      renderProvenanceChips(tokenList);
      reviewStatusBadge.textContent = "Regenerated ✓";
      item.has_caption = true;
    } catch (e) {
      alert("Regeneration failed: " + e.message);
    } finally {
      reviewRegenBtn.disabled = false;
      reviewRegenBtn.textContent = "🔄 Regenerate (R)";
    }
  });

  reviewCopyBtn.addEventListener("click", () => {
    navigator.clipboard.writeText(reviewCaptionText.value);
    const orig = reviewCopyBtn.textContent;
    reviewCopyBtn.textContent = "Copied ✓";
    setTimeout(() => { reviewCopyBtn.textContent = orig; }, 1500);
  });

  // Keyboard Shortcuts for Review
  document.addEventListener("keydown", (e) => {
    // Only handle if in review tab and not typing in input/textarea
    const inReview = document.getElementById("review-tab").classList.contains("active");
    const isEditing = ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName);

    if (e.ctrlKey && e.key === "s") {
      e.preventDefault();
      reviewSaveBtn.click();
      return;
    }

    if (inReview && !isEditing) {
      if (e.key === "a" || e.key === "ArrowLeft") {
        reviewPrevBtn.click();
      } else if (e.key === "d" || e.key === "ArrowRight") {
        reviewNextBtn.click();
      } else if (e.key === "r") {
        reviewRegenBtn.click();
      }
    }
  });

  // -----------------------------------------------------------
  // Preview System
  // -----------------------------------------------------------
  previewBtn.addEventListener("click", async () => {
    datasetPath = datasetPathInput.value.trim();
    if (!datasetPath) {
      alert("Please specify a dataset folder first.");
      return;
    }

    previewBtn.disabled = true;
    previewBtn.textContent = "Generating 5 Previews...";

    try {
      const res = await fetch("/api/batch/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_path: datasetPath,
          preset_name: presetSelect.value,
          sample_count: 5,
          caption_mode: captionModeSelect.value,
          caption_format: captionFormatSelect ? captionFormatSelect.value : "tags",
          trigger_placement: triggerPlacementSelect ? triggerPlacementSelect.value : "prepend",
          trigger_word: triggerWordInput.value.trim(),
          keep_underscores: underscoresChk.checked
        })
      });
      const data = await res.json();
      renderPreviewModal(data.previews || []);
    } catch (e) {
      alert("Preview generation failed: " + e.message);
    } finally {
      previewBtn.disabled = false;
      previewBtn.textContent = "👁️ Preview 5 Images";
    }
  });

  function renderPreviewModal(previews) {
    previewModalBody.innerHTML = "";
    if (previews.length === 0) {
      previewModalBody.innerHTML = `<div style="text-align:center; padding:30px; color:var(--text-dim);">No images found for preview.</div>`;
    } else {
      previews.forEach((p, i) => {
        const card = document.createElement("div");
        card.style = "display:flex; gap:16px; margin-bottom:16px; padding:12px; background:var(--bg-card); border-radius:var(--radius-sm); border:1px solid rgba(255,255,255,0.08);";
        const vBadge = p.validation_status ? `<span class="status-badge" style="font-size:10px; margin-left:8px;">${p.validation_status.toUpperCase()}</span>` : "";
        card.innerHTML = `
          <img src="/api/image?path=${encodeURIComponent(p.image_path)}" style="width:110px; height:110px; object-fit:cover; border-radius:4px;">
          <div style="flex:1;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
              <span style="font-size:12px; font-weight:bold; color:var(--cyan);">#${i+1} ${p.filename} (${p.execution_time.toFixed(2)}s)</span>
              ${vBadge}
            </div>
            <div style="font-size:12px; line-height:1.5; color:var(--text-main);">${p.caption}</div>
          </div>
        `;
        previewModalBody.appendChild(card);
      });
    }
    previewModal.classList.add("active");
  }

  modalCloseBtn.addEventListener("click", () => previewModal.classList.remove("active"));
  modalCancelBtn.addEventListener("click", () => previewModal.classList.remove("active"));
  modalAcceptBtn.addEventListener("click", () => {
    previewModal.classList.remove("active");
    startBatchBtn.click();
  });

  // -----------------------------------------------------------
  // Batch Execution & Progress Monitoring
  // -----------------------------------------------------------
  startBatchBtn.addEventListener("click", async () => {
    datasetPath = datasetPathInput.value.trim();
    if (!datasetPath) {
      alert("Please specify a dataset folder.");
      return;
    }

    startBatchBtn.disabled = true;
    cancelBatchBtn.style.display = "inline-flex";

    try {
      const res = await fetch("/api/batch/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_path: datasetPath,
          preset_name: presetSelect.value,
          caption_mode: captionModeSelect.value,
          caption_format: captionFormatSelect ? captionFormatSelect.value : "tags",
          trigger_placement: triggerPlacementSelect ? triggerPlacementSelect.value : "prepend",
          trigger_word: triggerWordInput.value.trim(),
          keep_underscores: underscoresChk.checked,
          overwrite: overwriteChk.checked,
          do_backup: backupChk.checked
        })
      });
      const data = await res.json();
      startBatchPolling();
    } catch (e) {
      alert("Failed to start batch: " + e.message);
      startBatchBtn.disabled = false;
      cancelBatchBtn.style.display = "none";
    }
  });

  cancelBatchBtn.addEventListener("click", async () => {
    await fetch("/api/batch/cancel", { method: "POST" });
    cancelBatchBtn.disabled = true;
    cancelBatchBtn.textContent = "Cancelling...";
  });

  function startBatchPolling() {
    if (batchPollInterval) clearInterval(batchPollInterval);
    batchPollInterval = setInterval(async () => {
      try {
        const res = await fetch("/api/batch/status");
        const data = await res.json();

        const prog = data.progress || {};
        const valStats = (prog.valid_count !== undefined) ? ` [V:${prog.valid_count} R:${prog.repaired_count} X:${prog.rejected_count}]` : "";
        batchCounter.textContent = `${prog.current || 0} / ${prog.total || 0} Processed${valStats}`;
        batchPercent.textContent = `${prog.percent || 0.0}%`;
        batchProgressBar.style.width = `${prog.percent || 0}%`;

        // Render live batch logs
        if (data.recent_records && data.recent_records.length > 0) {
          batchLogContainer.innerHTML = "";
          data.recent_records.forEach((rec, idx) => {
            const line = document.createElement("div");
            const numStr = String(idx + 1).padStart(3, "0");
            if (rec.status === "completed") {
              line.className = "log-item-ok";
              line.textContent = `${numStr} ✓ ${rec.image} -> "${rec.caption.substring(0, 75)}..."`;
            } else if (rec.status === "failed") {
              line.className = "log-item-fail";
              line.textContent = `${numStr} ✗ ${rec.image} -> Failed`;
            } else {
              line.className = "log-item-proc";
              line.textContent = `${numStr} ⌛ ${rec.image} [Processing]`;
            }
            batchLogContainer.appendChild(line);
          });
          batchLogContainer.scrollTop = batchLogContainer.scrollHeight;
        }

        if (!data.active) {
          clearInterval(batchPollInterval);
          batchPollInterval = null;
          startBatchBtn.disabled = false;
          cancelBatchBtn.style.display = "none";
          cancelBatchBtn.disabled = false;
          cancelBatchBtn.textContent = "Cancel Run";
          triggerScan();
        }
      } catch (e) {
        console.error("Batch status poll error:", e);
      }
    }, 1200);
  }

  // -----------------------------------------------------------
  // Vocabulary Management
  // -----------------------------------------------------------
  async function loadVocabulary(query = "") {
    if (!datasetPath) return;
    vocabTbody.innerHTML = `<tr><td colspan="4" style="text-align:center; padding:20px; color:var(--cyan);">Indexing dataset vocabulary...</td></tr>`;

    try {
      const url = `/api/vocab/index?dataset_path=${encodeURIComponent(datasetPath)}${query ? '&query=' + encodeURIComponent(query) : ''}`;
      const res = await fetch(url);
      const data = await res.json();
      const results = data.results || [];

      vocabTbody.innerHTML = "";
      if (results.length === 0) {
        vocabTbody.innerHTML = `<tr><td colspan="4" style="text-align:center; padding:20px; color:var(--text-dim);">No tags found matching query.</td></tr>`;
        return;
      }

      results.forEach(item => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${item.tag}</strong></td>
          <td style="color:var(--text-muted);">tag</td>
          <td style="color:var(--cyan); font-weight:bold;">${item.count}</td>
          <td>
            <button class="btn btn-secondary" style="padding:2px 8px; font-size:11px;" title="Set as target for normalization">Select</button>
          </td>
        `;
        tr.querySelector("button").addEventListener("click", () => {
          normOldTag.value = item.tag;
          normNewTag.focus();
        });
        vocabTbody.appendChild(tr);
      });
    } catch (e) {
      console.error("Vocabulary load failed:", e);
    }
  }

  vocabSearchBtn.addEventListener("click", () => loadVocabulary(vocabSearchInput.value.trim()));
  vocabSearchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") loadVocabulary(vocabSearchInput.value.trim());
  });
  vocabReloadBtn.addEventListener("click", () => {
    vocabSearchInput.value = "";
    loadVocabulary();
  });

  normApplyBtn.addEventListener("click", async () => {
    const oldTag = normOldTag.value.trim();
    const newTag = normNewTag.value.trim();
    if (!oldTag || !newTag || !datasetPath) {
      alert("Specify dataset, old tag, and new canonical tag.");
      return;
    }

    try {
      const res = await fetch("/api/vocab/replace", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_path: datasetPath, old_tag: oldTag, new_tag: newTag })
      });
      const data = await res.json();
      alert(`[OK] Normalized across ${data.modified_files} files.`);
      normOldTag.value = "";
      normNewTag.value = "";
      loadVocabulary();
      triggerScan();
    } catch (e) {
      alert("Normalization failed: " + e.message);
    }
  });

  normDeleteBtn.addEventListener("click", async () => {
    const target = normOldTag.value.trim();
    if (!target || !datasetPath) {
      alert("Specify tag to delete.");
      return;
    }
    if (!confirm(`Are you sure you want to completely delete "${target}" from all files in the dataset?`)) return;

    try {
      const res = await fetch("/api/vocab/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_path: datasetPath, target_tag: target })
      });
      const data = await res.json();
      alert(`[OK] Deleted tag from ${data.modified_files} files.`);
      normOldTag.value = "";
      loadVocabulary();
      triggerScan();
    } catch (e) {
      alert("Delete failed: " + e.message);
    }
  });

  // -----------------------------------------------------------
  // Export & Backup
  // -----------------------------------------------------------
  const exportFormatPlaceholders = {
    txt: "captions.txt",
    zip: "captions_txt_sidecars.zip",
    sidecars: "sidecars_synced.log",
    csv: "captions.csv",
    json: "captions.json"
  };

  exportFormatSelect.addEventListener("change", () => {
    exportFilenameInput.placeholder = exportFormatPlaceholders[exportFormatSelect.value] || "captions.txt";
  });

  exportBtn.addEventListener("click", async () => {
    const activePath = datasetPathInput.value.trim() || datasetPath;
    if (!activePath) {
      alert("Please specify a dataset folder.");
      return;
    }
    exportBtn.disabled = true;
    exportBtn.textContent = "Exporting...";
    try {
      const res = await fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_path: activePath,
          format: exportFormatSelect.value,
          output_filename: exportFilenameInput.value.trim() || undefined
        })
      });
      const data = await res.json();
      const downloadUrl = `/api/export/download?path=${encodeURIComponent(data.exported_file)}`;
      exportMsg.innerHTML = `[OK] Export complete: <strong>${data.filename || data.exported_file}</strong> &nbsp; <a href="${downloadUrl}" download style="color:var(--cyan); text-decoration:underline; font-weight:bold;">📥 Download File</a>`;
    } catch (e) {
      alert("Export failed: " + e.message);
    } finally {
      exportBtn.disabled = false;
      exportBtn.textContent = "Export File";
    }
  });

  backupBtn.addEventListener("click", async () => {
    if (!datasetPath) {
      alert("Please specify a dataset folder.");
      return;
    }
    try {
      const res = await fetch(`/api/backup?dataset_path=${encodeURIComponent(datasetPath)}`, { method: "POST" });
      const data = await res.json();
      backupMsg.textContent = `Backup created: ${data.backup_folder}`;
    } catch (e) {
      alert("Backup failed: " + e.message);
    }
  });

});
