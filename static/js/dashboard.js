/**
 * Real-Time Object Detection & Recognition Dashboard Controller
 * Handles UI interactions, video stream lifecycle, AJAX telemetry polling,
 * dynamic model parameter adjustments, and file uploads.
 */

document.addEventListener("DOMContentLoaded", () => {
    // State
    let currentSourceType = "youtube";
    let isRunning = false;
    let isPaused = false;
    let isRecording = false;
    let uploadedFilePath = "";
    let pollInterval = null;

    // Elements
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabContents = {
        youtube: document.getElementById("youtubeTab"),
        sample: document.getElementById("sampleTab"),
        local: document.getElementById("localTab"),
        webcam: document.getElementById("webcamTab")
    };

    const ytUrlInput = document.getElementById("ytUrlInput");
    const ytPresetSelect = document.getElementById("ytPresetSelect");
    const dropZone = document.getElementById("dropZone");
    const videoFileInput = document.getElementById("videoFileInput");
    const uploadStatus = document.getElementById("uploadStatus");
    const cameraIndex = document.getElementById("cameraIndex");

    const btnStart = document.getElementById("btnStart");
    const btnPause = document.getElementById("btnPause");
    const btnStop = document.getElementById("btnStop");
    const btnRecord = document.getElementById("btnRecord");

    const modelSelect = document.getElementById("modelSelect");
    const confSlider = document.getElementById("confSlider");
    const confValue = document.getElementById("confValue");
    const classCheckboxes = document.querySelectorAll(".cls-check");

    const statusPill = document.getElementById("statusPill");
    const statusText = document.getElementById("statusText");
    const modelPill = document.getElementById("modelPill");
    const liveDot = document.getElementById("liveDot");
    const sourceTag = document.getElementById("sourceTag");
    const videoTitleDisplay = document.getElementById("videoTitleDisplay");
    const fpsBadge = document.getElementById("fpsBadge");
    const latencyBadge = document.getElementById("latencyBadge");
    const recordingIndicator = document.getElementById("recordingIndicator");
    const videoStream = document.getElementById("videoStream");

    // KPI Counters
    const kpiTotal = document.getElementById("kpiTotal");
    const kpiActive = document.getElementById("kpiActive");
    const kpiCars = document.getElementById("kpiCars");
    const kpiCarsActive = document.getElementById("kpiCarsActive");
    const kpiBuses = document.getElementById("kpiBuses");
    const kpiBusesActive = document.getElementById("kpiBusesActive");
    const kpiTrucks = document.getElementById("kpiTrucks");
    const kpiTrucksActive = document.getElementById("kpiTrucksActive");
    const kpiBikes = document.getElementById("kpiBikes");
    const kpiPedestrians = document.getElementById("kpiPedestrians");
    const kpiPedActive = document.getElementById("kpiPedActive");
    const kpiSignals = document.getElementById("kpiSignals");

    // Charts & Logs
    const barCar = document.getElementById("barCar");
    const valCar = document.getElementById("valCar");
    const barPed = document.getElementById("barPed");
    const valPed = document.getElementById("valPed");
    const barBus = document.getElementById("barBus");
    const valBus = document.getElementById("valBus");
    const barTruck = document.getElementById("barTruck");
    const valTruck = document.getElementById("valTruck");
    const barBike = document.getElementById("barBike");
    const valBike = document.getElementById("valBike");

    const logsTableBody = document.getElementById("logsTableBody");
    const logCount = document.getElementById("logCount");
    const toast = document.getElementById("toast");

    // Toast notification utility
    function showToast(message, duration = 3500) {
        toast.textContent = message;
        toast.classList.add("show");
        setTimeout(() => toast.classList.remove("show"), duration);
    }

    // 1. Tab Switching
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            tabButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentSourceType = btn.dataset.source;

            Object.keys(tabContents).forEach(key => {
                if (key === currentSourceType) {
                    tabContents[key].classList.remove("hidden");
                } else {
                    tabContents[key].classList.add("hidden");
                }
            });
        });
    });

    // Preset dropdown auto-fill
    ytPresetSelect.addEventListener("change", (e) => {
        if (e.target.value) {
            ytUrlInput.value = e.target.value;
        }
    });

    // 2. Video Upload Handling
    videoFileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("video_file", file);

        uploadStatus.className = "upload-status";
        uploadStatus.textContent = `Uploading ${file.name}...`;
        uploadStatus.classList.remove("hidden");

        fetch("/api/upload", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                uploadedFilePath = data.filepath;
                uploadStatus.className = "upload-status success";
                uploadStatus.innerHTML = `<i class="fa-solid fa-check"></i> Uploaded: <strong>${data.filename}</strong>`;
                showToast(`Video uploaded: ${data.filename}`);
            } else {
                uploadStatus.className = "upload-status";
                uploadStatus.textContent = `Upload failed: ${data.error}`;
                showToast(`Upload failed: ${data.error}`);
            }
        })
        .catch(err => {
            uploadStatus.textContent = "Error during file upload.";
            showToast("Upload network error.");
        });
    });

    // 3. Execution Controls: Start
    btnStart.addEventListener("click", () => {
        let sourceInput = "";

        if (currentSourceType === "youtube") {
            sourceInput = ytUrlInput.value.trim();
            if (!sourceInput) {
                showToast("Please enter a YouTube video URL.");
                return;
            }
        } else if (currentSourceType === "sample") {
            sourceInput = "sample";
        } else if (currentSourceType === "local") {
            if (!uploadedFilePath) {
                showToast("Please select and upload an MP4/AVI video file first.");
                return;
            }
            sourceInput = uploadedFilePath;
        } else if (currentSourceType === "webcam") {
            sourceInput = cameraIndex.value;
        }

        const selectedClasses = Array.from(classCheckboxes)
            .filter(cb => cb.checked)
            .map(cb => cb.value);

        const payload = {
            source_type: currentSourceType,
            source_input: sourceInput,
            conf_threshold: parseFloat(confSlider.value) / 100.0,
            selected_classes: selectedClasses,
            model_name: modelSelect.value
        };

        btnStart.disabled = true;
        btnStart.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Initializing...`;
        showToast("Resolving video stream & loading neural weights...");

        fetch("/api/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            btnStart.disabled = false;
            btnStart.innerHTML = `<i class="fa-solid fa-play"></i> Start Detection`;

            if (data.success) {
                isRunning = true;
                isPaused = false;
                updateButtonStates();
                videoTitleDisplay.textContent = data.title || "Active Stream";
                sourceTag.textContent = currentSourceType.toUpperCase();
                sourceTag.style.color = "var(--accent-cyan)";
                liveDot.classList.add("active");
                
                // Refresh stream src to ensure fresh connection
                videoStream.src = `/video_feed?t=${Date.now()}`;
                showToast("Detection stream started successfully!");
                startPolling();
            } else {
                showToast(`Start Error: ${data.error}`, 5000);
            }
        })
        .catch(err => {
            btnStart.disabled = false;
            btnStart.innerHTML = `<i class="fa-solid fa-play"></i> Start Detection`;
            showToast("Failed to communicate with detection server.");
        });
    });

    // 4. Pause / Resume
    btnPause.addEventListener("click", () => {
        fetch("/api/pause", { method: "POST" })
        .then(res => res.json())
        .then(data => {
            isPaused = data.is_paused;
            btnPause.innerHTML = isPaused ? `<i class="fa-solid fa-play"></i> Resume` : `<i class="fa-solid fa-pause"></i> Pause`;
            btnPause.className = isPaused ? "btn btn-primary" : "btn btn-warning";
            showToast(isPaused ? "Video detection paused." : "Video detection resumed.");
        });
    });

    // 5. Stop
    btnStop.addEventListener("click", () => {
        fetch("/api/stop", { method: "POST" })
        .then(res => res.json())
        .then(data => {
            isRunning = false;
            isPaused = false;
            updateButtonStates();
            liveDot.classList.remove("active");
            sourceTag.textContent = "STOPPED";
            sourceTag.style.color = "var(--accent-rose)";
            showToast("Detection stopped.");
        });
    });

    // 6. Record MP4
    btnRecord.addEventListener("click", () => {
        if (!isRecording) {
            fetch("/api/record/start", { method: "POST" })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    isRecording = true;
                    btnRecord.innerHTML = `<i class="fa-solid fa-stop"></i> Stop Record`;
                    btnRecord.classList.remove("btn-secondary");
                    btnRecord.classList.add("btn-danger");
                    recordingIndicator.classList.remove("hidden");
                    showToast(`Recording started: ${data.filename}`);
                } else {
                    showToast(data.error);
                }
            });
        } else {
            fetch("/api/record/stop", { method: "POST" })
            .then(res => res.json())
            .then(data => {
                isRecording = false;
                btnRecord.innerHTML = `<i class="fa-solid fa-circle-dot"></i> Record MP4`;
                btnRecord.classList.remove("btn-danger");
                btnRecord.classList.add("btn-secondary");
                recordingIndicator.classList.add("hidden");
                showToast(`Video saved to recordings/${data.filename}!`);
            });
        }
    });

    function updateButtonStates() {
        btnStart.disabled = isRunning;
        btnPause.disabled = !isRunning;
        btnStop.disabled = !isRunning;
        btnPause.innerHTML = `<i class="fa-solid fa-pause"></i> Pause`;
        btnPause.className = "btn btn-warning";
    }

    // 7. Dynamic Settings (Confidence Slider, Model Selection, Class Checkboxes)
    confSlider.addEventListener("input", (e) => {
        confValue.textContent = `${e.target.value}%`;
    });

    confSlider.addEventListener("change", (e) => {
        sendSettingsUpdate();
    });

    modelSelect.addEventListener("change", () => {
        modelPill.textContent = modelSelect.value.replace(".pt", "").toUpperCase();
        sendSettingsUpdate();
    });

    classCheckboxes.forEach(cb => {
        cb.addEventListener("change", () => {
            sendSettingsUpdate();
        });
    });

    function sendSettingsUpdate() {
        const selectedClasses = Array.from(classCheckboxes)
            .filter(cb => cb.checked)
            .map(cb => cb.value);

        fetch("/api/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                conf_threshold: parseFloat(confSlider.value) / 100.0,
                selected_classes: selectedClasses,
                model_name: modelSelect.value
            })
        });
    }

    // 8. Telemetry Polling Loop
    function startPolling() {
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(fetchTelemetry, 600);
    }

    function fetchTelemetry() {
        fetch("/api/status")
        .then(res => res.json())
        .then(data => {
            // Update FPS and Latency Badges
            fpsBadge.textContent = data.fps.toFixed(1);
            latencyBadge.textContent = data.inference_ms.toFixed(1);

            // Status Pill
            if (data.is_running) {
                if (data.is_paused) {
                    statusPill.className = "status-pill status-paused";
                    statusText.textContent = "Pipeline Paused";
                } else {
                    statusPill.className = "status-pill status-running";
                    statusText.textContent = "Inference Active";
                }
            } else {
                statusPill.className = "status-pill status-ready";
                statusText.textContent = "System Ready";
            }

            // Counters
            kpiTotal.textContent = data.total_cumulative || 0;
            kpiActive.textContent = data.active_objects || 0;

            const cum = data.cumulative_counts || {};
            const act = data.active_counts || {};

            kpiCars.textContent = cum.car || 0;
            kpiCarsActive.textContent = act.car || 0;

            kpiBuses.textContent = cum.bus || 0;
            kpiBusesActive.textContent = act.bus || 0;

            kpiTrucks.textContent = cum.truck || 0;
            kpiTrucksActive.textContent = act.truck || 0;

            kpiBikes.textContent = (cum.motorcycle || 0) + (cum.bicycle || 0);

            kpiPedestrians.textContent = cum.person || 0;
            kpiPedActive.textContent = act.person || 0;

            kpiSignals.textContent = cum["traffic light"] || 0;

            // Composition Bar Percentage Calculations
            const totalVehAndPed = (cum.car || 0) + (cum.person || 0) + (cum.bus || 0) + (cum.truck || 0) + ((cum.motorcycle || 0) + (cum.bicycle || 0));
            if (totalVehAndPed > 0) {
                const pCar = Math.round(((cum.car || 0) / totalVehAndPed) * 100);
                const pPed = Math.round(((cum.person || 0) / totalVehAndPed) * 100);
                const pBus = Math.round(((cum.bus || 0) / totalVehAndPed) * 100);
                const pTruck = Math.round(((cum.truck || 0) / totalVehAndPed) * 100);
                const pBike = Math.round((((cum.motorcycle || 0) + (cum.bicycle || 0)) / totalVehAndPed) * 100);

                barCar.style.width = `${pCar}%`; valCar.textContent = `${pCar}%`;
                barPed.style.width = `${pPed}%`; valPed.textContent = `${pPed}%`;
                barBus.style.width = `${pBus}%`; valBus.textContent = `${pBus}%`;
                barTruck.style.width = `${pTruck}%`; valTruck.textContent = `${pTruck}%`;
                barBike.style.width = `${pBike}%`; valBike.textContent = `${pBike}%`;
            }

            // Real-Time Event Log Stream Table
            const logs = data.logs || [];
            logCount.textContent = `${logs.length} Events`;
            if (logs.length > 0) {
                logsTableBody.innerHTML = logs.map(l => `
                    <tr>
                        <td><code>${l.time}</code></td>
                        <td>#${l.frame}</td>
                        <td><span class="badge badge-accent">ID #${l.id}</span></td>
                        <td><strong>${l.class}</strong></td>
                        <td>${l.confidence}</td>
                        <td><code>${l.centroid}</code></td>
                    </tr>
                `).join("");
            }
        })
        .catch(() => {});
    }

    // Start background telemetry poll immediately
    startPolling();
});

