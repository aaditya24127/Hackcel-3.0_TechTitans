/**
 * Dashboard Frontend Controller
 * Manages video source controls, uploads, real-time polling, and UI updates.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const btnStartCamera = document.getElementById("btnStartCamera");
    const btnStopCamera = document.getElementById("btnStopCamera");
    const cameraSelect = document.getElementById("cameraSelect");
    
    const btnSelectFile = document.getElementById("btnSelectFile");
    const videoFileInput = document.getElementById("videoFileInput");
    const selectedFileName = document.getElementById("selectedFileName");
    const btnProcessVideo = document.getElementById("btnProcessVideo");
    
    const progressContainer = document.getElementById("progressContainer");
    const progressBar = document.getElementById("progressBar");
    const progressText = document.getElementById("progressText");

    const systemStatusDot = document.getElementById("systemStatusDot");
    const systemStatusText = document.getElementById("systemStatusText");
    const dbStatusDot = document.getElementById("dbStatusDot");
    const dbStatusText = document.getElementById("dbStatusText");
    const fpsBadge = document.getElementById("fpsBadge");
    const liveVehicleCount = document.getElementById("liveVehicleCount");

    const currentVehicleType = document.getElementById("currentVehicleType");
    const currentVehicleNumber = document.getElementById("currentVehicleNumber");
    const currentVehicleConf = document.getElementById("currentVehicleConf");
    const currentPlateConf = document.getElementById("currentPlateConf");
    const currentParkingAllocation = document.getElementById("currentParkingAllocation");
    const currentParkingSlot = document.getElementById("currentParkingSlot");

    const incomingTableBody = document.getElementById("incomingTableBody");
    const recordCount = document.getElementById("recordCount");

    let uploadedServerFilename = null;
    let pollInterval = null;

    // -------------------------------------------------------------
    // 1. Initial State & Health Checks
    // -------------------------------------------------------------
    checkDatabaseConnection();
    startTelemetryPolling();

    async function checkDatabaseConnection() {
        try {
            const res = await fetch("/api/db_status");
            const data = await res.json();
            if (data.connected) {
                dbStatusDot.className = "status-dot dot-running";
                dbStatusText.textContent = "MySQL Connected";
            } else {
                dbStatusDot.className = "status-dot dot-error";
                dbStatusText.textContent = "MySQL Offline";
            }
        } catch (e) {
            dbStatusDot.className = "status-dot dot-error";
            dbStatusText.textContent = "DB Error";
        }
    }

    // -------------------------------------------------------------
    // 2. Camera Controls
    // -------------------------------------------------------------
    btnStartCamera.addEventListener("click", async () => {
        const cameraIndex = cameraSelect.value;
        try {
            btnStartCamera.disabled = true;
            btnStopCamera.disabled = false;
            btnProcessVideo.disabled = true;

            const res = await fetch("/api/camera/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ camera_index: cameraIndex })
            });
            const data = await res.json();
            if (data.status !== "success") {
                alert("Error starting camera: " + data.message);
                btnStartCamera.disabled = false;
                btnStopCamera.disabled = true;
            }
        } catch (err) {
            console.error("Camera start error:", err);
            btnStartCamera.disabled = false;
            btnStopCamera.disabled = true;
        }
    });

    btnStopCamera.addEventListener("click", async () => {
        try {
            await fetch("/api/camera/stop", { method: "POST" });
            btnStartCamera.disabled = false;
            btnStopCamera.disabled = true;
            if (uploadedServerFilename) {
                btnProcessVideo.disabled = false;
            }
        } catch (err) {
            console.error("Stop error:", err);
        }
    });

    // -------------------------------------------------------------
    // 3. Media File Upload & Processing (Video / Image)
    // -------------------------------------------------------------
    btnSelectFile.addEventListener("click", () => {
        videoFileInput.click();
    });

    videoFileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            selectedFileName.textContent = file.name;
            btnProcessVideo.disabled = false;
            uploadedServerFilename = null;
            progressContainer.style.display = "none";
            progressBar.style.width = "0%";
        } else {
            selectedFileName.textContent = "No file selected";
            btnProcessVideo.disabled = true;
        }
    });

    btnProcessVideo.addEventListener("click", async () => {
        const file = videoFileInput.files[0];
        if (!file && !uploadedServerFilename) return;

        btnProcessVideo.disabled = true;
        btnStartCamera.disabled = true;
        btnStopCamera.disabled = false;
        progressContainer.style.display = "flex";
        progressText.textContent = "Uploading media...";
        progressBar.style.width = "10%";

        try {
            // If not already uploaded, upload first
            if (!uploadedServerFilename) {
                const formData = new FormData();
                formData.append("file", file);

                const uploadRes = await fetch("/api/file/upload", {
                    method: "POST",
                    body: formData
                });
                const uploadData = await uploadRes.json();

                if (uploadData.status !== "success") {
                    alert("Upload error: " + uploadData.message);
                    resetVideoControls();
                    return;
                }
                uploadedServerFilename = uploadData.filename;
            }

            // Start Processing
            progressText.textContent = "Processing media...";
            progressBar.style.width = "30%";

            const procRes = await fetch("/api/file/process", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename: uploadedServerFilename })
            });
            const procData = await procRes.json();

            if (procData.status !== "success") {
                alert("Processing error: " + procData.message);
                resetVideoControls();
            }
        } catch (err) {
            console.error("Media processing error:", err);
            resetVideoControls();
        }
    });

    function resetVideoControls() {
        btnProcessVideo.disabled = false;
        btnStartCamera.disabled = false;
        btnStopCamera.disabled = true;
    }

    // -------------------------------------------------------------
    // 4. Real-time Telemetry & Data Polling
    // -------------------------------------------------------------
    function startTelemetryPolling() {
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(async () => {
            await fetchStatus();
            await fetchHistory();
        }, 700);
    }

    async function fetchStatus() {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();

            // Update Vehicle Count
            liveVehicleCount.textContent = data.vehicle_count || 0;
            fpsBadge.textContent = `${data.fps || 0.0} FPS`;

            // Update Status Indicator & Controls
            if (data.status === "RUNNING_CAMERA") {
                systemStatusDot.className = "status-dot dot-running";
                systemStatusText.textContent = "Camera Live";
                btnStartCamera.disabled = true;
                btnStopCamera.disabled = false;
            } else if (data.status === "RUNNING_VIDEO") {
                systemStatusDot.className = "status-dot dot-running";
                systemStatusText.textContent = "Processing Video";
                btnStartCamera.disabled = true;
                btnStopCamera.disabled = false;
                
                // Update Progress
                progressContainer.style.display = "flex";
                progressBar.style.width = `${data.progress_percent}%`;
                progressText.textContent = `Processing: ${data.progress_percent}%`;
            } else if (data.status === "RUNNING_IMAGE") {
                systemStatusDot.className = "status-dot dot-running";
                systemStatusText.textContent = "Processing Image";
                btnStartCamera.disabled = true;
                btnStopCamera.disabled = false;
                
                progressContainer.style.display = "flex";
                progressBar.style.width = `80%`;
                progressText.textContent = `Processing Image...`;
            } else if (data.status === "COMPLETED") {
                systemStatusDot.className = "status-dot dot-completed";
                systemStatusText.textContent = "Processing complete.";
                btnStartCamera.disabled = false;
                btnStopCamera.disabled = true;
                btnProcessVideo.disabled = false;

                progressContainer.style.display = "flex";
                progressBar.style.width = "100%";
                progressText.textContent = "Processing complete.";
            } else if (data.status === "ERROR") {
                systemStatusDot.className = "status-dot dot-error";
                systemStatusText.textContent = "Error";
                btnStartCamera.disabled = false;
                btnStopCamera.disabled = true;
            } else {
                systemStatusDot.className = "status-dot dot-idle";
                systemStatusText.textContent = "System Idle";
                btnStartCamera.disabled = false;
                btnStopCamera.disabled = true;
            }

            // Update Current Vehicle Card
            if (data.current_vehicle) {
                const cv = data.current_vehicle;
                currentVehicleType.textContent = cv.vehicle_type || "-";
                currentVehicleNumber.textContent = cv.vehicle_number || "-";
                currentVehicleConf.textContent = cv.vehicle_confidence ? `${cv.vehicle_confidence}%` : "0.0%";
                currentPlateConf.textContent = cv.plate_confidence ? `${cv.plate_confidence}%` : "0.0%";
                currentParkingSlot.textContent = cv.parking_slot || "-";

                const alloc = (cv.parking_allocation || "").toUpperCase();
                if (alloc === "YES") {
                    currentParkingAllocation.className = "badge badge-yes";
                    currentParkingAllocation.textContent = "YES";
                } else if (alloc === "NO") {
                    currentParkingAllocation.className = "badge badge-no";
                    currentParkingAllocation.textContent = "NO";
                } else {
                    currentParkingAllocation.className = "badge badge-neutral";
                    currentParkingAllocation.textContent = "-";
                }
            }
        } catch (err) {
            console.debug("Error fetching status:", err);
        }
    }

    async function fetchHistory() {
        try {
            const res = await fetch("/api/history?limit=30");
            const data = await res.json();

            if (data.status === "success" && Array.isArray(data.history)) {
                renderHistoryTable(data.history);
            }
        } catch (err) {
            console.debug("Error fetching history:", err);
        }
    }

    function renderHistoryTable(records) {
        recordCount.textContent = `${records.length} records`;

        if (records.length === 0) {
            incomingTableBody.innerHTML = `
                <tr>
                    <td colspan="6" class="text-center text-muted">No detections recorded yet</td>
                </tr>
            `;
            return;
        }

        let html = "";
        records.forEach((row) => {
            const alloc = (row.parking_allocation || "").toUpperCase();
            const isYes = alloc === "YES";
            const badgeClass = isYes ? "badge badge-yes" : "badge badge-no";
            
            const plateDisplay = row.display_vehicle_number || row.vehicle_number || "Not clear";
            const isUnclear = plateDisplay === "Not clear" || plateDisplay.toLowerCase() === "not clear";
            const plateClass = isUnclear ? "text-unclear" : "font-mono highlight";

            const status = row.ocr_status || (isUnclear ? "NOT_CLEAR" : "CONFIRMED");
            let statusBadge = `<span class="badge badge-subtle">${escapeHtml(status)}</span>`;
            if (status === "CONFIRMED") {
                statusBadge = `<span class="badge badge-yes">CONFIRMED</span>`;
            } else if (status === "PROCESSING") {
                statusBadge = `<span class="badge badge-pending">PROCESSING</span>`;
            } else if (status === "NOT_CLEAR") {
                statusBadge = `<span class="badge badge-subtle">NOT_CLEAR</span>`;
            }

            const slotDisplay = row.parking_slot || (isYes ? "-" : "-");

            html += `
                <tr>
                    <td><strong>${row.series_number}</strong></td>
                    <td class="${plateClass}">${escapeHtml(plateDisplay)}</td>
                    <td><span class="${badgeClass}">${escapeHtml(alloc || "NO")}</span></td>
                    <td class="font-mono">${escapeHtml(slotDisplay)}</td>
                    <td>${statusBadge}</td>
                    <td>${escapeHtml(row.detected_at)}</td>
                </tr>
            `;
        });

        incomingTableBody.innerHTML = html;
    }

    function escapeHtml(str) {
        if (!str) return "";
        return str
            .toString()
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
