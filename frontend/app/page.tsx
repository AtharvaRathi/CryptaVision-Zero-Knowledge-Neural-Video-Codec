"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";

// ── Types for Stream A Semantic Analytics ────────────────────────────────────
interface LatencyBreakdown {
  pose?: number;
  clip?: number;
  total?: number;
}

interface SemanticFrame {
  frame_id: number;
  pts_s: number;
  num_humans: number;
  pose_keypoints: number[][][]; // [person][keypoint][x, y]
  pose_scores?: number[][];
  clip_embedding: number[];
  latency_ms?: LatencyBreakdown;
  timestamp?: string;
}

interface AnomalyAlert {
  id: string;
  timestamp: string;
  severity: "low" | "medium" | "high";
  message: string;
  frame_id: number;
}

// ── Types for Stream B Protected Latents ─────────────────────────────────────
interface LatentItem {
  frame_id: number;
  face_idx: number;
  box: [number, number, number, number]; // [x1, y1, x2, y2]
  pts_ns: number;
  ckks_bytes_b64: string;
  byte_size: number;
  sha256: string;
}

interface StreamBResponse {
  count: number;
  timestamp: string;
  officer_token_verified: boolean;
  latents: LatentItem[];
}

export default function SecurityComplianceDashboard() {
  // ── Configuration State ───────────────────────────────────────────────────
  const [apiBaseUrl, setApiBaseUrl] = useState<string>("http://localhost:8000");
  const [connectionStatus, setConnectionStatus] = useState<"CONNECTING" | "CONNECTED" | "DISCONNECTED">("DISCONNECTED");

  // ── Stream A State ────────────────────────────────────────────────────────
  const [currentFrame, setCurrentFrame] = useState<SemanticFrame | null>(null);
  const [processedCount, setProcessedCount] = useState<number>(0);
  const [lastFrameAt, setLastFrameAt] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  const feedLive = connectionStatus === "CONNECTED" && lastFrameAt !== null && now - lastFrameAt < 15000;
  const feedStatus = connectionStatus !== "CONNECTED" ? connectionStatus
    : lastFrameAt === null ? "WAITING FOR FRAMES" : feedLive ? "LIVE" : "FEED STALLED";
  const [alerts, setAlerts] = useState<AnomalyAlert[]>([]);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // ── Stream B Forensic Modal State ─────────────────────────────────────────
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [vaultToken, setVaultToken] = useState<string>("");
  const [incidentId, setIncidentId] = useState<string>("INC-2026-X88");
  const [investigationReason, setInvestigationReason] = useState<string>("Authorized forensic identification under compliance warrant");
  const [isFetchingLatents, setIsFetchingLatents] = useState<boolean>(false);
  const [forensicResult, setForensicResult] = useState<StreamBResponse | null>(null);
  const [forensicError, setForensicError] = useState<string | null>(null);

  // ══════════════════════════════════════════════════════════════════════════
  //  1. Server-Sent Events (SSE) Listener for Stream A
  // ══════════════════════════════════════════════════════════════════════════
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    let eventSource: EventSource | null = null;
    let reconnectTimeout: NodeJS.Timeout | null = null;

    const connectToStreamA = () => {
      setConnectionStatus("CONNECTING");
      setLastFrameAt(null);
      setCurrentFrame(null);
      const url = `${apiBaseUrl.replace(/\/$/, "")}/analytics/stream-a`;
      eventSource = new EventSource(url);

      eventSource.onopen = () => {
        setConnectionStatus("CONNECTED");
      };

      // Listen for semantic_frame custom SSE events
      eventSource.addEventListener("semantic_frame", (e: MessageEvent) => {
        try {
          const frameData: SemanticFrame = JSON.parse(e.data);
          frameData.timestamp = new Date().toLocaleTimeString();
          setCurrentFrame(frameData);
          setLastFrameAt(Date.now());
          setNow(Date.now());
          setProcessedCount((prev) => prev + 1);

          // Anomaly detection rules
          if (frameData.num_humans > 2) {
            setAlerts((prev) => [
              {
                id: Math.random().toString(36).substring(2, 9),
                timestamp: new Date().toLocaleTimeString(),
                severity: "medium",
                message: `Multi-person presence detected (${frameData.num_humans} subjects in zone)`,
                frame_id: frameData.frame_id,
              },
              ...prev.slice(0, 19),
            ]);
          }

          if (frameData.latency_ms?.total && frameData.latency_ms.total > 80) {
            setAlerts((prev) => [
              {
                id: Math.random().toString(36).substring(2, 9),
                timestamp: new Date().toLocaleTimeString(),
                severity: "low",
                message: `Inference pipeline jitter spike: ${frameData.latency_ms?.total?.toFixed(1)}ms`,
                frame_id: frameData.frame_id,
              },
              ...prev.slice(0, 19),
            ]);
          }
        } catch (err) {
          console.error("Failed to parse SSE payload:", err);
        }
      });

      eventSource.onerror = () => {
        setConnectionStatus("DISCONNECTED");
        if (eventSource) {
          eventSource.close();
        }
        // Auto-reconnect after 3 seconds
        reconnectTimeout = setTimeout(connectToStreamA, 3000);
      };
    };

    connectToStreamA();

    return () => {
      if (eventSource) eventSource.close();
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
    };
  }, [apiBaseUrl]);

  // ══════════════════════════════════════════════════════════════════════════
  //  2. Live Canvas Skeleton & HUD Rendering
  // ══════════════════════════════════════════════════════════════════════════
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw Cyberpunk Grid
    ctx.strokeStyle = "rgba(16, 185, 129, 0.15)";
    ctx.lineWidth = 1;
    const step = 40;
    for (let x = 0; x < canvas.width; x += step) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += step) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    if (!feedLive || !currentFrame || currentFrame.num_humans === 0) {
      ctx.fillStyle = "rgba(148, 163, 184, 0.5)";
      ctx.font = "14px monospace";
      ctx.fillText(!feedLive ? feedStatus : "NO PEOPLE DETECTED IN CURRENT FRAME", 20, 30);
      return;
    }

    // Scale from pipeline coords (1280x720) to canvas (640x360)
    const scaleX = canvas.width / 1280;
    const scaleY = canvas.height / 720;

    // COCO 17-Keypoint skeleton edges
    const SKELETON_PAIRS = [
      [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
      [6, 12], [7, 13], [6, 7], [6, 8], [8, 10],
      [7, 9], [9, 11], [2, 3], [1, 2], [1, 3],
      [2, 4], [3, 5], [4, 6], [5, 7]
    ];

    currentFrame.pose_keypoints.forEach((person, personIdx) => {
      // Calculate Bounding Box around keypoints
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      person.forEach(([kx, ky]) => {
        if (kx > 0 && ky > 0) {
          minX = Math.min(minX, kx * scaleX);
          minY = Math.min(minY, ky * scaleY);
          maxX = Math.max(maxX, kx * scaleX);
          maxY = Math.max(maxY, ky * scaleY);
        }
      });

      // Draw Subject Bounding Box
      if (minX !== Infinity) {
        ctx.strokeStyle = "#10b981"; // Emerald
        ctx.lineWidth = 2;
        ctx.strokeRect(minX - 10, minY - 10, (maxX - minX) + 20, (maxY - minY) + 20);

        // Target Tag
        ctx.fillStyle = "#10b981";
        ctx.fillRect(minX - 10, minY - 26, 100, 16);
        ctx.fillStyle = "#000000";
        ctx.font = "bold 10px monospace";
        ctx.fillText(`SUBJ #${personIdx + 1} TRACKED`, minX - 6, minY - 14);
      }

      // Draw Bones
      ctx.strokeStyle = "#06b6d4"; // Cyan
      ctx.lineWidth = 2;
      SKELETON_PAIRS.forEach(([p1, p2]) => {
        const kp1 = person[p1 - 1];
        const kp2 = person[p2 - 1];
        if (kp1 && kp2 && kp1[0] > 0 && kp2[0] > 0) {
          ctx.beginPath();
          ctx.moveTo(kp1[0] * scaleX, kp1[1] * scaleY);
          ctx.lineTo(kp2[0] * scaleX, kp2[1] * scaleY);
          ctx.stroke();
        }
      });

      // Draw Keypoints
      person.forEach(([kx, ky]) => {
        if (kx > 0 && ky > 0) {
          ctx.fillStyle = "#f59e0b"; // Amber joints
          ctx.beginPath();
          ctx.arc(kx * scaleX, ky * scaleY, 4, 0, 2 * Math.PI);
          ctx.fill();
        }
      });
    });
  }, [currentFrame, feedLive, feedStatus]);

  // ══════════════════════════════════════════════════════════════════════════
  //  3. Stream B Forensic Investigation (Authenticated Vault Fetch)
  // ══════════════════════════════════════════════════════════════════════════
  const handleFetchForensicLatents = async () => {
    if (!vaultToken.trim()) {
      setForensicError("Please input an authorized Vault Compliance Token.");
      return;
    }

    setIsFetchingLatents(true);
    setForensicError(null);
    setForensicResult(null);

    try {
      const url = `${apiBaseUrl.replace(/\/$/, "")}/analytics/stream-b`;
      const response = await fetch(url, {
        method: "GET",
        headers: {
          "Accept": "application/json",
          "X-Vault-Token": vaultToken.trim(),
          "X-Incident-ID": incidentId,
          "X-Investigation-Reason": investigationReason,
        },
      });

      if (!response.ok) {
        if (response.status === 401) {
          throw new Error("401 Unauthorized: Missing or malformed Vault Compliance Token.");
        } else if (response.status === 403) {
          throw new Error("403 Forbidden: Token is invalid, expired, or lacks forensic read permissions.");
        } else {
          const errData = await response.json().catch(() => ({}));
          throw new Error(`Error ${response.status}: ${errData.detail || "Server rejection."}`);
        }
      }

      const data: StreamBResponse = await response.json();
      setForensicResult(data);
    } catch (err: any) {
      setForensicError(err.message || "Failed to retrieve encrypted forensic latents.");
    } finally {
      setIsFetchingLatents(false);
    }
  };

  return (
    <div style={{ backgroundColor: "#090d16", minHeight: "100vh", color: "#f8fafc", fontFamily: "ui-monospace, monospace", padding: "24px" }}>
      {/* ── Top Header & Telemetry Bar ───────────────────────────────────── */}
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid #1e293b", paddingBottom: "16px", marginBottom: "24px" }}>
        <div>
          <h1 style={{ margin: 0, fontSize: "22px", color: "#10b981", display: "flex", alignItems: "center", gap: "10px" }}>
            <span style={{ display: "inline-block", width: "12px", height: "12px", borderRadius: "50%", backgroundColor: connectionStatus === "CONNECTED" ? "#10b981" : "#ef4444", boxShadow: connectionStatus === "CONNECTED" ? "0 0 10px #10b981" : "none" }}></span>
            CRYPTAVISION // SECURITY & FORENSIC COMPLIANCE CENTER
          </h1>
          <p style={{ margin: "4px 0 0 0", fontSize: "12px", color: "#64748b" }}>
            Dual-Stream Privacy Video Analytics &bull; Zero-Trust TenSEAL CKKS &bull; HashiCorp Vault RBAC
          </p>
        </div>

        <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
          <input
            type="text"
            value={apiBaseUrl}
            onChange={(e) => setApiBaseUrl(e.target.value)}
            style={{ backgroundColor: "#0f172a", border: "1px solid #334155", color: "#cbd5e1", padding: "6px 10px", borderRadius: "4px", fontSize: "12px" }}
            placeholder="Backend URL"
          />
          <button
            onClick={() => setIsModalOpen(true)}
            style={{ backgroundColor: "#b91c1c", color: "#ffffff", border: "1px solid #ef4444", padding: "8px 16px", borderRadius: "4px", cursor: "pointer", fontSize: "12px", fontWeight: "bold", display: "flex", alignItems: "center", gap: "6px", boxShadow: "0 0 12px rgba(239,68,68,0.3)" }}
          >
            🛡️ FORENSIC INVESTIGATION (STREAM B)
          </button>
        </div>
      </header>

      {/* ── Metric Badges ─────────────────────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: "12px", marginBottom: "24px" }}>
        <MetricCard title="PIPELINE STATUS" value={feedStatus} color={feedLive ? "#10b981" : "#f59e0b"} />
        <MetricCard title="ACTIVE SUBJECTS" value={currentFrame ? currentFrame.num_humans.toString() : "0"} color="#06b6d4" />
        <MetricCard title="FRAME SEQUENCE" value={currentFrame ? `#${currentFrame.frame_id}` : "N/A"} color="#f59e0b" />
        <MetricCard title="INFERENCE LATENCY" value={currentFrame?.latency_ms?.total ? `${currentFrame.latency_ms.total.toFixed(1)} ms` : "0.0 ms"} color="#8b5cf6" />
        <MetricCard title="CLIP EMBEDDING DIM" value={currentFrame?.clip_embedding?.length ? `${currentFrame.clip_embedding.length}-D (FP16)` : "512-D"} color="#ec4899" />
      </div>

      {/* ── Main Operations Grid ──────────────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: "24px" }}>
        {/* Left: Stream A Live HUD Visualization */}
        <div style={{ backgroundColor: "#0f172a", border: "1px solid #1e293b", borderRadius: "6px", padding: "16px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "12px" }}>
            <span style={{ fontSize: "12px", color: "#94a3b8", fontWeight: "bold" }}>STREAM A // REAL-TIME SEMANTIC HUD (YOLO POSE &amp; CLIP)</span>
            <span style={{ fontSize: "12px", color: feedLive ? "#10b981" : "#f59e0b" }}>{feedLive ? `LIVE • ${processedCount} FRAMES RECEIVED` : feedStatus}</span>
          </div>
          <div style={{ position: "relative", width: "100%", height: "360px", backgroundColor: "#020617", borderRadius: "4px", overflow: "hidden", display: "flex", justifyContent: "center", alignItems: "center" }}>
            <canvas ref={canvasRef} width={640} height={360} style={{ width: "100%", height: "100%", objectFit: "contain" }} />
          </div>

          {/* CLIP Embedding Vector Preview */}
          <div style={{ marginTop: "16px" }}>
            <span style={{ fontSize: "11px", color: "#64748b" }}>CLIP EMBEDDING VECTOR HEAD (FIRST 16 SLOTS):</span>
            <div style={{ display: "flex", gap: "4px", marginTop: "6px", overflowX: "auto", paddingBottom: "4px" }}>
              {currentFrame?.clip_embedding?.slice(0, 16).map((val, idx) => (
                <div key={idx} style={{ backgroundColor: "#1e293b", padding: "4px 6px", borderRadius: "3px", fontSize: "10px", color: "#38bdf8" }}>
                  {val.toFixed(3)}
                </div>
              )) || <span style={{ fontSize: "11px", color: "#475569" }}>No embeddings extracted yet.</span>}
            </div>
          </div>
        </div>

        {/* Right: Anomaly Alerts & Compliance Audit Log */}
        <div style={{ backgroundColor: "#0f172a", border: "1px solid #1e293b", borderRadius: "6px", padding: "16px", display: "flex", flexDirection: "column" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "12px" }}>
            <span style={{ fontSize: "12px", color: "#94a3b8", fontWeight: "bold" }}>LIVE SECURITY ALERTS</span>
            <span style={{ fontSize: "12px", color: "#f59e0b" }}>{alerts.length} NOTIFICATIONS</span>
          </div>
          <div style={{ flex: 1, overflowY: "auto", maxHeight: "380px", display: "flex", flexDirection: "column", gap: "8px" }}>
            {alerts.length === 0 ? (
              <div style={{ color: "#475569", fontSize: "12px", textAlign: "center", marginTop: "40px" }}>
                NO ANOMALIES DETECTED &bull; SYSTEM NORMAL
              </div>
            ) : (
              alerts.map((alert) => (
                <div key={alert.id} style={{ backgroundColor: "#020617", borderLeft: `3px solid ${alert.severity === "high" ? "#ef4444" : alert.severity === "medium" ? "#f59e0b" : "#3b82f6"}`, padding: "8px 12px", borderRadius: "3px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "10px", color: "#64748b" }}>
                    <span>{alert.timestamp} &bull; FRAME #{alert.frame_id}</span>
                    <span style={{ textTransform: "uppercase", color: alert.severity === "high" ? "#ef4444" : "#f59e0b" }}>{alert.severity} PRIORITY</span>
                  </div>
                  <div style={{ fontSize: "11px", marginTop: "4px", color: "#e2e8f0" }}>{alert.message}</div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════════════════ */}
      {/*  4. Forensic Investigation Modal (Stream B Protected Access)         */}
      {/* ══════════════════════════════════════════════════════════════════════ */}
      {isModalOpen && (
        <div style={{ position: "fixed", top: 0, left: 0, width: "100%", height: "100%", backgroundColor: "rgba(0, 0, 0, 0.85)", display: "flex", justifyContent: "center", alignItems: "center", zIndex: 1000, backdropFilter: "blur(4px)" }}>
          <div style={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: "8px", width: "680px", maxWidth: "90%", padding: "24px", boxShadow: "0 0 30px rgba(0, 0, 0, 0.9)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid #1e293b", paddingBottom: "12px", marginBottom: "16px" }}>
              <h3 style={{ margin: 0, fontSize: "16px", color: "#ef4444", display: "flex", alignItems: "center", gap: "8px" }}>
                🔒 AUTHORIZED FORENSIC INVESTIGATION (STREAM B)
              </h3>
              <button onClick={() => setIsModalOpen(false)} style={{ background: "none", border: "none", color: "#94a3b8", cursor: "pointer", fontSize: "18px" }}>&times;</button>
            </div>

            <p style={{ fontSize: "12px", color: "#94a3b8", lineHeight: "1.5" }}>
              Stream B contains obfuscated video. Face representations are locked under Homomorphic Encryption (TenSEAL CKKS, scale=2^40). Provide your 1-hour time-limited Vault Compliance Token to drain and extract the encrypted latents.
            </p>

            <div style={{ display: "flex", flexDirection: "column", gap: "12px", marginTop: "16px" }}>
              <div>
                <label style={{ fontSize: "11px", color: "#64748b" }}>VAULT COMPLIANCE TOKEN (X-Vault-Token):</label>
                <input
                  type="password"
                  value={vaultToken}
                  onChange={(e) => setVaultToken(e.target.value)}
                  placeholder="e.g. s.compliance_officer_token_xyz"
                  style={{ width: "100%", marginTop: "4px", backgroundColor: "#020617", border: "1px solid #334155", color: "#38bdf8", padding: "8px 12px", borderRadius: "4px", fontSize: "12px", boxSizing: "border-box" }}
                />
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
                <div>
                  <label style={{ fontSize: "11px", color: "#64748b" }}>INCIDENT CASE NUMBER:</label>
                  <input
                    type="text"
                    value={incidentId}
                    onChange={(e) => setIncidentId(e.target.value)}
                    style={{ width: "100%", marginTop: "4px", backgroundColor: "#020617", border: "1px solid #334155", color: "#cbd5e1", padding: "8px 12px", borderRadius: "4px", fontSize: "12px", boxSizing: "border-box" }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: "11px", color: "#64748b" }}>JUSTIFICATION:</label>
                  <input
                    type="text"
                    value={investigationReason}
                    onChange={(e) => setInvestigationReason(e.target.value)}
                    style={{ width: "100%", marginTop: "4px", backgroundColor: "#020617", border: "1px solid #334155", color: "#cbd5e1", padding: "8px 12px", borderRadius: "4px", fontSize: "12px", boxSizing: "border-box" }}
                  />
                </div>
              </div>

              {forensicError && (
                <div style={{ backgroundColor: "rgba(239, 68, 68, 0.15)", border: "1px solid #ef4444", color: "#fca5a5", padding: "10px", borderRadius: "4px", fontSize: "12px" }}>
                  ⚠️ {forensicError}
                </div>
              )}

              <button
                onClick={handleFetchForensicLatents}
                disabled={isFetchingLatents}
                style={{ backgroundColor: isFetchingLatents ? "#64748b" : "#10b981", color: "#000000", fontWeight: "bold", border: "none", padding: "10px", borderRadius: "4px", cursor: isFetchingLatents ? "not-allowed" : "pointer", fontSize: "12px", marginTop: "6px" }}
              >
                {isFetchingLatents ? "VERIFYING TOKEN & DRAINING CKKS LATENTS..." : "AUTHENTICATE & EXTRACT FORENSIC LATENTS"}
              </button>
            </div>

            {/* Success State Display */}
            {forensicResult && (
              <div style={{ marginTop: "16px", borderTop: "1px solid #1e293b", paddingTop: "16px" }}>
                <div style={{ backgroundColor: "rgba(16, 185, 129, 0.1)", border: "1px solid #10b981", borderRadius: "4px", padding: "12px" }}>
                  <div style={{ color: "#10b981", fontSize: "12px", fontWeight: "bold" }}>
                    ✓ AUTHENTICATION GRANTED &bull; {forensicResult.count} FACE LATENTS EXTRACTED
                  </div>
                  <div style={{ color: "#64748b", fontSize: "10px", marginTop: "2px" }}>
                    TIMESTAMP: {forensicResult.timestamp} &bull; VAULT VERIFIED: TRUE
                  </div>

                  {forensicResult.latents.length === 0 ? (
                    <p style={{ fontSize: "11px", color: "#94a3b8", marginTop: "8px" }}>
                      Buffer is empty: No new face latents detected since last drain.
                    </p>
                  ) : (
                    <div style={{ marginTop: "10px", maxHeight: "180px", overflowY: "auto", display: "flex", flexDirection: "column", gap: "6px" }}>
                      {forensicResult.latents.map((item, idx) => (
                        <div key={idx} style={{ backgroundColor: "#020617", padding: "8px", borderRadius: "3px", fontSize: "10px" }}>
                          <div style={{ display: "flex", justifyContent: "space-between", color: "#38bdf8" }}>
                            <span>FACE #{item.face_idx + 1} &bull; FRAME #{item.frame_id}</span>
                            <span>{item.byte_size.toLocaleString()} BYTES</span>
                          </div>
                          <div style={{ color: "#64748b", marginTop: "2px" }}>
                            BOX: [{item.box.join(", ")}] &bull; SHA-256: {item.sha256.substring(0, 16)}...
                          </div>
                          <div style={{ color: "#475569", marginTop: "2px", wordBreak: "break-all", fontFamily: "monospace" }}>
                            SEALED CKKS: {item.ckks_bytes_b64.substring(0, 48)}...
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Helper Metric Card Component ─────────────────────────────────────────────
function MetricCard({ title, value, color }: { title: string; value: string; color: string }) {
  return (
    <div style={{ backgroundColor: "#0f172a", border: "1px solid #1e293b", padding: "12px", borderRadius: "4px" }}>
      <div style={{ fontSize: "10px", color: "#64748b", letterSpacing: "0.5px" }}>{title}</div>
      <div style={{ fontSize: "16px", fontWeight: "bold", color, marginTop: "4px" }}>{value}</div>
    </div>
  );
}
