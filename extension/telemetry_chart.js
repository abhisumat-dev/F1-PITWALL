/**
 * F1 Pit-Wall — Smooth MoTeC Telemetry Compare Canvas (4Hz Catmull-Rom Splines)
 *
 * Renders 3 explicitly labeled lanes with zero text/trace overlap:
 *   1. SPEED (KPH)    [0 .. 360]
 *   2. THROTTLE (%)   [0 .. 100]
 *   3. BRAKE (%)      [0 .. 100]
 * Uses Catmull-Rom -> cubic Bezier smoothing so traces never look jagged/edgy.
 */
(function () {
  "use strict";

  const MAX_SAMPLES = 90;

  class TelemetryChartRenderer {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas ? canvas.getContext("2d") : null;
      this.historyA = [];
      this.historyB = [];
      this.metaA = { code: "VER", color: "#38BDF8" };
      this.metaB = { code: "NOR", color: "#FF8000" };
      this.available = true;

      if (this.canvas && typeof ResizeObserver !== "undefined") {
        this._ro = new ResizeObserver(() => this._resize());
        this._ro.observe(this.canvas);
      }
      this._resize();
    }

    _resize() {
      if (!this.canvas || !this.ctx) return;
      const rect = this.canvas.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.round(rect.width * dpr);
      const h = Math.round(rect.height * dpr);
      if (this.canvas.width !== w || this.canvas.height !== h) {
        this.canvas.width = w;
        this.canvas.height = h;
      }
      this.draw();
    }

    reset(metaA, metaB) {
      this.historyA = [];
      this.historyB = [];
      if (metaA) this.metaA = metaA;
      if (metaB) this.metaB = metaB;
      this.draw();
    }

    setDriversMeta(metaA, metaB) {
      if (metaA && metaA.code !== this.metaA.code) {
        this.historyA = [];
      }
      if (metaB && metaB.code !== this.metaB.code) {
        this.historyB = [];
      }
      if (metaA) this.metaA = metaA;
      if (metaB) this.metaB = metaB;
    }

    pushSample(telA, telB, available) {
      this.available = Boolean(available);
      if (!this.available) {
        this.draw();
        return;
      }
      if (telA) {
        this.historyA.push({
          speed: Number.isFinite(telA.speed_kph) ? telA.speed_kph : null,
          throttle: Number.isFinite(telA.throttle_pct) ? telA.throttle_pct : null,
          brake: Number.isFinite(telA.brake_pct) ? telA.brake_pct : null,
        });
        if (this.historyA.length > MAX_SAMPLES) this.historyA.shift();
      }
      if (telB) {
        this.historyB.push({
          speed: Number.isFinite(telB.speed_kph) ? telB.speed_kph : null,
          throttle: Number.isFinite(telB.throttle_pct) ? telB.throttle_pct : null,
          brake: Number.isFinite(telB.brake_pct) ? telB.brake_pct : null,
        });
        if (this.historyB.length > MAX_SAMPLES) this.historyB.shift();
      }
      this.draw();
    }

    seedDemoHistory(samplesA, samplesB) {
      this.available = true;
      this.historyA = samplesA.slice(-MAX_SAMPLES);
      this.historyB = samplesB.slice(-MAX_SAMPLES);
      this.draw();
    }

    _drawSmoothSpline(ctx, pts, strokeStyle, lineWidth, fillBottomY, minY, maxY) {
      if (pts.length < 2) return;
      const clampY = (v) => Math.max(minY, Math.min(maxY, v));
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 0; i < pts.length - 1; i++) {
        const p0 = pts[Math.max(0, i - 1)];
        const p1 = pts[i];
        const p2 = pts[i + 1];
        const p3 = pts[Math.min(pts.length - 1, i + 2)];

        const cp1x = p1.x + (p2.x - p0.x) / 6;
        const cp1y = clampY(p1.y + (p2.y - p0.y) / 6);
        const cp2x = p2.x - (p3.x - p1.x) / 6;
        const cp2y = clampY(p2.y - (p3.y - p1.y) / 6);
        ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
      }
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = lineWidth;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.stroke();

      if (fillBottomY != null) {
        ctx.lineTo(pts[pts.length - 1].x, fillBottomY);
        ctx.lineTo(pts[0].x, fillBottomY);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, fillBottomY - 56, 0, fillBottomY);
        grad.addColorStop(0, strokeStyle + "26");
        grad.addColorStop(1, strokeStyle + "00");
        ctx.fillStyle = grad;
        ctx.fill();
      }
    }

    draw() {
      if (!this.ctx || !this.canvas) return;
      const ctx = this.ctx;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = this.canvas.width / dpr;
      const h = this.canvas.height / dpr;
      if (w <= 0 || h <= 0) return;

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, w, h);

      const lanes = [
        { key: "speed", label: "SPEED (KPH)", min: 40, max: 360, ticks: ["350", "200", "50"], unit: "kph" },
        { key: "throttle", label: "THROTTLE (%)", min: 0, max: 100, ticks: ["100%", "50%", "0%"], unit: "%" },
        { key: "brake", label: "BRAKE (%)", min: 0, max: 100, ticks: ["100%", "50%", "0%"], unit: "%" },
      ];

      const laneH = h / 3;
      const padL = 40;
      const padR = 12;
      const plotW = Math.max(40, w - padL - padR);

      lanes.forEach((lane, idx) => {
        const yTop = idx * laneH;
        const headerH = 18;
        const plotTop = yTop + headerH + 4;
        const plotBottom = yTop + laneH - 7;
        const plotH = Math.max(24, plotBottom - plotTop);

        // Subtle lane divider
        if (idx > 0) {
          ctx.beginPath();
          ctx.moveTo(0, yTop);
          ctx.lineTo(w, yTop);
          ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
          ctx.lineWidth = 1;
          ctx.stroke();
        }

        // Dedicated top header row (never overlaps traces)
        ctx.fillStyle = "#E2E8F0";
        ctx.font = "700 9.5px 'JetBrains Mono', monospace";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(lane.label, 12, yTop + 10);

        const lastA = this.historyA.length ? this.historyA[this.historyA.length - 1][lane.key] : null;
        const lastB = this.historyB.length ? this.historyB[this.historyB.length - 1][lane.key] : null;
        const valText = this.available
          ? `${this.metaA.code} ${lastA != null ? Math.round(lastA) : "--"} · ${this.metaB.code} ${lastB != null ? Math.round(lastB) : "--"} ${lane.unit}`
          : "LOCKED";
        ctx.fillStyle = "#94A3B8";
        ctx.font = "500 9.5px 'JetBrains Mono', monospace";
        ctx.textAlign = "right";
        ctx.fillText(valText, w - 12, yTop + 10);

        // Horizontal gridlines & Y-axis ticks
        const yTicks = [plotTop, plotTop + plotH / 2, plotBottom];
        ctx.font = "400 8px 'JetBrains Mono', monospace";
        ctx.textAlign = "right";
        yTicks.forEach((ty, tIdx) => {
          ctx.beginPath();
          ctx.setLineDash(tIdx === 2 ? [] : [3, 3]);
          ctx.moveTo(padL, ty);
          ctx.lineTo(w - padR, ty);
          ctx.strokeStyle = "rgba(148, 163, 184, 0.14)";
          ctx.lineWidth = 1;
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = "#64748B";
          ctx.fillText(lane.ticks[tIdx], padL - 6, ty + 0.5);
        });

        if (!this.available) return;

        const mapPts = (arr) => {
          const out = [];
          const denom = Math.max(1, MAX_SAMPLES - 1);
          const offset = MAX_SAMPLES - arr.length;
          for (let i = 0; i < arr.length; i++) {
            const v = arr[i][lane.key];
            if (v == null || !Number.isFinite(v)) continue;
            const norm = Math.max(0, Math.min(1, (v - lane.min) / (lane.max - lane.min)));
            out.push({
              x: padL + ((offset + i) / denom) * plotW,
              y: plotBottom - norm * plotH,
            });
          }
          return out;
        };

        const ptsB = mapPts(this.historyB);
        const ptsA = mapPts(this.historyA);
        this._drawSmoothSpline(ctx, ptsB, this.metaB.color || "#FF8000", 1.8, null, plotTop, plotBottom);
        this._drawSmoothSpline(ctx, ptsA, this.metaA.color || "#38BDF8", 2.2, plotBottom, plotTop, plotBottom);
      });

      ctx.restore();
    }
  }

  window.TelemetryChartRenderer = TelemetryChartRenderer;
})();
