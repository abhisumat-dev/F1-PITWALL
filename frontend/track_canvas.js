/**
 * F1 Pit-Wall — Real 2D Circuit Map Canvas Renderer (60 FPS lerp)
 *
 * Features:
 * - Renders official MultiViewer / OpenF1 2D circuit geometry from either the
 *   live packet (`circuit.track_outline`, `sector_splits`, `corners`,
 *   `start_finish`) or the bundled 24-circuit catalog (`window.F1_CIRCUITS_2D`).
 * - Colors Sector 1 (#EF4444), Sector 2 (#38BDF8), and Sector 3 (#FACC15),
 *   marks S/F chequered gate and official corner numbers, with NO duplicated
 *   circuit name text inside the circuit graphic.
 * - Interpolates all 22 cars at 60 FPS using exponential lerp, dims cars in
 *   pit lane, and always draws P1 last on top (reverse position z-order).
 */
(function () {
  "use strict";

  const SECTOR_COLORS = ["#EF4444", "#38BDF8", "#FACC15"];

  class TrackCanvasRenderer {
    constructor(canvas, opts) {
      this.canvas = canvas;
      this.ctx = canvas ? canvas.getContext("2d") : null;
      this.compact = Boolean(opts && opts.compact);
      this.circuit = null;
      this.driversTarget = new Map();
      this.driversSmooth = new Map();
      this.selectedDrivers = new Set();
      this.gpsAvailable = true;
      this.rafId = null;
      this._bounds = null;
      this._ro = null;

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
      this._draw();
    }

    setCircuit(circuitData) {
      if (!circuitData) return;
      const pts = circuitData.track_outline || circuitData.points;
      if (!Array.isArray(pts) || pts.length < 10) return;
      this.circuit = {
        slug: circuitData.slug || null,
        points: pts,
        sector_splits: circuitData.sector_splits || null,
        corners: circuitData.corners || [],
        start_finish: circuitData.start_finish || pts[0],
      };
      this._computeBounds();
      this._draw();
    }

    setSelectedDrivers(nums) {
      this.selectedDrivers = new Set((nums || []).map(Number));
    }

    updateDrivers(drivers, gpsAvailable) {
      this.gpsAvailable = Boolean(gpsAvailable);
      const seen = new Set();
      for (const d of drivers || []) {
        const dn = Number(d.driver_number);
        seen.add(dn);
        const hasXY = typeof d.x === "number" && typeof d.y === "number" && Number.isFinite(d.x) && Number.isFinite(d.y);
        const target = {
          driver_number: dn,
          code: String(d.code || dn),
          position: d.position != null ? Number(d.position) : 99,
          team_colour: d.team_colour ? "#" + String(d.team_colour).replace("#", "") : "#94A3B8",
          in_pit: Boolean(d.in_pit),
          retired: Boolean(d.retired),
          x: hasXY ? d.x : null,
          y: hasXY ? d.y : null,
        };
        this.driversTarget.set(dn, target);
        if (hasXY && !this.driversSmooth.has(dn)) {
          this.driversSmooth.set(dn, { x: d.x, y: d.y });
        }
      }
      for (const dn of Array.from(this.driversTarget.keys())) {
        if (!seen.has(dn)) {
          this.driversTarget.delete(dn);
          this.driversSmooth.delete(dn);
        }
      }
      if (!this.rafId && this.gpsAvailable && this.driversTarget.size > 0) {
        this.start();
      } else {
        this._draw();
      }
    }

    start() {
      if (this.rafId) return;
      const tick = () => {
        this._stepLerp();
        this._draw();
        this.rafId = requestAnimationFrame(tick);
      };
      this.rafId = requestAnimationFrame(tick);
    }

    stop() {
      if (this.rafId) {
        cancelAnimationFrame(this.rafId);
        this.rafId = null;
      }
    }

    _stepLerp() {
      const alpha = 0.22;
      for (const [dn, target] of this.driversTarget.entries()) {
        if (target.x === null || target.y === null) continue;
        const cur = this.driversSmooth.get(dn);
        if (!cur) {
          this.driversSmooth.set(dn, { x: target.x, y: target.y });
          continue;
        }
        const dx = target.x - cur.x;
        const dy = target.y - cur.y;
        // Snap if car jumped (seek lap or lap wrap glitch)
        if (Math.hypot(dx, dy) > 2500) {
          cur.x = target.x;
          cur.y = target.y;
        } else {
          cur.x += dx * alpha;
          cur.y += dy * alpha;
        }
      }
    }

    _computeBounds() {
      if (!this.circuit || !this.circuit.points.length) {
        this._bounds = null;
        return;
      }
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const p of this.circuit.points) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
      }
      this._bounds = {
        minX,
        maxX,
        minY,
        maxY,
        spanX: Math.max(1, maxX - minX),
        spanY: Math.max(1, maxY - minY),
      };
    }

    _project(pt, w, h, pad) {
      const b = this._bounds;
      const scale = Math.min((w - pad * 2) / b.spanX, (h - pad * 2) / b.spanY);
      const ox = pad + ((w - pad * 2) - b.spanX * scale) / 2;
      const oy = pad + ((h - pad * 2) - b.spanY * scale) / 2;
      return {
        x: ox + (pt.x - b.minX) * scale,
        y: oy + (b.maxY - pt.y) * scale, // Invert Y for TV broadcast orientation
      };
    }

    _draw() {
      if (!this.ctx || !this.canvas) return;
      const ctx = this.ctx;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = this.canvas.width / dpr;
      const h = this.canvas.height / dpr;
      if (w <= 0 || h <= 0) return;

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, w, h);

      if (!this.circuit || !this._bounds) {
        ctx.fillStyle = "#64748B";
        ctx.font = "500 12px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText("Waiting for circuit geometry...", w / 2, h / 2);
        ctx.restore();
        return;
      }

      const pad = this.compact ? 18 : 30;
      const pts = this.circuit.points;
      const proj = pts.map((p) => this._project(p, w, h, pad));
      const splits = Array.isArray(this.circuit.sector_splits) && this.circuit.sector_splits.length >= 4
        ? this.circuit.sector_splits
        : [0, Math.floor(pts.length / 3), Math.floor((pts.length * 2) / 3), pts.length - 1];

      // 1. Dark asphalt roadbed halo
      ctx.beginPath();
      proj.forEach((p, idx) => (idx === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
      ctx.closePath();
      ctx.strokeStyle = "#1B2333";
      ctx.lineWidth = this.compact ? 8 : 11;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.stroke();

      // 2. Three Sector strokes (S1 Red, S2 Cyan, S3 Yellow)
      const ranges = [
        [splits[0], splits[1]],
        [splits[1], splits[2]],
        [splits[2], proj.length - 1],
      ];
      for (let s = 0; s < 3; s++) {
        const [a, b] = ranges[s];
        ctx.beginPath();
        for (let i = a; i <= Math.min(b, proj.length - 1); i++) {
          const p = proj[i];
          if (i === a) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }
        if (s === 2 && proj.length > 0) {
          ctx.lineTo(proj[0].x, proj[0].y);
        }
        ctx.strokeStyle = SECTOR_COLORS[s];
        ctx.lineWidth = this.compact ? 3.5 : 4.2;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.stroke();
      }

      // 3. Sector boundary markers + S/F chequered tick
      const sfProj = this._project(this.circuit.start_finish || pts[0], w, h, pad);
      const nextPt = proj[Math.min(2, proj.length - 1)];
      const ang = Math.atan2(nextPt.y - sfProj.y, nextPt.x - sfProj.x) + Math.PI / 2;
      const tickLen = this.compact ? 6 : 8;
      ctx.beginPath();
      ctx.moveTo(sfProj.x - Math.cos(ang) * tickLen, sfProj.y - Math.sin(ang) * tickLen);
      ctx.lineTo(sfProj.x + Math.cos(ang) * tickLen, sfProj.y + Math.sin(ang) * tickLen);
      ctx.strokeStyle = "#FFFFFF";
      ctx.lineWidth = 2.8;
      ctx.stroke();

      // S/F label
      ctx.fillStyle = "#0F172A";
      ctx.strokeStyle = "rgba(255,255,255,0.65)";
      ctx.lineWidth = 1;
      const sfLx = sfProj.x + 12;
      const sfLy = sfProj.y - 10;
      ctx.beginPath();
      ctx.roundRect(sfLx - 14, sfLy - 8, 28, 14, 3);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#FFFFFF";
      ctx.font = "700 8px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("S/F", sfLx, sfLy - 1);

      // Sector boundary dots
      for (const idx of [splits[1], splits[2]]) {
        const sp = proj[Math.min(idx, proj.length - 1)];
        if (!sp) continue;
        ctx.beginPath();
        ctx.arc(sp.x, sp.y, 3.6, 0, Math.PI * 2);
        ctx.fillStyle = "#0B0E14";
        ctx.fill();
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 1.4;
        ctx.stroke();
      }

      // On-map Sector tags (S1, S2, S3) positioned gently offset from midpoints
      if (!this.compact) {
        const cx = w / 2;
        const cy = h / 2;
        const labels = ["S1", "S2", "S3"];
        for (let s = 0; s < 3; s++) {
          const midIdx = Math.floor((ranges[s][0] + ranges[s][1]) / 2);
          const mp = proj[Math.min(midIdx, proj.length - 1)];
          if (!mp) continue;
          const dx = mp.x - cx;
          const dy = mp.y - cy;
          const dist = Math.max(1, Math.hypot(dx, dy));
          const lx = mp.x + (dx / dist) * 18;
          const ly = mp.y + (dy / dist) * 16;

          ctx.fillStyle = "rgba(11, 14, 20, 0.88)";
          ctx.strokeStyle = SECTOR_COLORS[s];
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.roundRect(lx - 13, ly - 8, 26, 15, 3);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = SECTOR_COLORS[s];
          ctx.font = "700 8.5px 'JetBrains Mono', monospace";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(labels[s], lx, ly - 0.5);
        }

        // Sparse key corner labels (avoid cluttering tight complexes)
        const corners = this.circuit.corners || [];
        const step = corners.length > 14 ? 2 : 1;
        ctx.fillStyle = "rgba(148, 163, 184, 0.72)";
        ctx.font = "500 8px 'JetBrains Mono', monospace";
        for (let i = 0; i < corners.length; i += step) {
          const cn = corners[i];
          const cp = this._project(cn, w, h, pad);
          const dx = cp.x - cx;
          const dy = cp.y - cy;
          const dist = Math.max(1, Math.hypot(dx, dy));
          ctx.fillText(`T${cn.number}`, cp.x + (dx / dist) * 11, cp.y + (dy / dist) * 11);
        }
      }

      // 4. Live Car Markers (drawn in reverse position order so P1 sits on top!)
      if (this.gpsAvailable && this.driversTarget.size > 0) {
        const ordered = Array.from(this.driversTarget.values())
          .filter((d) => !d.retired && d.x !== null && d.y !== null)
          .sort((a, b) => b.position - a.position);

        for (const d of ordered) {
          const sm = this.driversSmooth.get(d.driver_number) || d;
          const p = this._project(sm, w, h, pad);
          const isLeader = d.position === 1;
          const isSel = this.selectedDrivers.has(d.driver_number);
          const isTop3 = d.position <= 3;

          ctx.save();
          if (d.in_pit) ctx.globalAlpha = 0.42;

          const r = isLeader ? 6.5 : isSel || isTop3 ? 5.5 : 4.2;
          ctx.beginPath();
          ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
          ctx.fillStyle = d.team_colour;
          ctx.fill();
          ctx.strokeStyle = isLeader || isSel ? "#FFFFFF" : "#07090D";
          ctx.lineWidth = isLeader ? 2 : 1.3;
          ctx.stroke();

          if (!this.compact && (isLeader || isSel || isTop3)) {
            const tag = isLeader ? `P1 ${d.code}` : d.code;
            ctx.font = "700 8.5px 'JetBrains Mono', monospace";
            const tw = ctx.measureText(tag).width + 8;
            ctx.fillStyle = isLeader ? d.team_colour : "rgba(7, 9, 13, 0.88)";
            ctx.beginPath();
            ctx.roundRect(p.x - tw / 2, p.y - r - 17, tw, 13, 3);
            ctx.fill();
            ctx.fillStyle = "#FFFFFF";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(tag, p.x, p.y - r - 10.5);
          }
          ctx.restore();
        }
      }

      ctx.restore();
    }
  }

  /**
   * Generate a lightweight inline SVG string for the 24-circuit catalog thumbnails.
   */
  function buildCircuitThumbnailSvg(circuit) {
    const pts = circuit.points || circuit.track_outline || [];
    if (pts.length < 10) return "";
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of pts) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    const w = 108, h = 56, pad = 6;
    const sc = Math.min((w - pad * 2) / Math.max(1, maxX - minX), (h - pad * 2) / Math.max(1, maxY - minY));
    const ox = pad + ((w - pad * 2) - (maxX - minX) * sc) / 2;
    const oy = pad + ((h - pad * 2) - (maxY - minY) * sc) / 2;
    const mpts = pts.map((p) => [
      (ox + (p.x - minX) * sc).toFixed(1),
      (oy + (maxY - p.y) * sc).toFixed(1),
    ]);
    const splits = circuit.sector_splits && circuit.sector_splits.length >= 4
      ? circuit.sector_splits
      : [0, Math.floor(pts.length / 3), Math.floor((pts.length * 2) / 3), pts.length - 1];
    const mk = (a, b, close) => {
      const seg = [];
      for (let i = a; i <= Math.min(b, mpts.length - 1); i += 2) {
        seg.push(`${seg.length === 0 ? "M" : "L"}${mpts[i][0]} ${mpts[i][1]}`);
      }
      if (close && mpts.length) seg.push(`L${mpts[0][0]} ${mpts[0][1]}`);
      return seg.join(" ");
    };
    return (
      `<svg viewBox="0 0 ${w} ${h}" fill="none" aria-hidden="true">` +
      `<path d="${mk(splits[0], splits[1], false)}" stroke="#EF4444" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>` +
      `<path d="${mk(splits[1], splits[2], false)}" stroke="#38BDF8" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>` +
      `<path d="${mk(splits[2], mpts.length - 1, true)}" stroke="#FACC15" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>` +
      `</svg>`
    );
  }

  window.TrackCanvasRenderer = TrackCanvasRenderer;
  window.buildCircuitThumbnailSvg = buildCircuitThumbnailSvg;
})();
