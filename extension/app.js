/**
 * F1 Pit-Wall — Multi-Page Application Controller
 *
 * Detects active page via `document.body.dataset.page`:
 *   - "standings"   (index.html / sidepanel.html)
 *   - "live"        (live.html — Live SignalR + Past Race Replays & Results)
 *   - "predictions" (predictions.html — Dedicated Post-Quali Top-3 Forecast)
 */
(function () {
  "use strict";

  const STORAGE_KEY_HOST = "f1_pitwall_api_host";
  const STORAGE_KEY_DELAY = "f1_pitwall_sync_delay_s";

  const COMPOUND_META = {
    SOFT: { letter: "S", color: "#EF4444", text: "#FFFFFF" },
    MEDIUM: { letter: "M", color: "#FACC15", text: "#07090D" },
    HARD: { letter: "H", color: "#E2E8F0", text: "#07090D" },
    INTERMEDIATE: { letter: "I", color: "#10B981", text: "#07090D" },
    WET: { letter: "W", color: "#38BDF8", text: "#07090D" },
    UNKNOWN: { letter: "?", color: "#64748B", text: "#FFFFFF" },
  };

  const TEAM_COLOURS = {
    mercedes: "#27F4D2",
    ferrari: "#E80020",
    "red bull": "#3671C6",
    mclaren: "#FF8000",
    "aston martin": "#229971",
    alpine: "#00A1E8",
    williams: "#1868DB",
    haas: "#9C9FA2",
    "rb f1": "#6C98FF",
    "racing bulls": "#6C98FF",
    audi: "#F50537",
    sauber: "#52E252",
    cadillac: "#94A3B8",
  };

  const $ = (id) => document.getElementById(id);

  function resolveApiBase() {
    const saved = (localStorage.getItem(STORAGE_KEY_HOST) || "").trim().replace(/\/+$/, "");
    if (saved) return saved;
    if (window.location.protocol === "chrome-extension:" || window.location.protocol === "file:") {
      return "http://localhost:8000";
    }
    return window.location.origin;
  }

  function toWsUrl(httpBase) {
    return httpBase.replace(/^https:/i, "wss:").replace(/^http:/i, "ws:") + "/ws/live";
  }

  function esc(str) {
    return String(str == null ? "" : str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function hexCol(c, teamName) {
    if (c) {
      const s = String(c).replace("#", "").trim();
      if (/^[0-9a-fA-F]{6}$/.test(s)) return `#${s}`;
    }
    const tn = String(teamName || "").toLowerCase();
    for (const [k, v] of Object.entries(TEAM_COLOURS)) {
      if (tn.includes(k)) return v;
    }
    return "#64748B";
  }

  function fmtLap(sec) {
    if (typeof sec !== "number" || !Number.isFinite(sec) || sec <= 0) return "--";
    const mins = Math.floor(sec / 60);
    const rem = sec - mins * 60;
    return mins > 0 ? `${mins}:${rem.toFixed(3).padStart(6, "0")}` : `${rem.toFixed(3)}`;
  }

  function fmtGap(d) {
    if (d.retired) return { text: "DNF", cls: "is-out" };
    if (d.in_pit) return { text: "PIT", cls: "is-pit" };
    if (d.position === 1) return { text: "LEADER", cls: "" };
    if (d.laps_down && d.laps_down > 0) return { text: `+${d.laps_down}L`, cls: "" };
    if (typeof d.gap_to_leader_s === "number" && Number.isFinite(d.gap_to_leader_s)) {
      return { text: `+${d.gap_to_leader_s.toFixed(3)}`, cls: "" };
    }
    return { text: "--", cls: "" };
  }

  function fmtUtcClock(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "";
    return d.toISOString().slice(11, 19);
  }

  function buildStintTimelineSvg(driver, lapCurrent, lapTotal) {
    const total = Math.max(10, Number(lapTotal) || 50);
    const curLap = Math.max(1, Math.min(total, Number(lapCurrent) || 1));
    const W = 250;
    const H = 24;
    const pxPerLap = (W - 16) / total;

    let grid = "";
    for (let l = 10; l <= total; l += 10) {
      const gx = (8 + l * pxPerLap).toFixed(1);
      grid += `<line x1="${gx}" y1="1" x2="${gx}" y2="${H - 1}" stroke="rgba(148,163,184,0.14)" stroke-width="0.9"/>`;
    }

    const stints = Array.isArray(driver.stints) && driver.stints.length
      ? driver.stints
      : [{
          stint_number: 1,
          compound: (driver.tyre && driver.tyre.compound) || "UNKNOWN",
          lap_start: 1,
          lap_end: curLap,
          tyre_age_at_start: 0,
        }];

    let shapes = "";
    const op = driver.retired ? 0.45 : 0.94;
    for (let i = 0; i < stints.length; i++) {
      const st = stints[i];
      const nextSt = stints[i + 1];
      const meta = COMPOUND_META[String(st.compound || "UNKNOWN").toUpperCase()] || COMPOUND_META.UNKNOWN;
      const sLap = Math.max(1, Number(st.lap_start) || 1);
      let eLap = st.lap_end != null
        ? Number(st.lap_end)
        : nextSt && nextSt.lap_start
        ? Math.max(sLap, Number(nextSt.lap_start) - 1)
        : curLap;
      eLap = Math.max(sLap, Math.min(curLap, total, eLap));

      const x1 = 8 + (sLap - 1) * pxPerLap;
      const x2 = Math.max(x1 + 15, 8 + eLap * pxPerLap);
      const isLast = i === stints.length - 1;
      const rawAge = isLast && driver.tyre && driver.tyre.age_laps != null
        ? Number(driver.tyre.age_laps)
        : eLap - sLap + 1 + (Number(st.tyre_age_at_start) || 0);
      const age = Math.max(1, rawAge);

      shapes += `<line x1="${x1.toFixed(1)}" y1="12" x2="${x2.toFixed(1)}" y2="12" stroke="${meta.color}" stroke-width="3.5" stroke-linecap="round" opacity="${op}"/>`;
      shapes += `<circle cx="${x1.toFixed(1)}" cy="12" r="6.5" fill="#07090D" stroke="${meta.color}" stroke-width="1.6"/>`;
      shapes += `<text x="${x1.toFixed(1)}" y="14.6" fill="${meta.color}" font-family="JetBrains Mono, monospace" font-size="7.5" font-weight="700" text-anchor="middle">${meta.letter}</text>`;

      if (isLast) {
        const cx = Math.min(W - 9, x2);
        shapes += `<circle cx="${cx.toFixed(1)}" cy="12" r="7.5" fill="${meta.color}"/>`;
        shapes += `<text x="${cx.toFixed(1)}" y="14.7" fill="${meta.text}" font-family="JetBrains Mono, monospace" font-size="7.5" font-weight="700" text-anchor="middle">${Math.min(99, Math.round(age))}</text>`;
      }
    }
    return `<svg viewBox="0 0 ${W} ${H}" fill="none" aria-hidden="true">${grid}${shapes}</svg>`;
  }

  /* ------------------------------------------------------------------------
     Shared Settings Popover Binding
     ------------------------------------------------------------------------ */
  const state = {
    page: document.body.dataset.page || "standings",
    apiBase: resolveApiBase(),
    mode: "replay",
    syncDelayS: Number(localStorage.getItem(STORAGE_KEY_DELAY) || 0),
    delayBuffer: [],
    latestPacket: null,
    selectedDriverA: 1,
    selectedDriverB: 4,
    drawerFilter: "all",
    transcripts: new Map(),
    transcribingUrls: new Set(),
    playedAutoUrls: new Set(),
    currentlyPlayingUrl: null,
    ws: null,
    wsReconnectTimer: null,
    fixtureOverride: null,
  };

  function bindSettingsPopover(onHostSaved) {
    const pop = $("settings-popover");
    const btnSet = $("btn-settings");
    if (!pop || !btnSet) return;

    btnSet.addEventListener("click", () => {
      const hidden = pop.classList.toggle("hidden");
      btnSet.setAttribute("aria-expanded", hidden ? "false" : "true");
    });
    const closeBtn = $("btn-close-settings");
    if (closeBtn) {
      closeBtn.addEventListener("click", () => {
        pop.classList.add("hidden");
        btnSet.setAttribute("aria-expanded", "false");
      });
    }
    const hostInput = $("api-host-input");
    if (hostInput) hostInput.value = state.apiBase;

    const saveBtn = $("btn-save-host");
    if (saveBtn && hostInput) {
      saveBtn.addEventListener("click", () => {
        const v = hostInput.value.trim().replace(/\/+$/, "");
        if (v) {
          localStorage.setItem(STORAGE_KEY_HOST, v);
          state.apiBase = v;
        } else {
          localStorage.removeItem(STORAGE_KEY_HOST);
          state.apiBase = resolveApiBase();
        }
        pop.classList.add("hidden");
        if (onHostSaved) onHostSaved();
      });
    }

    const slider = $("sync-delay-slider");
    if (slider) {
      slider.value = String(state.syncDelayS);
      if ($("sync-delay-value")) $("sync-delay-value").textContent = `${state.syncDelayS}s`;
      slider.addEventListener("input", () => {
        state.syncDelayS = Number(slider.value) || 0;
        if ($("sync-delay-value")) $("sync-delay-value").textContent = `${state.syncDelayS}s`;
        localStorage.setItem(STORAGE_KEY_DELAY, String(state.syncDelayS));
        if (state.syncDelayS === 0) state.delayBuffer = [];
      });
    }

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") pop.classList.add("hidden");
    });
  }

  /* ========================================================================
     PAGE 1: STANDINGS & SCHEDULE (`index.html`)
     ======================================================================== */
  async function initStandingsPage() {
    // WDC / WCC sub-tabs
    document.querySelectorAll("[data-standings-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const tab = btn.dataset.standingsTab;
        document.querySelectorAll("[data-standings-tab]").forEach((b) => {
          const act = b === btn;
          b.classList.toggle("active", act);
          b.setAttribute("aria-selected", act ? "true" : "false");
        });
        $("panel-wdc").classList.toggle("hidden", tab !== "wdc");
        $("panel-wcc").classList.toggle("hidden", tab !== "wcc");
      });
    });

    const renderCircuits = (query) => {
      const box = $("circuits-catalog");
      const catalog = window.F1_CIRCUITS_2D || {};
      const q = String(query || "").trim().toLowerCase();
      const items = Object.values(catalog)
        .filter((c) => {
          if (!q) return true;
          return `${c.short_name || ""} ${c.meeting_name || ""} ${c.country_name || ""} ${c.location || ""}`.toLowerCase().includes(q);
        })
        .sort((a, b) => {
          if (a.slug === "las_vegas") return -1;
          if (b.slug === "las_vegas") return 1;
          return String(a.short_name || "").localeCompare(String(b.short_name || ""));
        });

      box.innerHTML = items.length
        ? items.map((c) => {
            const turns = Array.isArray(c.corners) ? c.corners.length : 0;
            const svg = window.buildCircuitThumbnailSvg ? window.buildCircuitThumbnailSvg(c) : "";
            return `
              <a href="live.html?circuit=${encodeURIComponent(c.slug)}" class="circuit-item">
                <div class="circuit-item-info">
                  <span class="circuit-item-name">${esc(c.short_name || c.slug)}</span>
                  <span class="circuit-item-meta">${esc(c.meeting_name || c.country_name || "")} · ${turns}T</span>
                </div>
                <div class="circuit-item-svg">${svg}</div>
              </a>`;
          }).join("")
        : `<div class="empty-state">No matching circuit found.</div>`;
    };

    const loadPastRaces = async (year) => {
      const box = $("past-races-list");
      box.innerHTML = `<div class="empty-state">Loading ${year} race sessions...</div>`;
      try {
        const r = await fetch(`${state.apiBase}/api/sessions?year=${year}`);
        if (!r.ok) throw new Error("fetch failed");
        const data = await r.json();
        const sessions = (data.sessions || []).slice().reverse();
        if (!sessions.length) {
          box.innerHTML = `<div class="empty-state">No completed race sessions found for ${year}.</div>`;
          return;
        }
        box.innerHTML = sessions.map((s) => `
          <a href="live.html?session_key=${encodeURIComponent(s.session_key)}&year=${year}" class="race-launch-row">
            <div class="race-launch-left">
              <span class="race-round-badge">R${String(s.round || "?").padStart(2, "0")}</span>
              <div class="drv-text">
                <span class="race-launch-name">${esc(s.grand_prix)}</span>
                <span class="race-launch-meta">${esc(s.circuit_name || "")}</span>
              </div>
            </div>
            <span class="race-launch-cta">Watch Replay &amp; Results</span>
          </a>`).join("");
      } catch (_) {
        box.innerHTML = `<div class="empty-state">Connect to backend server to list ${year} replays.</div>`;
      }
    };

    const loadAll = async () => {
      renderCircuits($("circuit-search").value);
      loadPastRaces($("season-replay-year").value);

      try {
        const r = await fetch(`${state.apiBase}/api/standings`);
        if (r.ok) {
          const data = await r.json();
          $("wdc-meta").textContent = data.after_round ? `${data.season} · After R${data.after_round}` : `${data.season}`;
          const drivers = data.drivers || [];
          const maxPts = Math.max(1, Number((drivers[0] && drivers[0].points) || 1));
          $("wdc-list").innerHTML = drivers.length
            ? drivers.map((d, i) => {
                const col = hexCol(d.team_colour, d.team_name);
                const pct = Math.max(4, Math.round(((Number(d.points) || 0) / maxPts) * 100));
                return `
                  <div class="wdc-row">
                    <span class="wdc-pos ${i === 0 ? "leader" : ""}">${String(d.position || i + 1).padStart(2, "0")}</span>
                    <div class="drv-identity">
                      <span class="team-bar" style="background:${col}"></span>
                      <div class="drv-text">
                        <div class="drv-name-line">
                          <span class="drv-code">${esc(d.driver_code)}</span>
                          <span class="drv-full" title="${esc(d.full_name)}">${esc(d.full_name)}</span>
                        </div>
                        <span class="drv-team">${esc(d.team_name)}</span>
                      </div>
                    </div>
                    <div class="pts-bar-track"><div class="pts-bar-fill" style="width:${pct}%;background:${col}"></div></div>
                    <span class="mono-muted align-right">${Number(d.wins || 0)}W</span>
                    <span class="mono-val align-right" style="font-weight:700">${Math.round(Number(d.points || 0))}</span>
                  </div>`;
              }).join("")
            : `<div class="empty-state">No championship standings available.</div>`;

          const cons = data.constructors || [];
          $("wcc-list").innerHTML = cons.length
            ? cons.map((c, i) => {
                const col = hexCol(c.team_colour, c.team_name);
                return `
                  <div class="wcc-row">
                    <span class="wdc-pos ${i === 0 ? "leader" : ""}">${String(c.position || i + 1).padStart(2, "0")}</span>
                    <span class="team-swatch" style="background:${col}"></span>
                    <span class="wcc-name">${esc(c.team_name)}</span>
                    <span class="mono-muted align-right">${Number(c.wins || 0)}W</span>
                    <span class="mono-val align-right" style="font-weight:700">${Math.round(Number(c.points || 0))} PTS</span>
                  </div>`;
              }).join("")
            : `<div class="empty-state">No constructor standings available.</div>`;
        }
      } catch (_) {}

      try {
        const r = await fetch(`${state.apiBase}/api/next-race`);
        if (r.ok) {
          const nr = await r.json();
          $("next-race-title").textContent = `Next Race · ${nr.grand_prix || "Grand Prix"}`;
          $("next-race-countdown").textContent = (nr.countdown && nr.countdown.label) || `ROUND ${nr.round || "--"}`;
          $("session-pill-text").textContent = `ROUND ${nr.round || "--"} · ${(nr.grand_prix || "").toUpperCase()}`;
          const sess = nr.sessions || [];
          $("next-race-sessions").innerHTML = sess.length
            ? sess.map((s) => {
                const t = s.times || {};
                const isMain = String(s.name || "").toLowerCase().includes("grand prix");
                const ist = t.ist ? `${t.ist.day_short || ""} ${t.ist.time_24h || "--"}`.trim() : "--";
                const uk = t.uk ? t.uk.time_24h || "--" : "--";
                const usEt = t.us_eastern ? t.us_eastern.time_24h || "--" : "--";
                const usPt = t.us_pacific ? t.us_pacific.time_24h || "--" : "--";
                return `
                  <div class="tz-row ${isMain ? "is-main" : ""}">
                    <span class="tz-session">${esc(s.name)}</span>
                    <span class="align-right tz-ist">${esc(ist)}</span>
                    <span class="align-right tz-uk">${esc(uk)}</span>
                    <span class="align-right tz-us">${esc(usEt)} / ${esc(usPt)}</span>
                  </div>`;
              }).join("")
            : `<div class="empty-state">No upcoming session times available.</div>`;
        }
      } catch (_) {}
    };

    $("circuit-search").addEventListener("input", (e) => renderCircuits(e.target.value));
    $("season-replay-year").addEventListener("change", (e) => loadPastRaces(e.target.value));
    bindSettingsPopover(loadAll);
    loadAll();
  }

  /* ========================================================================
     PAGE 2: LIVE & PAST-RACE REPLAY PIT-WALL (`live.html`)
     ======================================================================== */
  function initLivePage() {
    const trackRenderer = new window.TrackCanvasRenderer($("track-canvas"), { compact: false });
    const telemetryRenderer = new window.TelemetryChartRenderer($("telemetry-canvas"));

    // Timing vs Race Results sub-tabs
    document.querySelectorAll("[data-timing-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const tab = btn.dataset.timingTab;
        document.querySelectorAll("[data-timing-tab]").forEach((b) => {
          const act = b === btn;
          b.classList.toggle("active", act);
          b.setAttribute("aria-selected", act ? "true" : "false");
        });
        $("timing-stints-view").classList.toggle("hidden", tab !== "stints");
        $("timing-results-view").classList.toggle("hidden", tab !== "results");
      });
    });

    const openDrawer = () => {
      const bd = $("comms-backdrop");
      const dr = $("comms-drawer");
      bd.classList.remove("hidden");
      dr.classList.remove("hidden");
      requestAnimationFrame(() => { bd.classList.add("open"); dr.classList.add("open"); });
      renderComms();
    };
    const closeDrawer = () => {
      const bd = $("comms-backdrop");
      const dr = $("comms-drawer");
      bd.classList.remove("open");
      dr.classList.remove("open");
      setTimeout(() => {
        if (!dr.classList.contains("open")) {
          bd.classList.add("hidden");
          dr.classList.add("hidden");
        }
      }, 280);
    };

    $("btn-open-comms-full").addEventListener("click", openDrawer);
    $("btn-close-comms-drawer").addEventListener("click", closeDrawer);
    $("comms-backdrop").addEventListener("click", closeDrawer);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });

    document.querySelectorAll("[data-drawer-filter]").forEach((b) => {
      b.addEventListener("click", () => {
        state.drawerFilter = b.dataset.drawerFilter;
        document.querySelectorAll("[data-drawer-filter]").forEach((x) => x.classList.toggle("active", x === b));
        renderComms();
      });
    });

    const playRadioClip = (url) => {
      const audio = $("radio-audio-player");
      if (!audio || !url) return;
      if (state.currentlyPlayingUrl === url && !audio.paused) {
        audio.pause();
        state.currentlyPlayingUrl = null;
        renderComms();
        return;
      }
      state.currentlyPlayingUrl = url;
      audio.src = url;
      audio.play().catch(() => { state.currentlyPlayingUrl = null; renderComms(); });
      renderComms();
    };
    $("radio-audio-player").addEventListener("ended", () => {
      state.currentlyPlayingUrl = null;
      renderComms();
    });

    const requestWhisper = async (url) => {
      state.transcribingUrls.add(url);
      try {
        const res = await fetch(`${state.apiBase}/api/radio/transcribe?url=${encodeURIComponent(url)}`);
        if (res.ok) {
          const data = await res.json();
          state.transcripts.set(url, (data && data.transcript) ? data.transcript.trim() : "");
        }
      } catch (_) {} finally {
        state.transcribingUrls.delete(url);
        renderComms();
      }
    };

    const renderEventCards = (container, events) => {
      if (!container) return;
      if (!events.length) {
        container.innerHTML = `<div class="empty-state">No messages yet for this session.</div>`;
        return;
      }
      const barHeights = [6, 12, 15, 9, 14, 16, 10, 7, 13, 15, 11, 8, 14, 10, 6];
      container.innerHTML = events.map((ev) => {
        const clock = fmtUtcClock(ev.date);
        const timeMeta = ev.lap != null ? `L${ev.lap} · ${clock}` : clock;
        if (ev.kind === "radio") {
          const isPlaying = state.currentlyPlayingUrl === ev.recording_url;
          const quoteHtml = ev.transcript
            ? `<p class="comms-body">"${esc(ev.transcript)}"</p>`
            : ev.transcribing
            ? `<p class="comms-body transcribing">Transcribing audio with Whisper...</p>`
            : `<p class="comms-body transcribing">Tap Play to listen to team radio audio.</p>`;
          const bars = barHeights.map((bh) => `<span class="wf-bar" style="height:${bh}px;background:${ev.team_colour}"></span>`).join("");
          return `
            <article class="comms-item is-radio" style="border-color:${ev.team_colour}66">
              <div class="comms-item-head">
                <span class="comms-tag">${esc(ev.tag)}</span>
                <span class="comms-time">${esc(timeMeta)}</span>
              </div>
              ${quoteHtml}
              <div class="radio-controls">
                <button type="button" class="radio-play-btn ${isPlaying ? "playing" : ""}" data-audio-url="${esc(ev.recording_url)}">${isPlaying ? "Pause" : "Play"}</button>
                <div class="waveform-bars" aria-hidden="true">${bars}</div>
              </div>
            </article>`;
        }
        let cls = "";
        if (ev.flag.includes("PURPLE") || ev.message.toUpperCase().includes("FASTEST LAP")) cls = "is-purple";
        else if (ev.flag.includes("RED")) cls = "is-red";
        else if (ev.flag.includes("YELLOW") || ev.flag.includes("SC")) cls = "is-yellow";
        else if (ev.flag.includes("GREEN") || ev.flag.includes("CLEAR")) cls = "is-green";
        return `
          <article class="comms-item ${cls}">
            <div class="comms-item-head">
              <span class="comms-tag">${esc(ev.tag)}</span>
              <span class="comms-time">${esc(timeMeta)}</span>
            </div>
            <p class="comms-body">${esc(ev.message)}</p>
          </article>`;
      }).join("");

      container.querySelectorAll(".radio-play-btn").forEach((b) => {
        b.addEventListener("click", () => playRadioClip(b.dataset.audioUrl));
      });
    };

    const renderComms = () => {
      const race = (state.latestPacket && (state.latestPacket.race || state.latestPacket.tick)) || {};
      const fia = (race.race_control || []).map((m) => ({
        kind: "fia",
        date: m.date || "",
        lap: m.lap_number,
        flag: String(m.flag || "").toUpperCase(),
        tag: `FIA · ${m.category || m.flag || "RACE CONTROL"}`.toUpperCase(),
        message: m.message || "",
      })).sort((a, b) => String(b.date).localeCompare(String(a.date)));

      const radio = (race.team_radio || []).map((r) => ({
        kind: "radio",
        date: r.date || "",
        driver_number: r.driver_number,
        team_colour: hexCol(r.team_colour, r.team_name),
        recording_url: r.recording_url,
        transcript: r.transcript || state.transcripts.get(r.recording_url) || null,
        transcribing: state.transcribingUrls.has(r.recording_url),
        tag: `${r.driver_code || r.driver_number} #${r.driver_number} · ${(r.team_name || "RADIO").toUpperCase()}`,
      })).sort((a, b) => String(b.date).localeCompare(String(a.date)));

      $("comms-count-badge").textContent = String(fia.length + radio.length);
      $("fia-count-meta").textContent = `${fia.length} events`;
      renderEventCards($("fia-feed-inline"), fia);
      renderEventCards($("radio-feed-inline"), radio);

      if (!$("comms-drawer").classList.contains("hidden")) {
        const merged = [...fia, ...radio].sort((a, b) => String(b.date).localeCompare(String(a.date)));
        const filtered = state.drawerFilter === "all" ? merged : merged.filter((e) => e.kind === state.drawerFilter);
        renderEventCards($("comms-drawer-list"), filtered);
      }
    };

    const applyPacket = (pkt) => {
      if (!pkt || pkt.type === "post_quali_prediction") return;
      state.latestPacket = pkt;
      const race = pkt.race || pkt.tick || {};
      const drivers = Array.isArray(race.drivers) ? race.drivers : [];
      const lapCur = race.lap_current != null ? race.lap_current : 0;
      const lapTot = race.lap_total != null ? race.lap_total : 50;
      const caps = pkt.capabilities || { gps_available: false, telemetry_available: false };

      const status = String(race.track_status || "GREEN").toUpperCase();
      $("flag-pill").className = `status-pill flag-${status.toLowerCase()}`;
      $("flag-pill-text").textContent = status === "SC" ? "SAFETY CAR" : status === "VSC" ? "VIRTUAL SC" : `${status} FLAG`;
      const shortGp = String(pkt.grand_prix || "Live Session").replace(/\s*Grand Prix$/i, " GP").toUpperCase();
      $("session-pill-text").textContent = `${shortGp} · L${lapCur}/${lapTot}`;

      if (race.weather) {
        const air = race.weather.air_temperature_c != null ? `${Number(race.weather.air_temperature_c).toFixed(1)}°C` : "--°C";
        const trk = race.weather.track_temperature_c != null ? `${Number(race.weather.track_temperature_c).toFixed(1)}°C` : "--°C";
        $("weather-readout").textContent = `AIR ${air} · TRK ${trk}`;
      }

      if (pkt.circuit) {
        const resolved = (pkt.circuit.track_outline && pkt.circuit.track_outline.length >= 10)
          ? pkt.circuit
          : (pkt.circuit.slug && window.F1_CIRCUITS_2D && window.F1_CIRCUITS_2D[pkt.circuit.slug]);
        if (resolved) {
          trackRenderer.setCircuit(resolved);
          const rawName = String(pkt.circuit.name || resolved.short_name || "Circuit").trim();
          $("live-circuit-name").textContent = /circuit$/i.test(rawName) ? rawName : `${rawName} Circuit`;
          $("live-circuit-meta").textContent = Array.isArray(resolved.corners) ? `${resolved.corners.length}T` : "";
        }
      }

      $("gps-lock-overlay").classList.toggle("hidden", Boolean(caps.gps_available));
      $("telem-lock-overlay").classList.toggle("hidden", Boolean(caps.telemetry_available));

      const sb = race.session_best || {};
      $("fastest-lap-val").textContent = sb.fastest_lap
        ? `${sb.fastest_lap.formatted || "--"} · ${sb.fastest_lap.driver_code}${sb.fastest_lap.lap_number ? ` (L${sb.fastest_lap.lap_number})` : ""}`
        : "--:--.---";
      const setSec = (prefix, rec) => {
        $(`${prefix}-time`).textContent = rec && rec.formatted ? `${rec.formatted}s` : "--.---s";
        $(`${prefix}-who`).textContent = rec ? `${rec.driver_code}${rec.team_name ? ` · ${rec.team_name}` : ""}` : "--";
      };
      setSec("s1-best", sb.fastest_s1);
      setSec("s2-best", sb.fastest_s2);
      setSec("s3-best", sb.fastest_s3);

      if (drivers.length >= 2) {
        const nums = new Set(drivers.map((d) => Number(d.driver_number)));
        if (!nums.has(state.selectedDriverA)) state.selectedDriverA = Number(drivers[0].driver_number);
        if (!nums.has(state.selectedDriverB) || state.selectedDriverB === state.selectedDriverA) {
          const second = drivers.find((d) => Number(d.driver_number) !== state.selectedDriverA);
          if (second) state.selectedDriverB = Number(second.driver_number);
        }
      }

      // Ruler
      const total = Math.max(10, Number(lapTot) || 50);
      const ticks = [];
      for (let l = 10; l <= total; l += 10) {
        const pct = ((l / total) * 92 + 3.5).toFixed(1);
        ticks.push(`<span class="stint-ruler-tick" style="left:${pct}%">${l}</span>`);
      }
      $("stint-ruler").innerHTML = ticks.join("");

      // 22-Driver Timing & Tyre Stints
      const tList = $("timing-list");
      const rList = $("results-list");
      if (!drivers.length) {
        tList.innerHTML = `<div class="empty-state">Select a completed race above or switch to Live Race Feed.</div>`;
        rList.innerHTML = `<div class="empty-state">No race classification loaded yet.</div>`;
      } else {
        tList.innerHTML = drivers.map((d) => {
          const dn = Number(d.driver_number);
          const col = hexCol(d.team_colour, d.team_name);
          const gap = fmtGap(d);
          const secs = d.sectors || {};
          const selCls = dn === state.selectedDriverA ? "selected-a" : dn === state.selectedDriverB ? "selected-b" : "";
          const stintSvg = buildStintTimelineSvg(d, lapCur, lapTot);
          return `
            <button type="button" class="timing-row ${selCls} ${d.retired ? "is-retired" : ""}" data-dn="${dn}" role="listitem" title="Compare telemetry: ${esc(d.full_name)}">
              <span class="t-pos ${d.retired ? "out" : ""}">${d.retired ? "OUT" : String(d.position || "--").padStart(2, "0")}</span>
              <span class="t-drv">
                <span class="t-stripe" style="background:${col}"></span>
                <span class="t-code">${esc(d.code)}</span>
              </span>
              <span class="t-gap ${gap.cls}">${esc(gap.text)}</span>
              <span class="t-sectors">
                <span class="sec-pip ${esc(secs.s1_status || "NONE")}"></span>
                <span class="sec-pip ${esc(secs.s2_status || "NONE")}"></span>
                <span class="sec-pip ${esc(secs.s3_status || "NONE")}"></span>
              </span>
              <span class="stint-svg-cell">${stintSvg}</span>
            </button>`;
        }).join("");

        tList.querySelectorAll(".timing-row").forEach((row) => {
          row.addEventListener("click", () => {
            const dn = Number(row.dataset.dn);
            if (dn === state.selectedDriverA) return;
            state.selectedDriverB = state.selectedDriverA;
            state.selectedDriverA = dn;
            applyPacket(state.latestPacket);
          });
        });

        // Race Results / Classification Tab
        rList.innerHTML = drivers.map((d, i) => {
          const col = hexCol(d.team_colour, d.team_name);
          const gap = fmtGap(d);
          const bestStr = fmtLap(d.best_lap_time_s || d.last_lap_time_s);
          const compKey = String((d.tyre && d.tyre.compound) || "UNKNOWN").toUpperCase();
          const compMeta = COMPOUND_META[compKey] || COMPOUND_META.UNKNOWN;
          const tyreAge = d.tyre && d.tyre.age_laps != null ? `${Math.max(1, Math.round(Number(d.tyre.age_laps)))}L` : "";
          return `
            <div class="results-row">
              <span class="t-pos ${d.retired ? "out" : ""}">${d.retired ? "OUT" : String(d.position || i + 1).padStart(2, "0")}</span>
              <div class="drv-identity">
                <span class="team-bar" style="background:${col}"></span>
                <div class="drv-text">
                  <div class="drv-name-line">
                    <span class="drv-code">${esc(d.code)}</span>
                    <span class="drv-full">${esc(d.full_name)}</span>
                  </div>
                  <span class="drv-team">${esc(d.team_name)}</span>
                </div>
              </div>
              <span class="align-right"><span class="tyre-chip-sm" style="color:${compMeta.color};border-color:${compMeta.color}55">${compMeta.letter}${tyreAge ? ` ${tyreAge}` : ""}</span></span>
              <span class="mono-val align-right">${bestStr}</span>
              <span class="t-gap align-right ${gap.cls}">${esc(gap.text)}</span>
            </div>`;
        }).join("");
      }

      trackRenderer.setSelectedDrivers([state.selectedDriverA, state.selectedDriverB]);
      trackRenderer.updateDrivers(drivers, caps.gps_available);

      const drvA = drivers.find((d) => Number(d.driver_number) === state.selectedDriverA);
      const drvB = drivers.find((d) => Number(d.driver_number) === state.selectedDriverB);
      if (drvA) {
        const sp = drvA.telemetry && drvA.telemetry.speed_kph != null ? `${Math.round(drvA.telemetry.speed_kph)} KPH` : "--";
        const gr = drvA.telemetry && drvA.telemetry.gear != null ? ` · G${drvA.telemetry.gear}` : "";
        $("telem-drv-a").textContent = `${drvA.code} ${sp}${gr}`;
      }
      if (drvB) {
        const sp = drvB.telemetry && drvB.telemetry.speed_kph != null ? `${Math.round(drvB.telemetry.speed_kph)} KPH` : "--";
        const gr = drvB.telemetry && drvB.telemetry.gear != null ? ` · G${drvB.telemetry.gear}` : "";
        $("telem-drv-b").textContent = `${drvB.code} ${sp}${gr}`;
      }
      const drsVal = drvA && drvA.telemetry ? drvA.telemetry.drs_active : null;
      $("telem-drs-badge").textContent = drsVal === true ? "DRS OPEN" : drsVal === false ? "DRS CLOSED" : "DRS N/A";

      telemetryRenderer.setDriversMeta(
        { code: (drvA && drvA.code) || "A", color: "#38BDF8" },
        { code: (drvB && drvB.code) || "B", color: "#FF8000" }
      );
      telemetryRenderer.pushSample(drvA && drvA.telemetry, drvB && drvB.telemetry, caps.telemetry_available);

      const isReplay = pkt.type === "replay_frame";
      state.mode = pkt.mode || (isReplay ? "replay" : "free_live");
      document.querySelectorAll(".seg-opt[data-mode]").forEach((b) => {
        b.classList.toggle("active", b.dataset.mode === state.mode);
      });
      $("replay-controls").classList.toggle("hidden", !isReplay);
      if (isReplay && pkt.replay) {
        $("btn-play-pause").textContent = pkt.replay.paused ? "Play" : "Pause";
        document.querySelectorAll(".spd-btn").forEach((b) => {
          b.classList.toggle("active", Number(b.dataset.speed) === Number(pkt.replay.speed));
        });
        const seekEl = $("seek-lap-input");
        seekEl.max = String(lapTot);
        if (document.activeElement !== seekEl) seekEl.value = String(Math.max(1, lapCur));
        $("seek-lap-readout").textContent = `${lapCur}/${lapTot}`;
      }

      // Whisper transcripts & autoplay
      const clips = race.team_radio || [];
      for (const clip of clips) {
        const u = clip.recording_url;
        if (!u) continue;
        if (clip.transcript) state.transcripts.set(u, clip.transcript);
        else if (!state.transcripts.has(u) && !state.transcribingUrls.has(u) && !state.fixtureOverride) requestWhisper(u);
      }
      if ($("radio-autoplay-check").checked && clips.length) {
        const newest = clips[clips.length - 1];
        if (newest.recording_url && !state.playedAutoUrls.has(newest.recording_url)) {
          state.playedAutoUrls.add(newest.recording_url);
          playRadioClip(newest.recording_url);
        }
      }
      renderComms();
    };

    const loadSessionsForYear = async (year, autoSelectKey) => {
      const sel = $("session-select");
      try {
        const r = await fetch(`${state.apiBase}/api/sessions?year=${year}`);
        if (!r.ok) return;
        const d = await r.json();
        const sessions = d.sessions || [];
        sel.innerHTML = sessions.length
          ? sessions.map((s) => `<option value="${s.session_key}">R${s.round || "?"} · ${esc(s.grand_prix)}</option>`).join("")
          : `<option value="">No races found</option>`;
        if (autoSelectKey) sel.value = String(autoSelectKey);
        else if (sessions.length) sel.value = String(sessions[sessions.length - 1].session_key);
      } catch (_) {}
    };

    const loadReplay = async (sessionKey) => {
      if (!sessionKey) return;
      $("btn-load-session").textContent = "Loading...";
      try {
        await fetch(`${state.apiBase}/api/replay/load`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_key: Number(sessionKey), include_telemetry: true, autoplay: true }),
        });
      } finally {
        $("btn-load-session").textContent = "Load Race";
      }
    };

    const sendControl = async (payload) => {
      try {
        await fetch(`${state.apiBase}/api/replay/control`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      } catch (_) {}
    };

    const switchMode = async (mode) => {
      state.mode = mode;
      document.querySelectorAll(".seg-opt[data-mode]").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
      try {
        await fetch(`${state.apiBase}/api/mode`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode }),
        });
      } catch (_) {}
    };

    const connectWs = () => {
      if (state.ws) { try { state.ws.close(); } catch (_) {} }
      const url = toWsUrl(state.apiBase);
      $("ws-conn-status").textContent = `Connecting to ${url}...`;
      const ws = new WebSocket(url);
      state.ws = ws;
      ws.onopen = () => { $("ws-conn-status").textContent = `Connected · ${state.apiBase}`; };
      ws.onmessage = (evt) => {
        if (state.fixtureOverride) return;
        let pkt;
        try { pkt = JSON.parse(evt.data); } catch (_) { return; }
        if (!pkt || pkt.type === "pong") return;
        if (state.syncDelayS <= 0) applyPacket(pkt);
        else {
          state.delayBuffer.push({ at: Date.now() + state.syncDelayS * 1000, pkt });
          if (state.delayBuffer.length > 400) state.delayBuffer.shift();
        }
      };
      ws.onclose = () => {
        $("ws-conn-status").textContent = "Reconnecting...";
        clearTimeout(state.wsReconnectTimer);
        state.wsReconnectTimer = setTimeout(connectWs, 2500);
      };
    };

    setInterval(() => {
      if (state.fixtureOverride || !state.delayBuffer.length) return;
      const now = Date.now();
      let due = null;
      while (state.delayBuffer.length && state.delayBuffer[0].at <= now) due = state.delayBuffer.shift().pkt;
      if (due) applyPacket(due);
    }, 200);

    document.querySelectorAll(".seg-opt[data-mode]").forEach((b) => b.addEventListener("click", () => switchMode(b.dataset.mode)));
    $("replay-year-select").addEventListener("change", (e) => loadSessionsForYear(e.target.value));
    $("btn-load-session").addEventListener("click", () => loadReplay($("session-select").value));
    $("btn-play-pause").addEventListener("click", () => {
      const paused = state.latestPacket && state.latestPacket.replay && state.latestPacket.replay.paused;
      sendControl({ action: paused ? "play" : "pause" });
    });
    document.querySelectorAll(".spd-btn").forEach((b) => b.addEventListener("click", () => sendControl({ action: "set_speed", speed: Number(b.dataset.speed) })));
    $("seek-lap-input").addEventListener("change", (e) => sendControl({ action: "seek_lap", lap: Number(e.target.value) }));

    bindSettingsPopover(() => { connectWs(); loadSessionsForYear($("replay-year-select").value); });

    // Check URL query params (?session_key=... or ?circuit=... from Standings page)
    const params = new URLSearchParams(window.location.search);
    const qYear = params.get("year") || "2026";
    const qSession = params.get("session_key");
    const qCircuit = params.get("circuit");
    $("replay-year-select").value = qYear;
    if (qCircuit && window.F1_CIRCUITS_2D && window.F1_CIRCUITS_2D[qCircuit]) {
      const c = window.F1_CIRCUITS_2D[qCircuit];
      trackRenderer.setCircuit(c);
      const rawName = String(c.short_name || c.slug || "Circuit").trim();
      $("live-circuit-name").textContent = /circuit$/i.test(rawName) ? rawName : `${rawName} Circuit`;
      $("live-circuit-meta").textContent = `${(c.corners || []).length}T`;
    }
    connectWs();
    loadSessionsForYear(qYear, qSession).then(() => {
      if (qSession) {
        loadReplay(qSession);
      } else if (!qCircuit) {
        setTimeout(() => {
          if (!state.latestPacket && $("session-select").value) {
            loadReplay($("session-select").value);
          }
        }, 900);
      }
    });
  }

  /* ========================================================================
     PAGE 3: DEDICATED POST-QUALIFYING PREDICTIONS (`predictions.html`)
     ======================================================================== */
  function initPredictionsPage() {
    const predTrackRenderer = new window.TrackCanvasRenderer($("pred-circuit-canvas"), { compact: true });

    const renderPostQuali = (pkt) => {
      if (!pkt || !pkt.predictions) return;
      const items = Array.isArray(pkt.predictions.items) ? pkt.predictions.items : [];
      $("pred-gp-meta").textContent = `${pkt.season || 2026} · Round ${pkt.round || "--"} · ${pkt.grand_prix || ""}`;
      $("session-pill-text").textContent = `${(pkt.grand_prix || "Post-Quali Forecast").toUpperCase()}`;

      const top3 = items.slice(0, 3);
      $("pred-podium-cards").innerHTML = top3.map((it, idx) => {
        const col = hexCol(it.team_colour, it.team_name);
        return `
          <article class="podium-card" style="border-color:${col}99">
            <div class="podium-card-top">
              <span class="podium-rank-pill" style="background:${col}">PROJECTED P${idx + 1}</span>
              <span class="mono-muted" style="font-size:11px">GRID P${it.grid_position || idx + 1}</span>
            </div>
            <div class="podium-card-main">
              <div>
                <div class="podium-code">${esc(it.driver_code)} #${it.driver_number}</div>
                <div class="podium-sub">${esc(it.full_name)} · ${esc(it.team_name)}</div>
              </div>
              <div class="podium-pct-wrap">
                <div class="podium-pct">${Number(it.podium_prob_pct || 0).toFixed(1)}%</div>
                <div class="podium-win-sub">TOP-3 PODIUM · WIN ${Number(it.win_prob_pct || 0).toFixed(1)}%</div>
              </div>
            </div>
          </article>`;
      }).join("");

      $("pred-table-list").innerHTML = items.length
        ? items.map((it, idx) => {
            const col = hexCol(it.team_colour, it.team_name);
            const pod = Number(it.podium_prob_pct || 0);
            const win = Number(it.win_prob_pct || 0);
            return `
              <div class="pred-row">
                <span class="wdc-pos ${idx < 3 ? "leader" : ""}">${String(it.rank || idx + 1).padStart(2, "0")}</span>
                <span class="grid-badge">P${it.grid_position || idx + 1}</span>
                <div class="drv-identity">
                  <span class="team-bar" style="background:${col}"></span>
                  <div class="drv-text">
                    <div class="drv-name-line">
                      <span class="drv-code">${esc(it.driver_code)}</span>
                      <span class="drv-full">${esc(it.full_name)} · ${esc(it.team_name)}</span>
                    </div>
                  </div>
                </div>
                <div class="pred-bar-track"><div class="pred-bar-fill" style="width:${Math.max(2, Math.min(100, pod))}%;background:${col}"></div></div>
                <span class="mono-val align-right" style="font-weight:700">${pod.toFixed(1)}%</span>
                <span class="mono-muted align-right">${win.toFixed(1)}%</span>
              </div>`;
          }).join("")
        : `<div class="empty-state">No qualifying grid available for this round yet.</div>`;

      const circ = pkt.circuit || {};
      const c2d = (circ.track_outline && circ.track_outline.length >= 10)
        ? circ
        : (circ.slug && window.F1_CIRCUITS_2D && window.F1_CIRCUITS_2D[circ.slug])
        || (window.F1_CIRCUITS_2D && window.F1_CIRCUITS_2D.las_vegas);
      if (c2d) {
        const rawName = String(circ.name || c2d.short_name || "Circuit").trim();
        $("pred-circuit-heading").textContent = /circuit$/i.test(rawName) ? rawName : `${rawName} Circuit`;
        $("pred-circuit-caption").textContent = `${pkt.grand_prix || ""} · ${(c2d.corners || []).length} Turns · S1 / S2 / S3 Split`;
        predTrackRenderer.setCircuit(c2d);
      }

      const weights = [
        ["Qualifying Grid Slot (P1–P22)", 0.41, "#E10600"],
        ["Constructor Rolling 5-Race Points", 0.24, "#38BDF8"],
        ["Driver 10-Race Podium Conversion", 0.19, "#FACC15"],
        ["Circuit Overtake & Safety Car Prior", 0.16, "#10B981"],
      ];
      $("feature-weights-list").innerHTML = weights.map(([label, w, col]) => `
        <div class="weight-item">
          <div class="weight-top">
            <span>${esc(label)}</span>
            <span class="mono-val" style="font-weight:700">${w.toFixed(2)}</span>
          </div>
          <div class="pred-bar-track"><div class="pred-bar-fill" style="width:${Math.round(w * 200)}%;background:${col}"></div></div>
        </div>`).join("");
    };

    const loadRounds = async (season) => {
      const sel = $("pred-round-select");
      try {
        const r = await fetch(`${state.apiBase}/api/sessions?year=${season}`);
        if (r.ok) {
          const d = await r.json();
          const sessions = d.sessions || [];
          if (sessions.length) {
            sel.innerHTML = sessions.map((s) => `<option value="${s.round || 1}">Round ${s.round || 1} — ${esc(s.grand_prix)}</option>`).join("");
            sel.value = String(sessions[sessions.length - 1].round || 1);
            return;
          }
        }
      } catch (_) {}
      sel.innerHTML = `<option value="1">Latest Completed Qualifying</option>`;
    };

    const runForecast = async () => {
      const season = Number($("pred-season-select").value) || 2026;
      const round = Number($("pred-round-select").value) || undefined;
      $("btn-run-post-quali").textContent = "Computing...";
      try {
        const r = await fetch(`${state.apiBase}/api/predict/post-quali`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ season, round }),
        });
        if (r.ok) renderPostQuali(await r.json());
      } finally {
        $("btn-run-post-quali").textContent = "Run Post-Quali Forecast";
      }
    };

    $("pred-season-select").addEventListener("change", (e) => loadRounds(e.target.value).then(runForecast));
    $("btn-run-post-quali").addEventListener("click", runForecast);
    bindSettingsPopover(() => loadRounds($("pred-season-select").value).then(runForecast));
    loadRounds($("pred-season-select").value).then(runForecast);
  }

  document.addEventListener("DOMContentLoaded", () => {
    if (state.page === "live") initLivePage();
    else if (state.page === "predictions") initPredictionsPage();
    else initStandingsPage();
  });
})();
