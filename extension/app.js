/**
 * F1 Pit-Wall — Multi-Page Application Controller
 *
 * Detects active page via `document.body.dataset.page`:
 *   - "standings"   (index.html / sidepanel.html)
 *   - "live"        (live.html — Past Race Final Standings + Live Race Movement)
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

  function fmtLap(sec, fallbackStr) {
    if (fallbackStr) return fallbackStr;
    if (typeof sec !== "number" || !Number.isFinite(sec) || sec <= 0) return "--";
    const mins = Math.floor(sec / 60);
    const rem = sec - mins * 60;
    return mins > 0 ? `${mins}:${rem.toFixed(3).padStart(6, "0")}` : `${rem.toFixed(3)}`;
  }

  function fmtGap(d) {
    if (d.gap_display) {
      if (d.retired) return { text: d.gap_display, cls: "is-out" };
      if (d.position === 1) return { text: d.gap_display, cls: "" };
      return { text: d.gap_display, cls: "" };
    }
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

  function formatCircuitHeading(raw) {
    const s = String(raw || "Circuit").trim();
    return /(circuit|autodrome|autodromo|autódromo|ring|park|madring)$/i.test(s) ? s : `${s} Circuit`;
  }

  function buildStintTimelineSvg(driver, lapCurrent, lapTotal) {
    const total = Math.max(10, Number(lapTotal) || 50);
    const maxLapForDriver = driver.laps_completed != null
      ? Math.max(1, Math.min(total, Number(driver.laps_completed)))
      : Math.max(1, Math.min(total, Number(lapCurrent) || total));
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
          lap_end: maxLapForDriver,
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
        : maxLapForDriver;
      eLap = Math.max(sLap, Math.min(maxLapForDriver, total, eLap));

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
     Shared State & Settings Popover
     ------------------------------------------------------------------------ */
  const state = {
    page: document.body.dataset.page || "standings",
    apiBase: resolveApiBase(),
    viewMode: "results", // "results" (final standings after final lap) | "live" (live race feed)
    syncDelayS: Number(localStorage.getItem(STORAGE_KEY_DELAY) || 0),
    delayBuffer: [],
    latestLivePacket: null,
    latestResultsData: null,
    selectedDriverA: 1,
    selectedDriverB: 4,
    drawerFilter: "all",
    transcripts: new Map(),
    transcribingUrls: new Set(),
    playedAutoUrls: new Set(),
    currentlyPlayingUrl: null,
    ws: null,
    wsReconnectTimer: null,
    sessionsCache: [],
    nextRaceCache: null,
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

  function bindWindowControls() {
    const floatBtn = $("btn-float-window");
    const fsBtn = $("btn-fullscreen");
    const fsSvg = $("fullscreen-icon-svg");

    const params = new URLSearchParams(window.location.search);
    const isCompanion = Boolean(window.opener) || params.get("companion") === "1";

    if (floatBtn) {
      if (isCompanion) {
        floatBtn.title = "Toggle Compact / Expanded Window Size";
        floatBtn.setAttribute("aria-label", "Toggle compact or expanded window size");
      }
      floatBtn.addEventListener("click", () => {
        const hasExtWin = typeof chrome !== "undefined" && chrome.windows && typeof chrome.windows.create === "function";
        if (hasExtWin && !isCompanion) {
          const rel = `${window.location.pathname.replace(/^\//, "") || "index.html"}?companion=1`;
          chrome.windows.create({
            url: chrome.runtime ? chrome.runtime.getURL(rel) : window.location.href,
            type: "popup",
            width: 580,
            height: 860,
            focused: true,
          });
          return;
        }
        if (isCompanion) {
          // Inside a resizable companion window: toggle between compact companion and wide desktop
          const isCompact = window.outerWidth < 900;
          const targetW = isCompact ? Math.min(1440, window.screen.availWidth - 40) : 580;
          const targetH = isCompact ? Math.min(920, window.screen.availHeight - 40) : 860;
          try { window.resizeTo(targetW, targetH); } catch (_) {}
          return;
        }
        const nextUrl = new URL(window.location.href);
        nextUrl.searchParams.set("companion", "1");
        window.open(
          nextUrl.toString(),
          "F1PitWallCompanion",
          "popup=yes,width=580,height=860,left=48,top=48,resizable=yes,scrollbars=yes"
        );
      });
    }

    const syncFsIcon = () => {
      if (!fsSvg || !fsBtn) return;
      const active = Boolean(document.fullscreenElement);
      fsBtn.title = active ? "Exit Full Screen" : "Enter Full Screen";
      fsBtn.setAttribute("aria-label", active ? "Exit full screen" : "Enter full screen");
      fsSvg.innerHTML = active
        ? `<polyline points="4 14 10 14 10 20"></polyline><polyline points="20 10 14 10 14 4"></polyline><line x1="14" y1="10" x2="21" y2="3"></line><line x1="3" y1="21" x2="10" y2="14"></line>`
        : `<polyline points="15 3 21 3 21 9"></polyline><polyline points="9 21 3 21 3 15"></polyline><line x1="21" y1="3" x2="14" y2="10"></line><line x1="3" y1="21" x2="10" y2="14"></line>`;
    };

    if (fsBtn) {
      fsBtn.addEventListener("click", async () => {
        try {
          if (document.fullscreenElement) {
            await document.exitFullscreen();
          } else {
            await document.documentElement.requestFullscreen();
          }
        } catch (_) {
          // Fallback inside extension popup windows if fullscreen API is restricted
          if (typeof chrome !== "undefined" && chrome.windows && chrome.windows.getCurrent) {
            chrome.windows.getCurrent((win) => {
              if (!win || !win.id) return;
              const nextState = win.state === "maximized" || win.state === "fullscreen" ? "normal" : "maximized";
              chrome.windows.update(win.id, { state: nextState });
            });
          }
        }
        syncFsIcon();
      });
      document.addEventListener("fullscreenchange", syncFsIcon);
    }
  }

  /* ========================================================================
     PAGE 1: STANDINGS & SCHEDULE (`index.html`)
     ======================================================================== */
  async function initStandingsPage() {
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
            const turns = Array.isArray(c.corners) && c.corners.length ? ` · ${c.corners.length}T` : "";
            const svg = window.buildCircuitThumbnailSvg ? window.buildCircuitThumbnailSvg(c) : "";
            return `
              <a href="live.html?circuit=${encodeURIComponent(c.slug)}" class="circuit-item">
                <div class="circuit-item-info">
                  <span class="circuit-item-name">${esc(c.short_name || c.slug)}</span>
                  <span class="circuit-item-meta">${esc(c.meeting_name || c.country_name || "")}${turns}</span>
                </div>
                <div class="circuit-item-svg">${svg}</div>
              </a>`;
          }).join("")
        : `<div class="empty-state">No matching circuit found.</div>`;
    };

    const loadSeasonRaces = async (year) => {
      const box = $("past-races-list");
      box.innerHTML = `<div class="empty-state">Loading ${year} calendar...</div>`;
      try {
        const r = await fetch(`${state.apiBase}/api/sessions?year=${year}`);
        if (!r.ok) throw new Error("fetch failed");
        const data = await r.json();
        const sessions = (data.sessions || []).slice().reverse();
        if (!sessions.length) {
          box.innerHTML = `<div class="empty-state">No race sessions found for ${year}.</div>`;
          return;
        }
        box.innerHTML = sessions.map((s) => {
          const isCompleted = s.completed !== false;
          const rLabel = `R${String(s.round || "?").padStart(2, "0")}`;
          if (!isCompleted) {
            return `
              <div class="race-launch-row is-upcoming" aria-disabled="true">
                <div class="race-launch-left">
                  <span class="race-round-badge">${rLabel}</span>
                  <div class="drv-text">
                    <span class="race-launch-name">${esc(s.grand_prix)}</span>
                    <span class="race-launch-meta">${esc(s.circuit_name || "")}</span>
                  </div>
                </div>
                <span class="race-upcoming-badge">NOT COMPLETED</span>
              </div>`;
          }
          return `
            <a href="live.html?view=results&year=${year}&round=${encodeURIComponent(s.round)}&session_key=${encodeURIComponent(s.session_key)}" class="race-launch-row">
              <div class="race-launch-left">
                <span class="race-round-badge">${rLabel}</span>
                <div class="drv-text">
                  <span class="race-launch-name">${esc(s.grand_prix)}</span>
                  <span class="race-launch-meta">${esc(s.circuit_name || "")}</span>
                </div>
              </div>
              <span class="race-launch-cta">Final Standings</span>
            </a>`;
        }).join("");
      } catch (_) {
        box.innerHTML = `<div class="empty-state">Connect to backend server to load ${year} calendar.</div>`;
      }
    };

    const loadAll = async () => {
      renderCircuits($("circuit-search").value);
      loadSeasonRaces($("season-replay-year").value);

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
    $("season-replay-year").addEventListener("change", (e) => loadSeasonRaces(e.target.value));
    bindSettingsPopover(loadAll);
    loadAll();
  }

  /* ========================================================================
     PAGE 2: LIVE RACE MOVEMENT & FINAL RACE RESULTS (`live.html`)
     ======================================================================== */
  function initLivePage() {
    const trackRenderer = new window.TrackCanvasRenderer($("track-canvas"), { compact: false });
    const telemetryRenderer = new window.TelemetryChartRenderer($("telemetry-canvas"));

    // Keyed DOM map for ZERO-FLICKER hover on live timing rows
    const timingRowMap = new Map();
    const resultsRowMap = new Map();
    let lastRulerKey = "";

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
        container.innerHTML = `<div class="empty-state">No messages recorded for this session.</div>`;
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

    let lastCommsSig = "";
    const renderComms = (force) => {
      const source = state.viewMode === "live"
        ? ((state.latestLivePacket && (state.latestLivePacket.race || state.latestLivePacket.tick)) || {})
        : (state.latestResultsData || {});

      const fiaRaw = source.race_control || [];
      const radioRaw = source.team_radio || [];
      const sig = JSON.stringify([
        state.viewMode,
        fiaRaw.length,
        radioRaw.length,
        state.currentlyPlayingUrl,
        state.transcripts.size,
        state.transcribingUrls.size,
        state.drawerFilter,
        !$("comms-drawer").classList.contains("hidden"),
      ]);
      if (!force && sig === lastCommsSig) return;
      lastCommsSig = sig;

      const fia = fiaRaw.map((m) => ({
        kind: "fia",
        date: m.date || "",
        lap: m.lap_number,
        flag: String(m.flag || "").toUpperCase(),
        tag: `FIA · ${m.category || m.flag || "RACE CONTROL"}`.toUpperCase(),
        message: m.message || "",
      })).sort((a, b) => String(b.date).localeCompare(String(a.date)));

      const radio = radioRaw.map((r) => ({
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

    const setSectorBests = (sb) => {
      const data = sb || {};
      $("fastest-lap-val").textContent = data.fastest_lap
        ? `${data.fastest_lap.formatted || "--"} · ${data.fastest_lap.driver_code}${data.fastest_lap.lap_number ? ` (L${data.fastest_lap.lap_number})` : ""}`
        : "--:--.---";
      const setSec = (prefix, rec) => {
        $(`${prefix}-time`).textContent = rec && rec.formatted ? `${rec.formatted}s` : "--.---s";
        $(`${prefix}-who`).textContent = rec ? `${rec.driver_code}${rec.team_name ? ` · ${rec.team_name}` : ""}` : "--";
      };
      setSec("s1-best", data.fastest_s1);
      setSec("s2-best", data.fastest_s2);
      setSec("s3-best", data.fastest_s3);
    };

    const setRuler = (lapTot) => {
      const total = Math.max(10, Number(lapTot) || 50);
      const rKey = String(total);
      if (rKey === lastRulerKey) return;
      lastRulerKey = rKey;
      const ticks = [];
      for (let l = 10; l <= total; l += 10) {
        const pct = ((l / total) * 92 + 3.5).toFixed(1);
        ticks.push(`<span class="stint-ruler-tick" style="left:${pct}%">${l}</span>`);
      }
      $("stint-ruler").innerHTML = ticks.join("");
    };

    /**
     * Keyed DOM reconciliation for the timing & tyre stint list.
     * Reuses existing row DOM nodes by driver_number so hovering NEVER flickers
     * even during 4 Hz live race telemetry updates.
     */
    const renderDriversStable = (drivers, lapCur, lapTot, isLiveMode) => {
      const tList = $("timing-list");
      const rList = $("results-list");
      setRuler(lapTot);

      if (!drivers.length) {
        timingRowMap.clear();
        resultsRowMap.clear();
        tList.innerHTML = `<div class="empty-state">${isLiveMode ? "Waiting for live timing data from F1 SignalR feed..." : "Select a completed Grand Prix above to view final standings."}</div>`;
        rList.innerHTML = `<div class="empty-state">No classification data loaded yet.</div>`;
        return;
      }

      // Clear empty-state placeholder if switching from empty -> populated
      if (tList.querySelector(".empty-state")) tList.innerHTML = "";
      if (rList.querySelector(".empty-state")) rList.innerHTML = "";

      const activeKeys = new Set();

      drivers.forEach((d, idx) => {
        const dn = Number(d.driver_number || idx + 1);
        activeKeys.add(dn);
        const col = hexCol(d.team_colour, d.team_name);
        const gap = fmtGap(d);
        const secs = d.sectors || {};
        const selA = isLiveMode && dn === state.selectedDriverA;
        const selB = isLiveMode && dn === state.selectedDriverB;

        // ---- 1. Stable Timing + Tyre Stints Row ----
        let row = timingRowMap.get(dn);
        if (!row) {
          row = document.createElement("button");
          row.type = "button";
          row.className = "timing-row";
          row.setAttribute("role", "listitem");
          row.dataset.dn = String(dn);
          row.innerHTML = `
            <span class="t-pos"></span>
            <span class="t-drv">
              <span class="t-stripe"></span>
              <span class="t-code"></span>
            </span>
            <span class="t-gap"></span>
            <span class="t-sectors">
              <span class="sec-pip" data-s="1"></span>
              <span class="sec-pip" data-s="2"></span>
              <span class="sec-pip" data-s="3"></span>
            </span>
            <span class="stint-svg-cell"></span>`;
          row.addEventListener("click", () => {
            if (state.viewMode !== "live") return;
            const clickedDn = Number(row.dataset.dn);
            if (clickedDn === state.selectedDriverA) return;
            state.selectedDriverB = state.selectedDriverA;
            state.selectedDriverA = clickedDn;
            if (state.latestLivePacket) applyLivePacket(state.latestLivePacket);
          });
          timingRowMap.set(dn, row);
        }

        const posText = d.retired ? "OUT" : String(d.position || idx + 1).padStart(2, "0");
        const posEl = row.querySelector(".t-pos");
        if (posEl.textContent !== posText) posEl.textContent = posText;
        posEl.classList.toggle("out", Boolean(d.retired));

        const stripeEl = row.querySelector(".t-stripe");
        if (stripeEl.dataset.col !== col) {
          stripeEl.style.background = col;
          stripeEl.dataset.col = col;
        }
        const codeEl = row.querySelector(".t-code");
        const codeTxt = String(d.code || dn);
        if (codeEl.textContent !== codeTxt) codeEl.textContent = codeTxt;

        const gapEl = row.querySelector(".t-gap");
        const compactGap = d.position === 1 && !isLiveMode ? "WINNER" : gap.text;
        if (gapEl.textContent !== compactGap) gapEl.textContent = compactGap;
        const gapCls = `t-gap ${gap.cls}`.trim();
        if (gapEl.className !== gapCls) gapEl.className = gapCls;

        const p1 = row.querySelector('[data-s="1"]');
        const p2 = row.querySelector('[data-s="2"]');
        const p3 = row.querySelector('[data-s="3"]');
        const c1 = `sec-pip ${secs.s1_status || "NONE"}`;
        const c2 = `sec-pip ${secs.s2_status || "NONE"}`;
        const c3 = `sec-pip ${secs.s3_status || "NONE"}`;
        if (p1.className !== c1) p1.className = c1;
        if (p2.className !== c2) p2.className = c2;
        if (p3.className !== c3) p3.className = c3;

        row.classList.toggle("selected-a", selA);
        row.classList.toggle("selected-b", selB);
        row.classList.toggle("is-retired", Boolean(d.retired));
        row.title = isLiveMode ? `Compare live telemetry: ${d.full_name || codeTxt}` : `${d.full_name || codeTxt} · ${d.team_name || ""}`;

        // Only rebuild stint SVG when stints/lap/age actually change
        const stintSig = JSON.stringify([
          lapCur,
          lapTot,
          d.laps_completed,
          Boolean(d.retired),
          d.tyre && d.tyre.compound,
          d.tyre && d.tyre.age_laps,
          (d.stints || []).map((s) => [s.compound, s.lap_start, s.lap_end]),
        ]);
        if (row.dataset.stintSig !== stintSig) {
          row.querySelector(".stint-svg-cell").innerHTML = buildStintTimelineSvg(d, lapCur, lapTot);
          row.dataset.stintSig = stintSig;
        }

        // Only touch DOM order if position actually changed
        if (tList.children[idx] !== row) {
          tList.insertBefore(row, tList.children[idx] || null);
        }

        // ---- 2. Stable Full Classification Row ----
        let rRow = resultsRowMap.get(dn);
        if (!rRow) {
          rRow = document.createElement("div");
          rRow.className = "results-row";
          rRow.innerHTML = `
            <span class="t-pos"></span>
            <div class="drv-identity">
              <span class="team-bar"></span>
              <div class="drv-text">
                <div class="drv-name-line">
                  <span class="drv-code"></span>
                  <span class="drv-full"></span>
                </div>
                <span class="drv-team"></span>
              </div>
            </div>
            <span class="mono-muted align-right r-grid"></span>
            <span class="align-right r-tyre"></span>
            <span class="mono-val align-right r-best"></span>
            <span class="t-gap align-right r-gap"></span>`;
          resultsRowMap.set(dn, rRow);
        }

        const rPos = rRow.querySelector(".t-pos");
        if (rPos.textContent !== posText) rPos.textContent = posText;
        rPos.classList.toggle("out", Boolean(d.retired));

        const rBar = rRow.querySelector(".team-bar");
        if (rBar.dataset.col !== col) {
          rBar.style.background = col;
          rBar.dataset.col = col;
        }
        const rCode = rRow.querySelector(".drv-code");
        if (rCode.textContent !== codeTxt) rCode.textContent = codeTxt;
        const rFull = rRow.querySelector(".drv-full");
        const fullTxt = String(d.full_name || "");
        if (rFull.textContent !== fullTxt) rFull.textContent = fullTxt;
        const rTeam = rRow.querySelector(".drv-team");
        const teamTxt = String(d.team_name || "");
        if (rTeam.textContent !== teamTxt) rTeam.textContent = teamTxt;

        const gridTxt = d.grid_position ? `P${d.grid_position}` : "--";
        const rGrid = rRow.querySelector(".r-grid");
        if (rGrid.textContent !== gridTxt) rGrid.textContent = gridTxt;

        const compKey = String((d.tyre && d.tyre.compound) || "UNKNOWN").toUpperCase();
        const compMeta = COMPOUND_META[compKey] || COMPOUND_META.UNKNOWN;
        const tyreAge = d.tyre && d.tyre.age_laps != null ? `${Math.max(1, Math.round(Number(d.tyre.age_laps)))}L` : "";
        const tyreSig = `${compMeta.letter}:${tyreAge}`;
        const rTyre = rRow.querySelector(".r-tyre");
        if (rTyre.dataset.sig !== tyreSig) {
          rTyre.innerHTML = `<span class="tyre-chip-sm" style="color:${compMeta.color};border-color:${compMeta.color}55">${compMeta.letter}${tyreAge ? ` ${tyreAge}` : ""}</span>`;
          rTyre.dataset.sig = tyreSig;
        }

        const bestStr = fmtLap(d.best_lap_time_s || d.last_lap_time_s, d.best_lap_formatted);
        const rBest = rRow.querySelector(".r-best");
        if (rBest.textContent !== bestStr) rBest.textContent = bestStr;

        const ptsSuffix = !isLiveMode && d.points > 0 ? ` · ${Math.round(d.points)}P` : "";
        const gapFinalTxt = `${gap.text}${ptsSuffix}`;
        const rGap = rRow.querySelector(".r-gap");
        if (rGap.textContent !== gapFinalTxt) rGap.textContent = gapFinalTxt;
        const rGapCls = `t-gap align-right r-gap ${gap.cls}`.trim();
        if (rGap.className !== rGapCls) rGap.className = rGapCls;

        if (rList.children[idx] !== rRow) {
          rList.insertBefore(rRow, rList.children[idx] || null);
        }
      });

      // Prune drivers no longer present (e.g. switching race)
      for (const [dn, el] of timingRowMap.entries()) {
        if (!activeKeys.has(dn)) {
          el.remove();
          timingRowMap.delete(dn);
        }
      }
      for (const [dn, el] of resultsRowMap.entries()) {
        if (!activeKeys.has(dn)) {
          el.remove();
          resultsRowMap.delete(dn);
        }
      }
    };

    const renderFinalPodiumCards = (data) => {
      const box = $("race-podium-cards");
      const drivers = Array.isArray(data.drivers) ? data.drivers.slice(0, 3) : [];
      $("podium-laps-meta").textContent = data.lap_total ? `${data.lap_total} Laps · Final Classification` : "Final Classification";
      if (!drivers.length) {
        box.innerHTML = `<div class="empty-state">${esc(data.message || "Race not completed yet.")}</div>`;
        return;
      }
      box.innerHTML = drivers.map((d, idx) => {
        const col = hexCol(d.team_colour, d.team_name);
        const gridDelta = d.grid_position ? `Grid P${d.grid_position}` : "";
        return `
          <article class="race-podium-card" style="border-left-color:${col}">
            <div class="race-podium-left">
              <span class="race-podium-pos">P${idx + 1}</span>
              <div class="drv-text">
                <div class="drv-name-line">
                  <span class="drv-code">${esc(d.code)}</span>
                  <span class="drv-full">${esc(d.full_name)}</span>
                </div>
                <span class="drv-team">${esc(d.team_name)}${gridDelta ? ` · ${gridDelta}` : ""}</span>
              </div>
            </div>
            <div class="race-podium-right">
              <span class="race-podium-gap">${esc(d.gap_display || (idx === 0 ? "WINNER" : "--"))}</span>
              <span class="race-podium-pts">${d.points ? `+${Math.round(d.points)} PTS` : ""}</span>
            </div>
          </article>`;
      }).join("");
    };

    const resolveLocalCircuit = (nr) => {
      const cat = window.F1_CIRCUITS_2D || {};
      const aliases = {
        losail: "lusail", qatar: "lusail", vegas: "las_vegas", yas_marina: "yas_marina_circuit",
        abu_dhabi: "yas_marina_circuit", albert_park: "melbourne", villeneuve: "montreal",
        red_bull_ring: "spielberg", spa: "spa_francorchamps", marina_bay: "singapore",
        americas: "austin", rodriguez: "mexico_city", monaco: "monte_carlo", bahrain: "sakhir",
      };
      for (const raw of [nr && nr.circuit_id, nr && nr.circuit_name, nr && nr.grand_prix]) {
        if (!raw) continue;
        const norm = String(raw).toLowerCase().replace(/grand prix|international|circuit/g, "").trim().replace(/[\s-]+/g, "_");
        if (cat[norm]) return cat[norm];
        if (aliases[norm] && cat[aliases[norm]]) return cat[aliases[norm]];
        for (const [k, v] of Object.entries(aliases)) {
          if (norm.includes(k) && cat[v]) return cat[v];
        }
        for (const item of Object.values(cat)) {
          const hay = `${item.short_name || ""} ${item.meeting_name || ""} ${item.location || ""}`.toLowerCase();
          if (hay.includes(String(raw).toLowerCase())) return item;
        }
      }
      return null;
    };

    const renderIdleLiveSchedule = async () => {
      $("live-idle-panel").classList.remove("hidden");
      $("timing-main-panel").classList.add("hidden");
      $("podium-summary-panel").classList.add("hidden");
      $("live-telemetry-panel").classList.add("hidden");
      $("comms-section-wrap").classList.add("hidden");
      $("gps-lock-overlay").classList.add("hidden");
      $("fastest-lap-banner").classList.add("hidden");
      $("sector-records-row").classList.add("hidden");

      $("flag-pill").className = "status-pill flag-yellow";
      $("flag-pill-text").textContent = "NO LIVE RACE";
      if ($("live-badge-text")) $("live-badge-text").textContent = "SIGNALR FEED STANDBY";
      $("live-stream-meta").textContent = "Waiting for Next Session";
      $("comms-count-badge").textContent = "0";
      setSectorBests(null);
      trackRenderer.updateDrivers([], false);

      try {
        if (!state.nextRaceCache) {
          const r = await fetch(`${state.apiBase}/api/next-race`);
          if (r.ok) state.nextRaceCache = await r.json();
        }
        const nr = state.nextRaceCache;
        if (nr) {
          const gpShort = String(nr.grand_prix || "Next Race").replace(/\s*Grand Prix$/i, " GP").toUpperCase();
          $("session-pill-text").textContent = `NEXT · ${gpShort}`;
          $("idle-next-race-title").textContent = `Next Race · ${nr.grand_prix || "Grand Prix"}`;
          $("idle-next-countdown").textContent = (nr.countdown && nr.countdown.label) || `ROUND ${nr.round || "--"}`;
          $("idle-next-round-meta").textContent = `ROUND ${nr.round || "--"}${nr.date ? ` · ${nr.date}` : ""}`;

          const sess = nr.sessions || [];
          $("idle-next-race-sessions").innerHTML = sess.length
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

          const c2d = resolveLocalCircuit(nr);
          if (c2d) {
            trackRenderer.setCircuit(c2d);
            $("live-circuit-name").textContent = formatCircuitHeading(nr.circuit_name || c2d.short_name);
            $("live-circuit-meta").textContent = Array.isArray(c2d.corners) && c2d.corners.length
              ? `Upcoming R${nr.round || "--"} · ${c2d.corners.length}T`
              : `Upcoming R${nr.round || "--"}`;
          }
        }
      } catch (_) {}
    };

    const applyFinalResults = (data) => {
      state.latestResultsData = data;
      $("live-idle-panel").classList.add("hidden");
      $("timing-main-panel").classList.remove("hidden");
      $("podium-summary-panel").classList.remove("hidden");
      $("live-telemetry-panel").classList.add("hidden");
      $("comms-section-wrap").classList.remove("hidden");
      $("fastest-lap-banner").classList.remove("hidden");
      $("sector-records-row").classList.remove("hidden");

      const completed = data.completed !== false;
      const shortGp = String(data.grand_prix || "Grand Prix").replace(/\s*Grand Prix$/i, " GP").toUpperCase();

      $("flag-pill").className = completed ? "status-pill flag-chequered" : "status-pill flag-yellow";
      $("flag-pill-text").textContent = completed ? "FINAL CLASSIFICATION" : "NOT COMPLETED";
      $("session-pill-text").textContent = completed
        ? `${shortGp} · ${data.lap_total || "--"} LAPS`
        : `${shortGp} · UPCOMING`;
      $("race-date-chip").textContent = completed
        ? `R${data.round || "--"} · ${data.date || data.season} · Final Grid`
        : `R${data.round || "--"} · Not Completed Yet`;

      if (data.weather) {
        const air = data.weather.air_temperature_c != null ? `${Number(data.weather.air_temperature_c).toFixed(1)}°C` : "--°C";
        const trk = data.weather.track_temperature_c != null ? `${Number(data.weather.track_temperature_c).toFixed(1)}°C` : "--°C";
        $("weather-readout").textContent = `AIR ${air} · TRK ${trk}`;
      } else {
        $("weather-readout").textContent = `${data.season || 2026} · ROUND ${data.round || "--"}`;
      }

      if (data.circuit) {
        const resolved = (data.circuit.track_outline && data.circuit.track_outline.length >= 10)
          ? data.circuit
          : (data.circuit.slug && window.F1_CIRCUITS_2D && window.F1_CIRCUITS_2D[data.circuit.slug]);
        if (resolved) {
          trackRenderer.setCircuit(resolved);
          $("live-circuit-name").textContent = formatCircuitHeading(data.circuit.name || resolved.short_name);
          $("live-circuit-meta").textContent = Array.isArray(resolved.corners) && resolved.corners.length
            ? `${resolved.corners.length}T`
            : "";
        }
      }

      // Past race results: show clean static 2D circuit map (no moving dots, no lock overlay)
      $("gps-lock-overlay").classList.add("hidden");
      trackRenderer.updateDrivers([], false);

      setSectorBests(data.session_best);
      renderDriversStable(data.drivers || [], data.lap_total || 50, data.lap_total || 50, false);
      renderFinalPodiumCards(data);

      // Queue Whisper transcription for recent radio clips
      const clips = (data.team_radio || []).slice(-6);
      for (const clip of clips) {
        const u = clip.recording_url;
        if (!u) continue;
        if (clip.transcript) state.transcripts.set(u, clip.transcript);
        else if (!state.transcripts.has(u) && !state.transcribingUrls.has(u)) requestWhisper(u);
      }
      renderComms(true);
    };

    const applyLivePacket = (pkt) => {
      if (!pkt || pkt.type === "post_quali_prediction" || state.viewMode !== "live") return;
      state.latestLivePacket = pkt;

      // When no live F1 session is actively running on track, show Next Race Schedule + Circuit
      if (pkt.session_active === false) {
        renderIdleLiveSchedule();
        return;
      }

      $("live-idle-panel").classList.add("hidden");
      $("timing-main-panel").classList.remove("hidden");
      $("podium-summary-panel").classList.add("hidden");
      $("live-telemetry-panel").classList.remove("hidden");
      $("comms-section-wrap").classList.remove("hidden");
      $("fastest-lap-banner").classList.remove("hidden");
      $("sector-records-row").classList.remove("hidden");
      if ($("live-badge-text")) $("live-badge-text").textContent = "LIVE TRACK MOVEMENT · 60 FPS";

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
      $("live-stream-meta").textContent = caps.gps_available ? "Live GPS + Timing Active" : "Live Timing Active (GPS Locked)";

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
          $("live-circuit-name").textContent = formatCircuitHeading(pkt.circuit.name || resolved.short_name);
          $("live-circuit-meta").textContent = Array.isArray(resolved.corners) && resolved.corners.length
            ? `${resolved.corners.length}T`
            : "";
        }
      }

      $("gps-lock-overlay").classList.toggle("hidden", Boolean(caps.gps_available));
      $("telem-lock-overlay").classList.toggle("hidden", Boolean(caps.telemetry_available));

      setSectorBests(race.session_best);

      if (drivers.length >= 2) {
        const nums = new Set(drivers.map((d) => Number(d.driver_number)));
        if (!nums.has(state.selectedDriverA)) state.selectedDriverA = Number(drivers[0].driver_number);
        if (!nums.has(state.selectedDriverB) || state.selectedDriverB === state.selectedDriverA) {
          const second = drivers.find((d) => Number(d.driver_number) !== state.selectedDriverA);
          if (second) state.selectedDriverB = Number(second.driver_number);
        }
      }

      renderDriversStable(drivers, lapCur, lapTot, true);

      // Live 60 FPS car movement on 2D circuit map
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

      const clips = race.team_radio || [];
      for (const clip of clips) {
        const u = clip.recording_url;
        if (!u) continue;
        if (clip.transcript) state.transcripts.set(u, clip.transcript);
        else if (!state.transcripts.has(u) && !state.transcribingUrls.has(u)) requestWhisper(u);
      }
      if ($("radio-autoplay-check").checked && clips.length) {
        const newest = clips[clips.length - 1];
        if (newest.recording_url && !state.playedAutoUrls.has(newest.recording_url)) {
          state.playedAutoUrls.add(newest.recording_url);
          playRadioClip(newest.recording_url);
        }
      }
      renderComms(false);
    };

    const disconnectWs = () => {
      clearTimeout(state.wsReconnectTimer);
      if (state.ws) {
        try { state.ws.onclose = null; state.ws.close(); } catch (_) {}
        state.ws = null;
      }
      $("ws-conn-status").textContent = "Standby (Past Race Results)";
    };

    const connectLiveWs = () => {
      disconnectWs();
      const url = toWsUrl(state.apiBase);
      $("ws-conn-status").textContent = `Connecting to ${url}...`;
      $("live-stream-meta").textContent = "Connecting to Live Feed...";
      const ws = new WebSocket(url);
      state.ws = ws;
      ws.onopen = () => {
        $("ws-conn-status").textContent = `Live Connected · ${state.apiBase}`;
        $("live-stream-meta").textContent = "Live Stream Connected";
      };
      ws.onmessage = (evt) => {
        if (state.viewMode !== "live") return;
        let pkt;
        try { pkt = JSON.parse(evt.data); } catch (_) { return; }
        if (!pkt || pkt.type === "pong") return;
        if (state.syncDelayS <= 0) applyLivePacket(pkt);
        else {
          state.delayBuffer.push({ at: Date.now() + state.syncDelayS * 1000, pkt });
          if (state.delayBuffer.length > 400) state.delayBuffer.shift();
        }
      };
      ws.onclose = () => {
        if (state.viewMode !== "live") return;
        $("ws-conn-status").textContent = "Reconnecting...";
        $("live-stream-meta").textContent = "Reconnecting...";
        clearTimeout(state.wsReconnectTimer);
        state.wsReconnectTimer = setTimeout(connectLiveWs, 2500);
      };
    };

    setInterval(() => {
      if (state.viewMode !== "live" || !state.delayBuffer.length) return;
      const now = Date.now();
      let due = null;
      while (state.delayBuffer.length && state.delayBuffer[0].at <= now) due = state.delayBuffer.shift().pkt;
      if (due) applyLivePacket(due);
    }, 200);

    const loadSessionsForYear = async (year, targetRound, targetSk) => {
      const sel = $("session-select");
      try {
        const r = await fetch(`${state.apiBase}/api/sessions?year=${year}`);
        if (!r.ok) return [];
        const d = await r.json();
        const sessions = d.sessions || [];
        state.sessionsCache = sessions;
        sel.innerHTML = sessions.length
          ? sessions.map((s) => {
              const done = s.completed !== false;
              const label = `R${s.round || "?"} · ${s.grand_prix}${done ? "" : " (Not Completed)"}`;
              return `<option value="${s.round}" data-sk="${s.session_key || ""}" ${done ? "" : "disabled"}>${esc(label)}</option>`;
            }).join("")
          : `<option value="">No races found</option>`;

        const completed = sessions.filter((s) => s.completed !== false);
        let chosen = null;
        if (targetRound) {
          chosen = completed.find((s) => String(s.round) === String(targetRound));
        }
        if (!chosen && targetSk) {
          chosen = completed.find((s) => String(s.session_key) === String(targetSk));
        }
        if (!chosen && completed.length) {
          chosen = completed[completed.length - 1];
        }
        if (chosen) sel.value = String(chosen.round);
        return sessions;
      } catch (_) {
        return [];
      }
    };

    const loadFinalResults = async () => {
      const year = Number($("results-year-select").value) || 2026;
      const sel = $("session-select");
      const round = Number(sel.value) || undefined;
      const opt = sel.options[sel.selectedIndex];
      const sk = opt && opt.dataset.sk ? Number(opt.dataset.sk) : undefined;
      if (!round) return;

      $("btn-load-session").textContent = "Loading...";
      try {
        const qs = new URLSearchParams({ season: String(year), round: String(round) });
        if (sk) qs.set("session_key", String(sk));
        const r = await fetch(`${state.apiBase}/api/race-results?${qs.toString()}`);
        if (r.ok) {
          const data = await r.json();
          applyFinalResults(data);
        }
      } finally {
        $("btn-load-session").textContent = "Load Results";
      }
    };

    const setViewMode = async (mode) => {
      state.viewMode = mode === "live" ? "live" : "results";
      const isLive = state.viewMode === "live";

      document.querySelectorAll(".seg-opt[data-view]").forEach((b) => {
        b.classList.toggle("active", b.dataset.view === state.viewMode);
      });
      $("results-selector-group").classList.toggle("hidden", isLive);
      $("results-meta-strip").classList.toggle("hidden", isLive);
      $("live-status-strip").classList.toggle("hidden", !isLive);
      $("podium-summary-panel").classList.toggle("hidden", isLive);
      $("live-telemetry-panel").classList.toggle("hidden", !isLive);

      if (isLive) {
        renderIdleLiveSchedule();
        try {
          await fetch(`${state.apiBase}/api/mode`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode: "free_live" }),
          });
        } catch (_) {}
        connectLiveWs();
      } else {
        disconnectWs();
        $("live-idle-panel").classList.add("hidden");
        $("timing-main-panel").classList.remove("hidden");
        $("podium-summary-panel").classList.remove("hidden");
        $("live-telemetry-panel").classList.add("hidden");
        $("comms-section-wrap").classList.remove("hidden");
        if (state.latestResultsData) {
          applyFinalResults(state.latestResultsData);
        } else {
          await loadFinalResults();
        }
      }
    };

    if ($("btn-switch-to-results")) {
      $("btn-switch-to-results").addEventListener("click", () => setViewMode("results"));
    }

    document.querySelectorAll(".seg-opt[data-view]").forEach((b) => {
      b.addEventListener("click", () => setViewMode(b.dataset.view));
    });
    $("results-year-select").addEventListener("change", (e) => {
      loadSessionsForYear(e.target.value).then(() => loadFinalResults());
    });
    $("session-select").addEventListener("change", () => loadFinalResults());
    $("btn-load-session").addEventListener("click", () => loadFinalResults());

    bindSettingsPopover(() => {
      if (state.viewMode === "live") connectLiveWs();
      else loadSessionsForYear($("results-year-select").value).then(() => loadFinalResults());
    });

    // Parse query params (?view=live|results, ?year=..., ?round=..., ?session_key=..., ?circuit=...)
    const params = new URLSearchParams(window.location.search);
    const qView = params.get("view") === "live" ? "live" : "results";
    const qYear = params.get("year") || "2026";
    const qRound = params.get("round");
    const qSession = params.get("session_key");
    const qCircuit = params.get("circuit");
    $("results-year-select").value = qYear;

    if (qCircuit && window.F1_CIRCUITS_2D && window.F1_CIRCUITS_2D[qCircuit]) {
      const c = window.F1_CIRCUITS_2D[qCircuit];
      trackRenderer.setCircuit(c);
      $("live-circuit-name").textContent = formatCircuitHeading(c.short_name || c.slug);
      $("live-circuit-meta").textContent = Array.isArray(c.corners) && c.corners.length ? `${c.corners.length}T` : "";
    }

    loadSessionsForYear(qYear, qRound, qSession).then((sessions) => {
      // If a circuit was clicked from Page 1, match the session for that circuit if completed
      if (qCircuit && !qRound && !qSession && sessions.length) {
        const match = sessions.find((s) => s.completed !== false && s.circuit_slug === qCircuit);
        if (match) $("session-select").value = String(match.round);
      }
      if (qView === "live") {
        setViewMode("live");
      } else {
        loadFinalResults();
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
        $("pred-circuit-heading").textContent = formatCircuitHeading(circ.name || c2d.short_name);
        const turnsTxt = Array.isArray(c2d.corners) && c2d.corners.length ? ` · ${c2d.corners.length} Turns` : "";
        $("pred-circuit-caption").textContent = `${pkt.grand_prix || ""}${turnsTxt} · S1 / S2 / S3 Split`;
        predTrackRenderer.setCircuit(c2d);
      }
    };

    const loadRounds = async (season) => {
      const sel = $("pred-round-select");
      try {
        const r = await fetch(`${state.apiBase}/api/sessions?year=${season}`);
        if (r.ok) {
          const d = await r.json();
          const sessions = d.sessions || [];
          if (sessions.length) {
            sel.innerHTML = sessions.map((s) => {
              const done = s.completed !== false;
              const label = `Round ${s.round || 1} — ${s.grand_prix}${done ? "" : " (Not Completed)"}`;
              return `<option value="${s.round || 1}" ${done ? "" : "disabled"}>${esc(label)}</option>`;
            }).join("");
            const completed = sessions.filter((s) => s.completed !== false);
            const latest = completed.length ? completed[completed.length - 1] : sessions[sessions.length - 1];
            sel.value = String(latest.round || 1);
            return;
          }
        }
      } catch (_) {}
      sel.innerHTML = `<option value="1">Latest Completed Qualifying</option>`;
    };

    let forecastReqId = 0;
    let progressTimer = null;

    const startHourglassOverlay = (targetDurationMs) => {
      clearInterval(progressTimer);
      const overlay = $("pred-loading-overlay");
      const stepEl = $("pred-loading-step");
      const barEl = $("pred-loading-bar-fill");
      const stageEl = $("pred-loading-stage-tag");
      const elapsedEl = $("pred-loading-elapsed");
      if (!overlay) return;

      overlay.classList.remove("hidden");
      const t0 = performance.now();
      const update = () => {
        const elapsedMs = performance.now() - t0;
        const pct = Math.min(96, Math.max(6, (elapsedMs / Math.max(1000, targetDurationMs)) * 96));
        if (barEl) barEl.style.width = `${pct.toFixed(1)}%`;
        if (elapsedEl) elapsedEl.textContent = `${(elapsedMs / 1000).toFixed(1)}s`;

        if (pct < 34) {
          if (stageEl) stageEl.textContent = "STAGE 1 / 3";
          if (stepEl) stepEl.textContent = "Loading 2018–2026 qualifying & starting grid data...";
        } else if (pct < 68) {
          if (stageEl) stageEl.textContent = "STAGE 2 / 3";
          if (stepEl) stepEl.textContent = "Evaluating driver, constructor & circuit history...";
        } else {
          if (stageEl) stageEl.textContent = "STAGE 3 / 3";
          if (stepEl) stepEl.textContent = "Computing Top-3 podium & race-win probabilities...";
        }
      };
      update();
      progressTimer = setInterval(update, 100);
    };

    const stopHourglassOverlay = () => {
      clearInterval(progressTimer);
      progressTimer = null;
      const barEl = $("pred-loading-bar-fill");
      if (barEl) barEl.style.width = "100%";
      const overlay = $("pred-loading-overlay");
      if (overlay) overlay.classList.add("hidden");
    };

    const runForecast = async (minDurationMs = 11000) => {
      const reqId = ++forecastReqId;
      const season = Number($("pred-season-select").value) || 2026;
      const round = Number($("pred-round-select").value) || undefined;
      const btn = $("btn-run-post-quali");
      btn.textContent = "Computing Forecast...";
      btn.disabled = true;
      startHourglassOverlay(minDurationMs);
      const startedAt = performance.now();

      try {
        const fetchPromise = fetch(`${state.apiBase}/api/predict/post-quali`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ season, round }),
        });
        const [r] = await Promise.all([
          fetchPromise,
          new Promise((resolve) => setTimeout(resolve, minDurationMs)),
        ]);
        if (reqId !== forecastReqId) return;
        if (r.ok) {
          renderPostQuali(await r.json());
        } else {
          const err = await r.json().catch(() => ({}));
          $("pred-table-list").innerHTML = `<div class="empty-state">${esc(err.detail || "Qualifying data not available for this round yet.")}</div>`;
        }
      } finally {
        if (reqId === forecastReqId) {
          const took = performance.now() - startedAt;
          if ($("pred-loading-elapsed")) $("pred-loading-elapsed").textContent = `${(took / 1000).toFixed(1)}s`;
          stopHourglassOverlay();
          btn.textContent = "Run Post-Quali Forecast";
          btn.disabled = false;
        }
      }
    };

    $("pred-season-select").addEventListener("change", (e) => loadRounds(e.target.value).then(() => runForecast(11000)));
    $("pred-round-select").addEventListener("change", () => runForecast(11000));
    $("btn-run-post-quali").addEventListener("click", () => runForecast(11000));
    bindSettingsPopover(() => loadRounds($("pred-season-select").value).then(() => runForecast(1800)));
    loadRounds($("pred-season-select").value).then(() => runForecast(1800));
  }

  document.addEventListener("DOMContentLoaded", () => {
    if ("serviceWorker" in navigator && /^https?:$/i.test(window.location.protocol)) {
      navigator.serviceWorker.register("sw.js").catch(() => {});
    }
    bindWindowControls();
    if (state.page === "live") initLivePage();
    else if (state.page === "predictions") initPredictionsPage();
    else initStandingsPage();
  });
})();
