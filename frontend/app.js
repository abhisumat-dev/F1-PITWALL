/**
 * F1 Pit-Wall — Main Application Controller
 *
 * Shared across the Responsive Web PWA (mounted at `/`) and Chrome Side Panel
 * Extension (`extension/sidepanel.html`) with zero external CDN dependencies.
 */
(function () {
  "use strict";

  const STORAGE_KEY_HOST = "f1_pitwall_api_host";
  const STORAGE_KEY_DELAY = "f1_pitwall_sync_delay_s";
  const STORAGE_KEY_SCREEN = "f1_pitwall_active_screen";

  const COMPOUND_META = {
    SOFT: { letter: "S", color: "#EF4444", text: "#FFFFFF" },
    MEDIUM: { letter: "M", color: "#FACC15", text: "#07090D" },
    HARD: { letter: "H", color: "#E2E8F0", text: "#07090D" },
    INTERMEDIATE: { letter: "I", color: "#10B981", text: "#07090D" },
    WET: { letter: "W", color: "#38BDF8", text: "#07090D" },
    UNKNOWN: { letter: "?", color: "#64748B", text: "#FFFFFF" },
  };

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

  function fmtGap(d) {
    if (d.retired) return { text: "DNF", cls: "is-out" };
    if (d.in_pit) return { text: "PIT", cls: "is-pit" };
    if (d.position === 1) return { text: "LEADER", cls: "" };
    if (d.laps_down && d.laps_down > 0) {
      return { text: `+${d.laps_down}L`, cls: "" };
    }
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

  /* ------------------------------------------------------------------------
     Horizontal F1 Tyre Stint Timeline SVG Builder (User's Image 2 Style)
     ------------------------------------------------------------------------ */
  function buildStintTimelineSvg(driver, lapCurrent, lapTotal) {
    const total = Math.max(10, Number(lapTotal) || 50);
    const curLap = Math.max(1, Math.min(total, Number(lapCurrent) || 1));
    const W = 210;
    const H = 22;
    const pxPerLap = (W - 14) / total;

    // Subtle vertical lap grid every 10 laps
    let grid = "";
    const step = total >= 60 ? 10 : 10;
    for (let l = step; l <= total; l += step) {
      const gx = (6 + l * pxPerLap).toFixed(1);
      grid += `<line x1="${gx}" y1="1" x2="${gx}" y2="${H - 1}" stroke="rgba(148,163,184,0.14)" stroke-width="0.9"/>`;
    }

    let stints = Array.isArray(driver.stints) && driver.stints.length
      ? driver.stints
      : [{
          stint_number: 1,
          compound: (driver.tyre && driver.tyre.compound) || "UNKNOWN",
          lap_start: 1,
          lap_end: curLap,
          tyre_age_at_start: 0,
        }];

    let shapes = "";
    const isOut = Boolean(driver.retired);
    const op = isOut ? 0.45 : 0.94;

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
      // Never project future OpenF1 stint end laps beyond the current race lap
      eLap = Math.max(sLap, Math.min(curLap, total, eLap));

      const x1 = 7 + (sLap - 1) * pxPerLap;
      const x2 = Math.max(x1 + 14, 7 + eLap * pxPerLap);
      const isLast = i === stints.length - 1;
      const rawAge = isLast && driver.tyre && driver.tyre.age_laps != null
        ? Number(driver.tyre.age_laps)
        : eLap - sLap + 1 + (Number(st.tyre_age_at_start) || 0);
      const age = Math.max(1, rawAge);

      shapes += `<line x1="${x1.toFixed(1)}" y1="11" x2="${x2.toFixed(1)}" y2="11" stroke="${meta.color}" stroke-width="3.2" stroke-linecap="round" opacity="${op}"/>`;
      // Start-of-stint compound circle
      shapes += `<circle cx="${x1.toFixed(1)}" cy="11" r="6" fill="#07090D" stroke="${meta.color}" stroke-width="1.6"/>`;
      shapes += `<text x="${x1.toFixed(1)}" y="13.5" fill="${meta.color}" font-family="JetBrains Mono, monospace" font-size="7" font-weight="700" text-anchor="middle">${meta.letter}</text>`;

      // Current-lap trailing circle showing stint tyre age
      if (isLast) {
        const cx = Math.min(W - 8, x2);
        shapes += `<circle cx="${cx.toFixed(1)}" cy="11" r="7" fill="${meta.color}"/>`;
        shapes += `<text x="${cx.toFixed(1)}" y="13.6" fill="${meta.text}" font-family="JetBrains Mono, monospace" font-size="7" font-weight="700" text-anchor="middle">${Math.min(99, Math.max(0, Math.round(age)))}</text>`;
      }
    }

    return `<svg viewBox="0 0 ${W} ${H}" fill="none" aria-hidden="true">${grid}${shapes}</svg>`;
  }

  /* ------------------------------------------------------------------------
     Built-in Offline / Initial Fallback Demo Data (22 Drivers on 2026 Grid)
     Replaced immediately as live REST & WS /ws/live packets arrive.
     ------------------------------------------------------------------------ */
  function create22GridFixture(variant) {
    const isWorst = variant === "worst";
    const roster = [
      [1, "VER", isWorst ? "Aleksandra Wiśniewska-Kowalczyk-Verstappen" : "Max Verstappen", "Red Bull Racing", "3671C6", [["MEDIUM", 1, 16], ["HARD", 16, 32]], 94.2, 46.8],
      [4, "NOR", isWorst ? "Jo" : "Lando Norris", "McLaren", "FF8000", [["MEDIUM", 1, 15], ["HARD", 15, 32]], 86.5, 31.4],
      [16, "LEC", "Charles Leclerc", "Ferrari", "E80020", [["MEDIUM", 1, 14], ["HARD", 14, 32]], 74.1, 15.2],
      [81, "PIA", "Oscar Piastri", "McLaren", "FF8000", [["MEDIUM", 1, 16], ["HARD", 16, 32]], 24.8, 4.1],
      [63, "RUS", "George Russell", "Mercedes", "27F4D2", [["HARD", 1, 22], ["MEDIUM", 22, 32]], 10.6, 1.4],
      [44, "HAM", "Lewis Hamilton", "Ferrari", "E80020", [["MEDIUM", 1, 17], ["HARD", 17, 32]], 5.2, 0.7],
      [12, "ANT", "Kimi Antonelli", "Mercedes", "27F4D2", [["SOFT", 1, 11], ["HARD", 11, 32]], 2.4, 0.2],
      [55, "SAI", "Carlos Sainz", "Williams", "1868DB", [["MEDIUM", 1, 16], ["HARD", 16, 32]], 1.1, 0.1],
      [14, "ALO", "Fernando Alonso", "Aston Martin", "229971", [["HARD", 1, 20], ["MEDIUM", 20, 32]], 0.7, 0.1],
      [10, "GAS", "Pierre Gasly", "Alpine", "00A1E8", [["MEDIUM", 1, 15], ["HARD", 15, 31], ["SOFT", 31, 32]], 0.4, 0.0],
      [23, "ALB", "Alexander Albon", "Williams", "1868DB", [["MEDIUM", 1, 18], ["HARD", 18, 32]], 0.0, 0.0],
      [18, "STR", "Lance Stroll", "Aston Martin", "229971", [["HARD", 1, 21], ["MEDIUM", 21, 32]], 0.0, 0.0],
      [30, "LAW", "Liam Lawson", "Racing Bulls", "6C98FF", [["MEDIUM", 1, 16], ["HARD", 16, 32]], 0.0, 0.0],
      [22, "TSU", "Yuki Tsunoda", "Racing Bulls", "6C98FF", [["SOFT", 1, 10], ["HARD", 10, 32]], 0.0, 0.0],
      [27, "HUL", "Nico Hulkenberg", "Audi", "F50537", [["HARD", 1, 19], ["MEDIUM", 19, 32]], 0.0, 0.0],
      [5, "BOR", "Gabriel Bortoleto", "Audi", "F50537", [["MEDIUM", 1, 15], ["HARD", 15, 32]], 0.0, 0.0],
      [87, "BEA", "Oliver Bearman", "Haas F1 Team", "9C9FA2", [["SOFT", 1, 12], ["HARD", 12, 32]], 0.0, 0.0],
      [31, "OCO", "Esteban Ocon", "Haas F1 Team", "9C9FA2", [["HARD", 1, 23], ["MEDIUM", 23, 32]], 0.0, 0.0],
      [7, "DOO", "Jack Doohan", "Alpine", "00A1E8", [["MEDIUM", 1, 14], ["HARD", 14, 32]], 0.0, 0.0],
      [6, "HAD", "Isack Hadjar", "Red Bull Racing", "3671C6", [["HARD", 1, 18], ["MEDIUM", 18, 32]], 0.0, 0.0],
      [77, "BOT", "Valtteri Bottas", "Cadillac F1", "909090", [["MEDIUM", 1, 13], ["HARD", 13, 31]], 0.0, 0.0],
      [11, "PER", "Sergio Perez", "Cadillac F1", "909090", [["SOFT", 1, 8]], 0.0, 0.0],
    ];

    const vegas = window.F1_CIRCUITS_2D && window.F1_CIRCUITS_2D.las_vegas;
    const pts = vegas ? vegas.points : [];
    const freeLock = variant === "free_live";

    const drivers = roster.map((r, idx) => {
      const [num, code, full, team, col, stintsRaw, pod, win] = r;
      const isOut = idx === 21;
      const isPit = idx === 9;
      const pt = pts.length ? pts[(pts.length - 12 - idx * 8 + pts.length * 4) % pts.length] : { x: 0, y: 0 };
      const stints = isWorst && idx === 0
        ? [
            { stint_number: 1, compound: "SOFT", lap_start: 1, lap_end: 8, tyre_age_at_start: 0 },
            { stint_number: 2, compound: "MEDIUM", lap_start: 9, lap_end: 17, tyre_age_at_start: 0 },
            { stint_number: 3, compound: "INTERMEDIATE", lap_start: 18, lap_end: 23, tyre_age_at_start: 0 },
            { stint_number: 4, compound: "WET", lap_start: 24, lap_end: 27, tyre_age_at_start: 0 },
            { stint_number: 5, compound: "HARD", lap_start: 28, lap_end: 32, tyre_age_at_start: 0 },
          ]
        : stintsRaw.map((s, si) => ({
            stint_number: si + 1,
            compound: s[0],
            lap_start: s[1],
            lap_end: si === stintsRaw.length - 1 ? null : s[2],
            tyre_age_at_start: 0,
          }));
      const lastSt = stints[stints.length - 1];
      return {
        driver_number: num,
        code,
        full_name: full,
        team_name: isWorst && idx === 0 ? "Oracle Red Bull Racing Honda RBPT Powertrains International" : team,
        team_colour: col,
        position: isOut ? null : idx + 1,
        grid_position: idx + 1,
        gap_to_leader_s: isOut ? null : idx === 0 ? 0 : +(idx * 1.41 + 0.32).toFixed(3),
        interval_s: isOut ? null : idx === 0 ? 0 : 1.412,
        laps_down: idx === 20 ? 1 : 0,
        last_lap_time_s: 92.008 + idx * 0.14,
        best_lap_time_s: 92.008 + idx * 0.11,
        tyre: {
          compound: lastSt.compound,
          age_laps: Math.max(1, 32 - lastSt.lap_start + 1),
          stint_number: stints.length,
          is_new: true,
        },
        stints,
        pit_stops: [],
        in_pit: isPit,
        retired: isOut,
        status: isOut ? "DNF" : isPit ? "PIT" : "RUNNING",
        x: freeLock ? null : pt.x,
        y: freeLock ? null : pt.y,
        sectors: {
          s1_s: 26.812 + idx * 0.08,
          s2_s: 31.044 + idx * 0.09,
          s3_s: 33.918 + idx * 0.07,
          s1_status: idx === 0 ? "PURPLE" : idx % 2 === 0 ? "GREEN" : "YELLOW",
          s2_status: idx === 1 ? "PURPLE" : idx % 3 === 0 ? "GREEN" : "YELLOW",
          s3_status: idx === 2 ? "PURPLE" : "GREEN",
        },
        podium_prob_pct: pod,
        win_prob_pct: win,
        telemetry: freeLock ? null : {
          speed_kph: 342 - idx * 1.6,
          throttle_pct: 99,
          brake_pct: 0,
          gear: 8,
          rpm: 11850,
          drs: null,
          drs_active: null,
        },
      };
    });

    const raceControl = variant === "empty" ? [] : [
      {
        date: "2026-11-22T04:47:55.000Z",
        lap_number: 31,
        category: "Other",
        flag: "PURPLE",
        message: "FASTEST LAP CAR 1 (VER) — 1:32.008 ON LAP 31",
      },
      {
        date: "2026-11-22T04:44:02.000Z",
        lap_number: 29,
        category: "Stewards",
        flag: "YELLOW",
        message: isWorst
          ? "CAR 16 (LEC) LAP TIME DELETED — TRACK LIMITS AT TURN 7 LAP 28 — STEWARDS NOTE REFERENCE https://fia.example.com/stewards/2026/las-vegas/decision-document-049-exceeding-track-limits-repeatedly"
          : "CAR 16 (LEC) TIME DELETED — TRACK LIMITS AT TURN 7 LAP 28",
      },
      {
        date: "2026-11-22T04:36:10.000Z",
        lap_number: 24,
        category: "Flag",
        flag: "GREEN",
        message: "TRACK CLEAR",
      },
    ];

    const teamRadio = variant === "empty" ? [] : [
      {
        date: "2026-11-22T04:48:12.000Z",
        driver_number: 1,
        driver_code: "VER",
        team_name: "Red Bull Racing",
        team_colour: "3671C6",
        recording_url: "https://livetiming.formula1.com/static/2026/demo_ver.mp3",
        transcript: "Tyres feeling good in Sector 1, keep me posted on Lando's pace.",
      },
      {
        date: "2026-11-22T04:46:19.000Z",
        driver_number: 4,
        driver_code: "NOR",
        team_name: "McLaren",
        team_colour: "FF8000",
        recording_url: "https://livetiming.formula1.com/static/2026/demo_nor.mp3",
        transcript: "Front-left starting to grain slightly into Turn 12 braking.",
      },
      {
        date: "2026-11-22T04:42:40.000Z",
        driver_number: 16,
        driver_code: "LEC",
        team_name: "Ferrari",
        team_colour: "E80020",
        recording_url: "https://livetiming.formula1.com/static/2026/demo_lec.mp3",
        transcript: "Plan B plus two laps, push now in Sector 3.",
      },
    ];

    return {
      schema_version: "1.0",
      type: freeLock ? "live_race_update" : "replay_frame",
      mode: freeLock ? "free_live" : "replay",
      grand_prix: "Las Vegas Grand Prix",
      season: 2026,
      round: 22,
      capabilities: {
        gps_available: !freeLock,
        telemetry_available: !freeLock,
      },
      circuit: vegas ? {
        slug: vegas.slug,
        name: "Las Vegas Strip Circuit",
        rotation_deg: vegas.rotation_deg,
        track_outline: vegas.points,
        sector_splits: vegas.sector_splits,
        start_finish: vegas.start_finish,
        corners: vegas.corners,
      } : { name: "Las Vegas Strip Circuit", rotation_deg: 90 },
      replay: { speed: 4.0, paused: false, finished: false },
      race: {
        lap_current: variant === "empty" ? 0 : 32,
        lap_total: 50,
        track_status: "GREEN",
        weather: { air_temperature_c: 17.2, track_temperature_c: 21.5 },
        drivers: variant === "empty" ? [] : drivers,
        race_control: raceControl,
        team_radio: teamRadio,
        session_best: variant === "empty" ? null : {
          fastest_lap: { driver_number: 1, driver_code: "VER", team_name: "Red Bull Racing", time_s: 92.008, formatted: "1:32.008", lap_number: 31 },
          fastest_s1: { driver_number: 1, driver_code: "VER", team_name: "Red Bull Racing", time_s: 26.812, formatted: "26.812", lap_number: 31 },
          fastest_s2: { driver_number: 4, driver_code: "NOR", team_name: "McLaren", time_s: 31.044, formatted: "31.044", lap_number: 30 },
          fastest_s3: { driver_number: 16, driver_code: "LEC", team_name: "Ferrari", time_s: 33.918, formatted: "33.918", lap_number: 29 },
        },
      },
    };
  }

  /* ------------------------------------------------------------------------
     Application State
     ------------------------------------------------------------------------ */
  const state = {
    apiBase: resolveApiBase(),
    activeScreen: localStorage.getItem(STORAGE_KEY_SCREEN) || "standings",
    mode: "replay",
    syncDelayS: Number(localStorage.getItem(STORAGE_KEY_DELAY) || 0),
    delayBuffer: [],
    latestPacket: null,
    postQualiPacket: null,
    selectedDriverA: 1,
    selectedDriverB: 4,
    commsFilter: "all",
    drawerFilter: "all",
    activeCircuitSlug: "las_vegas",
    sessions: [],
    transcripts: new Map(),
    transcribingUrls: new Set(),
    playedAutoUrls: new Set(),
    currentlyPlayingUrl: null,
    ws: null,
    wsReconnectTimer: null,
    fixtureOverride: null,
  };

  let liveTrackRenderer = null;
  let predTrackRenderer = null;
  let telemetryRenderer = null;

  const $ = (id) => document.getElementById(id);

  /* ------------------------------------------------------------------------
     Navigation & Drawers
     ------------------------------------------------------------------------ */
  function setScreen(screenId) {
    if (!["standings", "live", "predictions"].includes(screenId)) return;
    state.activeScreen = screenId;
    localStorage.setItem(STORAGE_KEY_SCREEN, screenId);

    document.querySelectorAll(".tab-btn").forEach((btn) => {
      const act = btn.dataset.screen === screenId;
      btn.setAttribute("aria-selected", act ? "true" : "false");
    });
    document.querySelectorAll(".screen").forEach((sec) => {
      const act = sec.id === `screen-${screenId}`;
      sec.classList.toggle("active", act);
      sec.hidden = !act;
      sec.style.display = act ? "flex" : "none";
    });
    window.scrollTo({ top: 0, behavior: "instant" });
    document.querySelectorAll(".hub-card").forEach((card) => {
      card.classList.toggle("is-active", card.dataset.goto === screenId);
    });

    // Resize canvases once visible
    requestAnimationFrame(() => {
      if (screenId === "live") {
        if (liveTrackRenderer) liveTrackRenderer._resize();
        if (telemetryRenderer) telemetryRenderer._resize();
      } else if (screenId === "predictions") {
        if (predTrackRenderer) predTrackRenderer._resize();
      }
    });
  }

  function openCommsDrawer() {
    const bd = $("comms-backdrop");
    const dr = $("comms-drawer");
    bd.classList.remove("hidden");
    dr.classList.remove("hidden");
    requestAnimationFrame(() => {
      bd.classList.add("open");
      dr.classList.add("open");
    });
    renderCommsLists();
  }

  function closeCommsDrawer() {
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
  }

  /* ------------------------------------------------------------------------
     Screen 1 Renderers: WDC, WCC, Multi-Timezone Next Race, 24 Circuits
     ------------------------------------------------------------------------ */
  function renderStandings(data) {
    const wdcList = $("wdc-list");
    const wccList = $("wcc-list");
    if (!wdcList || !wccList) return;

    const drivers = (data && data.drivers) || [];
    const constructors = (data && data.constructors) || [];
    $("wdc-meta").textContent = data && data.after_round
      ? `${data.season} · After R${data.after_round}`
      : `${(data && data.season) || 2026}`;

    if (!drivers.length) {
      wdcList.innerHTML = `<div class="empty-state">No championship standings loaded yet.</div>`;
    } else {
      const maxPts = Math.max(1, Number(drivers[0].points) || 1);
      wdcList.innerHTML = drivers.map((d, i) => {
        const col = hexCol(d.team_colour);
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
                <span class="drv-team" title="${esc(d.team_name)}">${esc(d.team_name)}</span>
              </div>
            </div>
            <div class="pts-bar-track"><div class="pts-bar-fill" style="width:${pct}%;background:${col}"></div></div>
            <span class="mono-muted align-right">${Number(d.wins || 0)}W</span>
            <span class="mono-val align-right" style="font-weight:700">${Math.round(Number(d.points || 0))}</span>
          </div>`;
      }).join("");
    }

    if (!constructors.length) {
      wccList.innerHTML = `<div class="empty-state">No constructor standings available.</div>`;
    } else {
      wccList.innerHTML = constructors.map((c, i) => {
        const col = hexCol(c.team_colour);
        return `
          <div class="wcc-row">
            <span class="wdc-pos">${String(c.position || i + 1).padStart(2, "0")}</span>
            <span class="team-swatch" style="background:${col}"></span>
            <span class="wcc-name" title="${esc(c.team_name)}">${esc(c.team_name)}</span>
            <span class="mono-muted align-right">${Number(c.wins || 0)}W</span>
            <span class="mono-val align-right" style="font-weight:700">${Math.round(Number(c.points || 0))} PTS</span>
          </div>`;
      }).join("");
    }
  }

  function renderNextRace(data) {
    const wrap = $("next-race-sessions");
    if (!wrap || !data) return;
    $("next-race-title").textContent = `Next Race · ${data.grand_prix || "Grand Prix"}`;
    const cd = $("next-race-countdown");
    if (data.countdown && data.countdown.label) {
      cd.textContent = data.countdown.label;
    } else if (data.status === "season_complete") {
      cd.textContent = "SEASON COMPLETE";
    } else {
      cd.textContent = `ROUND ${data.round || "--"}`;
    }

    if (!state.latestPacket) {
      $("session-pill-text").textContent = `ROUND ${data.round || "--"} · ${(data.grand_prix || "").toUpperCase()}`;
    }

    const sessions = Array.isArray(data.sessions) ? data.sessions : [];
    if (!sessions.length) {
      wrap.innerHTML = `<div class="empty-state">Upcoming session schedule not published yet.</div>`;
      return;
    }
    wrap.innerHTML = sessions.map((s) => {
      const t = s.times || {};
      const isMain = String(s.name || "").toLowerCase().includes("grand prix");
      const istStr = t.ist ? `${t.ist.day_short || ""} ${t.ist.time_24h || "--"}`.trim() : "--";
      const ukStr = t.uk ? t.uk.time_24h || "--" : "--";
      const usEt = t.us_eastern ? t.us_eastern.time_24h || "--" : "--";
      const usPt = t.us_pacific ? t.us_pacific.time_24h || "--" : "--";
      return `
        <div class="tz-row ${isMain ? "is-main" : ""}">
          <span class="tz-session" title="${esc(s.name)}">${esc(s.name)}</span>
          <span class="align-right tz-ist">${esc(istStr)}</span>
          <span class="align-right tz-uk">${esc(ukStr)}</span>
          <span class="align-right tz-us">${esc(usEt)} / ${esc(usPt)}</span>
        </div>`;
    }).join("");
  }

  function renderCircuitsCatalog(query) {
    const container = $("circuits-catalog");
    if (!container) return;
    const catalog = window.F1_CIRCUITS_2D || {};
    const q = String(query || "").trim().toLowerCase();
    const list = Object.values(catalog)
      .filter((c) => {
        if (!q) return true;
        const hay = `${c.short_name || ""} ${c.meeting_name || ""} ${c.country_name || ""} ${c.location || ""}`.toLowerCase();
        return hay.includes(q);
      })
      .sort((a, b) => {
        if (a.slug === "las_vegas") return -1;
        if (b.slug === "las_vegas") return 1;
        return String(a.short_name || "").localeCompare(String(b.short_name || ""));
      });

    if (!list.length) {
      container.innerHTML = `<div class="empty-state">No matching circuit found.</div>`;
      return;
    }
    container.innerHTML = list.map((c) => {
      const act = c.slug === state.activeCircuitSlug;
      const turns = Array.isArray(c.corners) ? c.corners.length : 0;
      const svg = window.buildCircuitThumbnailSvg ? window.buildCircuitThumbnailSvg(c) : "";
      return `
        <button type="button" class="circuit-item ${act ? "is-active" : ""}" data-circuit-slug="${esc(c.slug)}">
          <div class="circuit-item-info">
            <span class="circuit-item-name">${esc(c.short_name || c.location || c.slug)}</span>
            <span class="circuit-item-meta">${esc(c.meeting_name || c.country_name || "")} · ${turns}T</span>
          </div>
          <div class="circuit-item-svg">${svg}</div>
        </button>`;
    }).join("");

    container.querySelectorAll(".circuit-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        const slug = btn.dataset.circuitSlug;
        selectCircuitPreview(slug);
      });
    });
  }

  function selectCircuitPreview(slug) {
    const c = window.F1_CIRCUITS_2D && window.F1_CIRCUITS_2D[slug];
    if (!c) return;
    state.activeCircuitSlug = slug;
    renderCircuitsCatalog($("circuit-search").value);
    if (liveTrackRenderer) liveTrackRenderer.setCircuit(c);
    if (predTrackRenderer) predTrackRenderer.setCircuit(c);
    $("live-circuit-name").textContent = c.meeting_name ? `${c.short_name} Circuit` : c.short_name;
    $("live-circuit-meta").textContent = `${(c.corners || []).length}T`;
    $("pred-circuit-heading").textContent = `${c.short_name} Circuit`;
    // If a matching OpenF1 session is loaded in sessions list, offer instant replay load
    const match = state.sessions.find((s) =>
      String(s.circuit_name || "").toLowerCase().includes(String(c.short_name || "").toLowerCase()) ||
      String(s.grand_prix || "").toLowerCase().includes(String(c.short_name || "").toLowerCase())
    );
    if (match && match.session_key) {
      loadReplaySession(match.session_key);
    }
    setScreen("live");
  }

  /* ------------------------------------------------------------------------
     Screen 2 Renderer: 22-Driver Timing + Tyre Stint Timeline + Circuit + Telemetry
     ------------------------------------------------------------------------ */
  function renderStintRuler(lapTotal) {
    const ruler = $("stint-ruler");
    if (!ruler) return;
    const total = Math.max(10, Number(lapTotal) || 50);
    const ticks = [];
    for (let l = 10; l <= total; l += 10) {
      const pct = ((l / total) * 93 + 3).toFixed(1);
      ticks.push(`<span class="stint-ruler-tick" style="left:${pct}%">${l}</span>`);
    }
    ruler.innerHTML = ticks.join("");
  }

  function applyLivePacket(pkt) {
    if (!pkt) return;
    if (pkt.type === "post_quali_prediction") {
      state.postQualiPacket = pkt;
      renderPostQuali(pkt);
      return;
    }

    state.latestPacket = pkt;
    const race = pkt.race || pkt.tick || {};
    const drivers = Array.isArray(race.drivers) ? race.drivers : [];
    const lapCur = race.lap_current != null ? race.lap_current : 0;
    const lapTot = race.lap_total != null ? race.lap_total : 50;
    const caps = pkt.capabilities || { gps_available: false, telemetry_available: false };

    // Top command bar flag & session context
    const flagPill = $("flag-pill");
    const status = String(race.track_status || "GREEN").toUpperCase();
    flagPill.className = `status-pill flag-${status.toLowerCase()}`;
    $("flag-pill-text").textContent = status === "SC" ? "SAFETY CAR" : status === "VSC" ? "VIRTUAL SC" : `${status} FLAG`;
    const shortGp = String(pkt.grand_prix || "Live Session").replace(/\s*Grand Prix$/i, " GP").toUpperCase();
    $("session-pill-text").textContent = `${shortGp} · L${lapCur}/${lapTot}`;

    // Weather
    const w = race.weather;
    if (w) {
      const air = w.air_temperature_c != null ? `${Number(w.air_temperature_c).toFixed(1)}°C` : "--°C";
      const trk = w.track_temperature_c != null ? `${Number(w.track_temperature_c).toFixed(1)}°C` : "--°C";
      $("weather-readout").textContent = `AIR ${air} · TRK ${trk}`;
    }

    // Circuit geometry
    if (pkt.circuit) {
      const resolved = (pkt.circuit.track_outline && pkt.circuit.track_outline.length >= 10)
        ? pkt.circuit
        : (pkt.circuit.slug && window.F1_CIRCUITS_2D && window.F1_CIRCUITS_2D[pkt.circuit.slug]);
      if (resolved) {
        if (liveTrackRenderer) liveTrackRenderer.setCircuit(resolved);
        const turnCnt = Array.isArray(resolved.corners) ? resolved.corners.length : 0;
        $("live-circuit-name").textContent = pkt.circuit.name || resolved.short_name || "Circuit";
        $("live-circuit-meta").textContent = turnCnt ? `${turnCnt}T` : "";
      }
    }

    // Lock overlays in free_live mode
    $("gps-lock-overlay").classList.toggle("hidden", Boolean(caps.gps_available));
    $("telem-lock-overlay").classList.toggle("hidden", Boolean(caps.telemetry_available));

    // Session Best: Fastest Lap & Fastest S1 / S2 / S3
    const sb = race.session_best || {};
    if (sb.fastest_lap) {
      const fl = sb.fastest_lap;
      $("fastest-lap-val").textContent = `${fl.formatted || "--"} · ${fl.driver_code}${fl.lap_number ? ` (L${fl.lap_number})` : ""}`;
    } else {
      $("fastest-lap-val").textContent = "--:--.---";
    }
    const setSec = (idPrefix, rec) => {
      $(`${idPrefix}-time`).textContent = rec && rec.formatted ? `${rec.formatted}s` : "--.---s";
      $(`${idPrefix}-who`).textContent = rec ? `${rec.driver_code}${rec.team_name ? ` · ${rec.team_name}` : ""}` : "--";
    };
    setSec("s1-best", sb.fastest_s1);
    setSec("s2-best", sb.fastest_s2);
    setSec("s3-best", sb.fastest_s3);

    // Ensure selected drivers exist in grid
    if (drivers.length >= 2) {
      const nums = new Set(drivers.map((d) => Number(d.driver_number)));
      if (!nums.has(state.selectedDriverA)) state.selectedDriverA = Number(drivers[0].driver_number);
      if (!nums.has(state.selectedDriverB) || state.selectedDriverB === state.selectedDriverA) {
        const second = drivers.find((d) => Number(d.driver_number) !== state.selectedDriverA);
        if (second) state.selectedDriverB = Number(second.driver_number);
      }
    }

    // Render Timing Tower + 22-Driver Tyre Stint Timeline
    renderStintRuler(lapTot);
    const tList = $("timing-list");
    if (!drivers.length) {
      tList.innerHTML = `<div class="empty-state">Waiting for live timing data...</div>`;
    } else {
      tList.innerHTML = drivers.map((d) => {
        const dn = Number(d.driver_number);
        const col = hexCol(d.team_colour);
        const gap = fmtGap(d);
        const secs = d.sectors || {};
        const selCls = dn === state.selectedDriverA ? "selected-a" : dn === state.selectedDriverB ? "selected-b" : "";
        const stintSvg = buildStintTimelineSvg(d, lapCur, lapTot);
        return `
          <button type="button" class="timing-row ${selCls} ${d.retired ? "is-retired" : ""}" data-dn="${dn}" role="listitem" title="Click to compare telemetry (${esc(d.full_name)})">
            <span class="t-pos ${d.retired ? "out" : ""}">${d.retired ? "OUT" : String(d.position || "--").padStart(2, "0")}</span>
            <span class="t-drv">
              <span class="t-stripe" style="background:${col}"></span>
              <span class="t-code">${esc(d.code)}</span>
            </span>
            <span class="t-gap ${gap.cls}">${esc(gap.text)}</span>
            <span class="t-sectors" aria-label="Sector status">
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
          if (liveTrackRenderer) liveTrackRenderer.setSelectedDrivers([state.selectedDriverA, state.selectedDriverB]);
          applyLivePacket(state.latestPacket);
        });
      });
    }

    // Update Track Canvas
    if (liveTrackRenderer) {
      liveTrackRenderer.setSelectedDrivers([state.selectedDriverA, state.selectedDriverB]);
      liveTrackRenderer.updateDrivers(drivers, caps.gps_available);
    }

    // Update Telemetry Compare Chart
    const drvA = drivers.find((d) => Number(d.driver_number) === state.selectedDriverA);
    const drvB = drivers.find((d) => Number(d.driver_number) === state.selectedDriverB);
    if (drvA) {
      const spA = drvA.telemetry && drvA.telemetry.speed_kph != null ? `${Math.round(drvA.telemetry.speed_kph)} KPH` : "--";
      const gA = drvA.telemetry && drvA.telemetry.gear != null ? ` · G${drvA.telemetry.gear}` : "";
      $("telem-drv-a").textContent = `${drvA.code} ${spA}${gA}`;
    }
    if (drvB) {
      const spB = drvB.telemetry && drvB.telemetry.speed_kph != null ? `${Math.round(drvB.telemetry.speed_kph)} KPH` : "--";
      const gB = drvB.telemetry && drvB.telemetry.gear != null ? ` · G${drvB.telemetry.gear}` : "";
      $("telem-drv-b").textContent = `${drvB.code} ${spB}${gB}`;
    }
    const drsVal = drvA && drvA.telemetry ? drvA.telemetry.drs_active : null;
    $("telem-drs-badge").textContent = drsVal === true ? "DRS OPEN" : drsVal === false ? "DRS CLOSED" : "DRS N/A";

    if (telemetryRenderer) {
      telemetryRenderer.setDriversMeta(
        { code: (drvA && drvA.code) || "A", color: "#38BDF8" },
        { code: (drvB && drvB.code) || "B", color: "#FF8000" }
      );
      telemetryRenderer.pushSample(drvA && drvA.telemetry, drvB && drvB.telemetry, caps.telemetry_available);
    }

    // Replay Controls state
    const isReplay = pkt.type === "replay_frame";
    $("replay-controls").classList.toggle("hidden", !isReplay);
    if (isReplay && pkt.replay) {
      $("btn-play-pause").textContent = pkt.replay.paused ? "Play" : "Pause";
      document.querySelectorAll(".spd-btn").forEach((b) => {
        b.classList.toggle("active", Number(b.dataset.speed) === Number(pkt.replay.speed));
      });
      const seekEl = $("seek-lap-input");
      seekEl.max = String(lapTot);
      if (document.activeElement !== seekEl) {
        seekEl.value = String(Math.max(1, lapCur));
      }
      $("seek-lap-readout").textContent = `${lapCur}/${lapTot}`;
    }

    // Team Radio auto-transcription & Auto-Play
    handleIncomingTeamRadio(race.team_radio || []);
    renderCommsLists();
  }

  /* ------------------------------------------------------------------------
     Team Radio MP3 Playback & Local faster-whisper Transcript Fetcher
     ------------------------------------------------------------------------ */
  function handleIncomingTeamRadio(clips) {
    const autoplay = $("radio-autoplay-check") && $("radio-autoplay-check").checked;
    for (const clip of clips) {
      const url = clip.recording_url;
      if (!url) continue;
      if (clip.transcript) {
        state.transcripts.set(url, clip.transcript);
      } else if (!state.transcripts.has(url) && !state.transcribingUrls.has(url) && !state.fixtureOverride) {
        requestWhisperTranscript(url);
      }
    }
    if (autoplay && clips.length) {
      const newest = clips[clips.length - 1];
      if (newest.recording_url && !state.playedAutoUrls.has(newest.recording_url)) {
        state.playedAutoUrls.add(newest.recording_url);
        playRadioClip(newest.recording_url);
      }
    }
  }

  async function requestWhisperTranscript(url) {
    state.transcribingUrls.add(url);
    try {
      const res = await fetch(`${state.apiBase}/api/radio/transcribe?url=${encodeURIComponent(url)}`);
      if (res.ok) {
        const data = await res.json();
        if (data && typeof data.transcript === "string" && data.transcript.trim()) {
          state.transcripts.set(url, data.transcript.trim());
        } else {
          state.transcripts.set(url, "");
        }
      }
    } catch (_) {
      // Network unavailable or server offline; keep audio playable
    } finally {
      state.transcribingUrls.delete(url);
      renderCommsLists();
    }
  }

  function playRadioClip(url) {
    const audio = $("radio-audio-player");
    if (!audio || !url) return;
    if (state.currentlyPlayingUrl === url && !audio.paused) {
      audio.pause();
      state.currentlyPlayingUrl = null;
      renderCommsLists();
      return;
    }
    state.currentlyPlayingUrl = url;
    audio.src = url;
    audio.play().catch(() => {
      state.currentlyPlayingUrl = null;
      renderCommsLists();
    });
    renderCommsLists();
  }

  function buildUnifiedCommsEvents() {
    const race = (state.latestPacket && (state.latestPacket.race || state.latestPacket.tick)) || {};
    const fia = (race.race_control || []).map((m) => ({
      kind: "fia",
      date: m.date || "",
      lap: m.lap_number,
      flag: String(m.flag || "").toUpperCase(),
      tag: `FIA · ${m.category || m.flag || "RACE CONTROL"}`.toUpperCase(),
      message: m.message || "",
    }));
    const radio = (race.team_radio || []).map((r) => ({
      kind: "radio",
      date: r.date || "",
      driver_code: r.driver_code || `#${r.driver_number}`,
      driver_number: r.driver_number,
      team_name: r.team_name || "Team Radio",
      team_colour: hexCol(r.team_colour),
      recording_url: r.recording_url,
      transcript: r.transcript || state.transcripts.get(r.recording_url) || null,
      transcribing: state.transcribingUrls.has(r.recording_url),
      tag: `${r.driver_code || r.driver_number} #${r.driver_number} · ${(r.team_name || "RADIO").toUpperCase()}`,
    }));

    return [...fia, ...radio].sort((a, b) => String(b.date).localeCompare(String(a.date)));
  }

  function renderCommsTarget(container, events, filter) {
    if (!container) return;
    const filtered = events.filter((e) => (filter === "all" ? true : e.kind === filter));
    if (!filtered.length) {
      container.innerHTML = `<div class="empty-state">No Race Control or Team Radio messages yet.</div>`;
      return;
    }
    const barHeights = [6, 12, 15, 9, 14, 16, 10, 7, 13, 15, 11, 8, 14, 10, 6];
    container.innerHTML = filtered.map((ev) => {
      const clock = fmtUtcClock(ev.date);
      const timeMeta = ev.lap != null ? `L${ev.lap} · ${clock}` : clock;
      if (ev.kind === "radio") {
        const isPlaying = state.currentlyPlayingUrl === ev.recording_url;
        const quoteHtml = ev.transcript
          ? `<p class="comms-body">"${esc(ev.transcript)}"</p>`
          : ev.transcribing
          ? `<p class="comms-body transcribing">Transcribing audio with Whisper...</p>`
          : `<p class="comms-body transcribing">Tap Play to listen to team radio audio.</p>`;
        const bars = barHeights
          .map((bh) => `<span class="wf-bar" style="height:${bh}px;background:${ev.team_colour}"></span>`)
          .join("");
        return `
          <article class="comms-item is-radio" style="border-color:${ev.team_colour}66">
            <div class="comms-item-head">
              <span class="comms-tag">${esc(ev.tag)}</span>
              <span class="comms-time">${esc(timeMeta)}</span>
            </div>
            ${quoteHtml}
            <div class="radio-controls">
              <button type="button" class="radio-play-btn ${isPlaying ? "playing" : ""}" data-audio-url="${esc(ev.recording_url)}">
                ${isPlaying ? "Pause" : "Play"}
              </button>
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

    container.querySelectorAll(".radio-play-btn").forEach((btn) => {
      btn.addEventListener("click", () => playRadioClip(btn.dataset.audioUrl));
    });
  }

  function renderCommsLists() {
    const allEvents = buildUnifiedCommsEvents();
    $("comms-count-all").textContent = String(allEvents.length);
    renderCommsTarget($("comms-feed"), allEvents, state.commsFilter);
    if (!$("comms-drawer").classList.contains("hidden")) {
      renderCommsTarget($("comms-drawer-list"), allEvents, state.drawerFilter);
    }
  }

  /* ------------------------------------------------------------------------
     Screen 3 Renderer: Separate Post-Qualifying Predictions Page
     ------------------------------------------------------------------------ */
  function renderPostQuali(pkt) {
    if (!pkt || !pkt.predictions) return;
    const items = Array.isArray(pkt.predictions.items) ? pkt.predictions.items : [];
    $("pred-gp-meta").textContent = `${pkt.season || 2026} · Round ${pkt.round || "--"} · ${pkt.grand_prix || ""}`;

    // Top-3 Projected Hero Cards
    const heroWrap = $("pred-podium-cards");
    const top3 = items.slice(0, 3);
    heroWrap.innerHTML = top3.map((it, idx) => {
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

    // Full Grid Continuous Probability Table (clean layout matching user's edited Figma PNG)
    const tableWrap = $("pred-table-list");
    if (!items.length) {
      tableWrap.innerHTML = `<div class="empty-state">Click "Run Post-Quali Forecast" to load grid probabilities.</div>`;
    } else {
      tableWrap.innerHTML = items.map((it, idx) => {
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
                  <span class="drv-full" title="${esc(it.full_name)}">${esc(it.full_name)} · ${esc(it.team_name)}</span>
                </div>
              </div>
            </div>
            <div class="pred-bar-track"><div class="pred-bar-fill" style="width:${Math.max(2, Math.min(100, pod))}%;background:${col}"></div></div>
            <span class="mono-val align-right" style="font-weight:700">${pod.toFixed(1)}%</span>
            <span class="mono-muted align-right">${win.toFixed(1)}%</span>
          </div>`;
      }).join("");
    }

    // Dynamic Circuit Preview on Screen 3
    const circ = pkt.circuit || {};
    const c2d = (circ.track_outline && circ.track_outline.length >= 10)
      ? circ
      : (circ.slug && window.F1_CIRCUITS_2D && window.F1_CIRCUITS_2D[circ.slug])
      || (window.F1_CIRCUITS_2D && window.F1_CIRCUITS_2D.las_vegas);
    if (c2d) {
      const rawName = String(circ.name || c2d.short_name || "Circuit").trim();
      $("pred-circuit-heading").textContent = /circuit$/i.test(rawName) ? rawName : `${rawName} Circuit`;
      $("pred-circuit-caption").textContent = `${pkt.grand_prix || ""} · ${(c2d.corners || []).length} Turns · S1 / S2 / S3 Split`;
      if (predTrackRenderer) predTrackRenderer.setCircuit(c2d);
    }

    // Feature weights
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
  }

  /* ------------------------------------------------------------------------
     API Calls & WebSocket + Broadcast Sync Delay Ring Buffer
     ------------------------------------------------------------------------ */
  async function loadInitialApiData() {
    // 1. Next Race (IST / UK / US)
    try {
      const r = await fetch(`${state.apiBase}/api/next-race`);
      if (r.ok) renderNextRace(await r.json());
    } catch (_) {}

    // 2. Pre-race Standings (WDC & WCC)
    try {
      const r = await fetch(`${state.apiBase}/api/standings`);
      if (r.ok) renderStandings(await r.json());
    } catch (_) {}

    // 3. Available 2026 Replay Sessions
    try {
      const r = await fetch(`${state.apiBase}/api/sessions?year=2026`);
      if (r.ok) {
        const d = await r.json();
        state.sessions = d.sessions || [];
        populateSessionDropdowns(state.sessions);
      }
    } catch (_) {}

    // 4. Health check for active mode
    try {
      const r = await fetch(`${state.apiBase}/api/health`);
      if (r.ok) {
        const h = await r.json();
        state.mode = h.mode || "replay";
        updateModeButtons();
      }
    } catch (_) {}

    // 5. Post-Quali forecast seed
    runPostQualiForecast(false);
  }

  function populateSessionDropdowns(sessions) {
    const sel = $("session-select");
    const roundSel = $("pred-round-select");
    if (sel) {
      sel.innerHTML = sessions.length
        ? sessions.map((s) => `<option value="${s.session_key}">R${s.round || "?"} · ${esc(s.grand_prix)}</option>`).join("")
        : `<option value="">No sessions cached</option>`;
    }
    if (roundSel) {
      roundSel.innerHTML = sessions.length
        ? sessions.map((s) => `<option value="${s.round || 1}">Round ${s.round || 1} — ${esc(s.grand_prix)}</option>`).join("")
        : `<option value="22">Round 22 — Las Vegas Grand Prix</option>`;
    }
  }

  async function runPostQualiForecast(showAlert) {
    const season = Number($("pred-season-select").value) || 2026;
    const round = Number($("pred-round-select").value) || undefined;
    try {
      const res = await fetch(`${state.apiBase}/api/predict/post-quali`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ season, round }),
      });
      if (res.ok) {
        const pkt = await res.json();
        state.postQualiPacket = pkt;
        renderPostQuali(pkt);
        return;
      }
    } catch (_) {}
    // Offline fallback derived from active 22-car grid
    if (!state.postQualiPacket) {
      const base = create22GridFixture("demo");
      const items = base.race.drivers.map((d, i) => ({
        rank: i + 1,
        driver_number: d.driver_number,
        driver_code: d.code,
        full_name: d.full_name,
        team_name: d.team_name,
        team_colour: d.team_colour,
        grid_position: d.grid_position,
        podium_prob_pct: d.podium_prob_pct,
        win_prob_pct: d.win_prob_pct,
      }));
      renderPostQuali({
        type: "post_quali_prediction",
        season: 2026,
        round: 22,
        grand_prix: "Las Vegas Grand Prix",
        circuit: base.circuit,
        predictions: { items },
      });
    }
    if (showAlert) $("pred-gp-meta").textContent = "Using local fallback grid";
  }

  async function loadReplaySession(sessionKey) {
    if (!sessionKey) return;
    $("ws-conn-status").textContent = `Loading session ${sessionKey}...`;
    try {
      const res = await fetch(`${state.apiBase}/api/replay/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_key: Number(sessionKey), include_telemetry: true, autoplay: true }),
      });
      if (res.ok) {
        state.mode = "replay";
        updateModeButtons();
        $("ws-conn-status").textContent = "Session loaded · Streaming 4Hz";
      }
    } catch (_) {
      $("ws-conn-status").textContent = "Failed to load replay session";
    }
  }

  async function sendReplayControl(payload) {
    try {
      await fetch(`${state.apiBase}/api/replay/control`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (_) {}
  }

  async function switchBackendMode(mode) {
    state.mode = mode;
    updateModeButtons();
    try {
      await fetch(`${state.apiBase}/api/mode`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
      });
    } catch (_) {}
  }

  function updateModeButtons() {
    document.querySelectorAll(".seg-opt[data-mode]").forEach((b) => {
      b.classList.toggle("active", b.dataset.mode === state.mode);
    });
  }

  function connectWebSocket() {
    if (state.ws) {
      try { state.ws.close(); } catch (_) {}
    }
    const url = toWsUrl(state.apiBase);
    $("ws-conn-status").textContent = `Connecting to ${url}...`;
    let ws;
    try {
      ws = new WebSocket(url);
    } catch (_) {
      $("ws-conn-status").textContent = "WebSocket unavailable";
      return;
    }
    state.ws = ws;

    ws.onopen = () => {
      $("ws-conn-status").textContent = `Connected · ${state.apiBase}`;
    };

    ws.onmessage = (evt) => {
      if (state.fixtureOverride) return;
      let pkt;
      try { pkt = JSON.parse(evt.data); } catch (_) { return; }
      if (!pkt || pkt.type === "pong") return;

      if (state.syncDelayS <= 0 || pkt.type === "post_quali_prediction") {
        applyLivePacket(pkt);
      } else {
        state.delayBuffer.push({ at: Date.now() + state.syncDelayS * 1000, pkt });
        if (state.delayBuffer.length > 400) state.delayBuffer.shift();
      }
    };

    ws.onclose = () => {
      $("ws-conn-status").textContent = "Reconnecting...";
      clearTimeout(state.wsReconnectTimer);
      state.wsReconnectTimer = setTimeout(connectWebSocket, 2500);
    };
  }

  // Drain Broadcast Sync Delay ring buffer at 4Hz
  setInterval(() => {
    if (state.fixtureOverride || !state.delayBuffer.length) return;
    const now = Date.now();
    let latestDue = null;
    while (state.delayBuffer.length && state.delayBuffer[0].at <= now) {
      latestDue = state.delayBuffer.shift().pkt;
    }
    if (latestDue) applyLivePacket(latestDue);
  }, 200);

  /* ------------------------------------------------------------------------
     Dev-Only Break-UI Stress Test Toggle (?data=worst or ?debug=1)
     ------------------------------------------------------------------------ */
  function initDebugFixtureBar() {
    const params = new URLSearchParams(window.location.search);
    const mode = params.get("data");
    const debug = params.get("debug");
    if (!mode && !debug) return;

    const bar = $("debug-fixture-bar");
    bar.classList.remove("hidden");

    const applyFixture = (kind) => {
      bar.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.fixture === kind));
      if (kind === "live") {
        state.fixtureOverride = null;
        if (state.latestPacket) applyLivePacket(state.latestPacket);
        return;
      }
      state.fixtureOverride = kind;
      const fx = create22GridFixture(kind);
      applyLivePacket(fx);
      renderStandings({
        season: 2026,
        after_round: 22,
        drivers: fx.race.drivers.map((d, i) => ({
          position: i + 1,
          driver_code: d.code,
          full_name: d.full_name,
          team_name: d.team_name,
          team_colour: d.team_colour,
          points: Math.max(0, 393 - i * 18),
          wins: i === 0 ? 8 : i === 1 ? 5 : i === 2 ? 4 : 0,
        })),
        constructors: kind === "empty" ? [] : [
          { position: 1, team_name: "McLaren Formula 1 Team", team_colour: "FF8000", wins: 8, points: 647 },
          { position: 2, team_name: "Scuderia Ferrari", team_colour: "E80020", wins: 5, points: 572 },
          { position: 3, team_name: "Oracle Red Bull Racing", team_colour: "3671C6", wins: 8, points: 541 },
        ],
      });
    };

    bar.querySelectorAll("button").forEach((b) => {
      b.addEventListener("click", () => {
        const k = b.dataset.fixture;
        const u = new URL(window.location.href);
        u.searchParams.set("data", k);
        window.history.replaceState({}, "", u.toString());
        applyFixture(k);
      });
    });

    if (mode && mode !== "live") applyFixture(mode);
  }

  /* ------------------------------------------------------------------------
     Seed Smooth Demo Telemetry Curve (until live replay telemetry streams)
     ------------------------------------------------------------------------ */
  function seedSmoothTelemetryDemo() {
    if (!telemetryRenderer) return;
    const sA = [];
    const sB = [];
    for (let i = 0; i < 65; i++) {
      const t = i / 64;
      const spdA = 210 + 115 * Math.sin(t * Math.PI * 2.1 - 0.5) + 20 * Math.cos(t * Math.PI * 4.2);
      const spdB = spdA - 5 - 3 * Math.sin(t * Math.PI * 3);
      const brkWindow = t > 0.42 && t < 0.62 ? Math.pow(Math.sin(((t - 0.42) / 0.2) * Math.PI), 1.6) * 96 : 0;
      const thrA = brkWindow > 8 ? Math.max(0, 100 - brkWindow * 1.15) : Math.min(100, 82 + 18 * Math.sin(t * Math.PI * 2));
      const thrB = Math.max(0, thrA - 3);
      sA.push({ speed: Math.max(75, Math.min(346, spdA)), throttle: thrA, brake: brkWindow });
      sB.push({ speed: Math.max(72, Math.min(341, spdB)), throttle: thrB, brake: brkWindow * 0.94 });
    }
    telemetryRenderer.seedDemoHistory(sA, sB);
  }

  /* ------------------------------------------------------------------------
     Bootstrap & Event Bindings
     ------------------------------------------------------------------------ */
  function init() {
    liveTrackRenderer = new window.TrackCanvasRenderer($("track-canvas"), { compact: false });
    predTrackRenderer = new window.TrackCanvasRenderer($("pred-circuit-canvas"), { compact: true });
    telemetryRenderer = new window.TelemetryChartRenderer($("telemetry-canvas"));

    // Screen tabs & hub cards
    document.querySelectorAll(".tab-btn").forEach((b) => {
      b.addEventListener("click", () => setScreen(b.dataset.screen));
    });
    document.querySelectorAll(".hub-card").forEach((c) => {
      c.addEventListener("click", () => setScreen(c.dataset.goto));
    });

    // Settings popover
    const pop = $("settings-popover");
    const btnSet = $("btn-settings");
    btnSet.addEventListener("click", () => {
      const hidden = pop.classList.toggle("hidden");
      btnSet.setAttribute("aria-expanded", hidden ? "false" : "true");
    });
    $("btn-close-settings").addEventListener("click", () => {
      pop.classList.add("hidden");
      btnSet.setAttribute("aria-expanded", "false");
    });

    // Mode switcher
    document.querySelectorAll(".seg-opt[data-mode]").forEach((b) => {
      b.addEventListener("click", () => switchBackendMode(b.dataset.mode));
    });

    // Session load
    $("btn-load-session").addEventListener("click", () => {
      const sk = $("session-select").value;
      if (sk) loadReplaySession(sk);
    });

    // Broadcast Sync Delay slider
    const slider = $("sync-delay-slider");
    slider.value = String(state.syncDelayS);
    $("sync-delay-value").textContent = `${state.syncDelayS}s`;
    slider.addEventListener("input", () => {
      state.syncDelayS = Number(slider.value) || 0;
      $("sync-delay-value").textContent = `${state.syncDelayS}s`;
      localStorage.setItem(STORAGE_KEY_DELAY, String(state.syncDelayS));
      if (state.syncDelayS === 0) state.delayBuffer = [];
    });

    // Custom LAN Host URL
    const hostInput = $("api-host-input");
    hostInput.value = state.apiBase;
    $("btn-save-host").addEventListener("click", () => {
      const v = hostInput.value.trim().replace(/\/+$/, "");
      if (v) {
        localStorage.setItem(STORAGE_KEY_HOST, v);
        state.apiBase = v;
      } else {
        localStorage.removeItem(STORAGE_KEY_HOST);
        state.apiBase = resolveApiBase();
      }
      connectWebSocket();
      loadInitialApiData();
    });

    // Replay transport controls
    $("btn-play-pause").addEventListener("click", () => {
      const paused = state.latestPacket && state.latestPacket.replay && state.latestPacket.replay.paused;
      sendReplayControl({ action: paused ? "play" : "pause" });
    });
    document.querySelectorAll(".spd-btn").forEach((b) => {
      b.addEventListener("click", () => sendReplayControl({ action: "set_speed", speed: Number(b.dataset.speed) }));
    });
    $("seek-lap-input").addEventListener("change", (e) => {
      sendReplayControl({ action: "seek_lap", lap: Number(e.target.value) });
    });

    // Comms filters & full-view drawer
    document.querySelectorAll("[data-comms-filter]").forEach((b) => {
      b.addEventListener("click", () => {
        state.commsFilter = b.dataset.commsFilter;
        document.querySelectorAll("[data-comms-filter]").forEach((x) => x.classList.toggle("active", x === b));
        renderCommsLists();
      });
    });
    document.querySelectorAll("[data-drawer-filter]").forEach((b) => {
      b.addEventListener("click", () => {
        state.drawerFilter = b.dataset.drawerFilter;
        document.querySelectorAll("[data-drawer-filter]").forEach((x) => x.classList.toggle("active", x === b));
        renderCommsLists();
      });
    });
    $("btn-open-comms-full").addEventListener("click", openCommsDrawer);
    $("btn-close-comms-drawer").addEventListener("click", closeCommsDrawer);
    $("comms-backdrop").addEventListener("click", closeCommsDrawer);
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        closeCommsDrawer();
        pop.classList.add("hidden");
      }
    });

    // Circuit search
    $("circuit-search").addEventListener("input", (e) => renderCircuitsCatalog(e.target.value));

    // Post-Quali Forecast button
    $("btn-run-post-quali").addEventListener("click", () => runPostQualiForecast(true));

    // Audio player ended reset
    $("radio-audio-player").addEventListener("ended", () => {
      state.currentlyPlayingUrl = null;
      renderCommsLists();
    });

    // Render 24 official circuits + initial 22-car preview immediately so screen never renders blank
    renderCircuitsCatalog("");
    const initialDemo = create22GridFixture("demo");
    applyLivePacket(initialDemo);
    seedSmoothTelemetryDemo();
    renderStandings({
      season: 2026,
      after_round: 22,
      drivers: initialDemo.race.drivers.map((d, i) => ({
        position: i + 1,
        driver_code: d.code,
        full_name: d.full_name,
        team_name: d.team_name,
        team_colour: d.team_colour,
        points: Math.max(0, 393 - i * 18),
        wins: i === 0 ? 8 : i === 1 ? 5 : i === 2 ? 4 : i === 3 ? 3 : i === 4 ? 1 : 0,
      })),
      constructors: [
        { position: 1, team_name: "McLaren Formula 1 Team", team_colour: "FF8000", wins: 8, points: 647 },
        { position: 2, team_name: "Scuderia Ferrari", team_colour: "E80020", wins: 5, points: 572 },
        { position: 3, team_name: "Oracle Red Bull Racing", team_colour: "3671C6", wins: 8, points: 541 },
        { position: 4, team_name: "Mercedes-AMG PETRONAS", team_colour: "27F4D2", wins: 1, points: 394 },
        { position: 5, team_name: "Atlassian Williams Racing", team_colour: "1868DB", wins: 0, points: 138 },
        { position: 6, team_name: "Aston Martin Aramco", team_colour: "229971", wins: 0, points: 104 },
      ],
    });
    renderNextRace({
      grand_prix: "Qatar Grand Prix",
      round: 23,
      countdown: { label: "UPCOMING" },
      sessions: [
        { name: "Practice 1", times: { ist: { day_short: "Fri", time_24h: "19:00" }, uk: { time_24h: "13:30" }, us_eastern: { time_24h: "08:30" }, us_pacific: { time_24h: "05:30" } } },
        { name: "Sprint Qualifying", times: { ist: { day_short: "Fri", time_24h: "23:00" }, uk: { time_24h: "17:30" }, us_eastern: { time_24h: "12:30" }, us_pacific: { time_24h: "09:30" } } },
        { name: "Sprint", times: { ist: { day_short: "Sat", time_24h: "19:30" }, uk: { time_24h: "14:00" }, us_eastern: { time_24h: "09:00" }, us_pacific: { time_24h: "06:00" } } },
        { name: "Qualifying", times: { ist: { day_short: "Sat", time_24h: "23:30" }, uk: { time_24h: "18:00" }, us_eastern: { time_24h: "13:00" }, us_pacific: { time_24h: "10:00" } } },
        { name: "Grand Prix", times: { ist: { day_short: "Sun", time_24h: "21:30" }, uk: { time_24h: "16:00" }, us_eastern: { time_24h: "11:00" }, us_pacific: { time_24h: "08:00" } } },
      ],
    });

    setScreen(state.activeScreen);
    initDebugFixtureBar();

    // Connect to live FastAPI backend & WebSocket stream
    loadInitialApiData();
    connectWebSocket();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
