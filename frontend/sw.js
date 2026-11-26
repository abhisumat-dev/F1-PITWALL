/**
 * F1 Pit-Wall — Progressive Web App Service Worker
 *
 * Enables standalone desktop & mobile PWA installation (chromeless resizable OS
 * window) with offline shell caching for fonts, icons, and 2D circuit geometry,
 * while keeping all `/api/*` and HTML requests strictly network-first.
 */
const CACHE_NAME = "f1-pitwall-shell-v2";
const PRECACHE_ASSETS = [
  "/",
  "/index.html",
  "/live.html",
  "/predictions.html",
  "/pwa-manifest.json",
  "/icons/icon-192.png",
  "/icons/icon-512.png",
  "/fonts/barlow-400.woff2",
  "/fonts/barlow-500.woff2",
  "/fonts/barlow-600.woff2",
  "/fonts/barlow-700.woff2",
  "/fonts/barlow-condensed-600.woff2",
  "/fonts/barlow-condensed-700.woff2",
  "/fonts/jetbrains-mono-400.woff2",
  "/fonts/jetbrains-mono-500.woff2",
  "/fonts/jetbrains-mono-700.woff2",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(PRECACHE_ASSETS))
      .catch(() => {})
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/ws/")) return;

  // Cache-first for immutable self-hosted fonts & icons
  if (url.pathname.startsWith("/fonts/") || url.pathname.startsWith("/icons/")) {
    event.respondWith(
      caches.match(req).then((cached) => cached || fetch(req))
    );
    return;
  }

  // Network-first with cache fallback for HTML/CSS/JS shell
  event.respondWith(
    fetch(req)
      .then((res) => {
        if (res && res.ok) {
          const clone = res.clone();
          caches.open(CACHE_NAME).then((c) => c.put(req, clone)).catch(() => {});
        }
        return res;
      })
      .catch(() => caches.match(req))
  );
});
