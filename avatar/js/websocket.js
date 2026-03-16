// js/websocket.js — Secure WebSocket with automatic exponential-backoff reconnect

const ALLOWED_ORIGINS = [
  'ws://localhost:8080',
  'wss://localhost:8443',
];

export class SecureWebSocket {
  /**
   * @param {string}   url            - WebSocket URL (must be in ALLOWED_ORIGINS)
   * @param {function} onMessage      - Called with parsed JSON payload
   * @param {function} onStatusChange - Called with 'connected'|'disconnected'|'reconnecting'
   */
  constructor(url, onMessage, onStatusChange) {
    // ── Origin whitelist check ─────────────────────────────────────────
    const allowed = ALLOWED_ORIGINS.some(o => url.startsWith(o));
    if (!allowed) {
      throw new Error(`[WS] Refused connection to non-whitelisted URL: ${url}`);
    }

    this._url            = url;
    this._onMessage      = onMessage;
    this._onStatusChange = onStatusChange;
    this._ws             = null;
    this._reconnectDelay = 1000;   // start at 1 s
    this._maxDelay       = 30000;  // cap at 30 s
    this._stopped        = false;

    this._connect();
  }

  // ── Connect ──────────────────────────────────────────────────────────────
  _connect() {
    if (this._stopped) return;

    console.log(`[WS] Connecting to ${this._url}…`);
    this._onStatusChange('reconnecting');

    try {
      this._ws = new WebSocket(this._url);
    } catch (e) {
      console.error('[WS] Failed to create socket:', e);
      this._scheduleReconnect();
      return;
    }

    this._ws.onopen = () => {
      console.log('[WS] Connected');
      this._reconnectDelay = 1000; // reset backoff on success
      this._onStatusChange('connected');
    };

    this._ws.onmessage = event => {
      // ── Message validation ─────────────────────────────────────────
      if (typeof event.data !== 'string') {
        console.warn('[WS] Ignored non-string message');
        return;
      }
      if (event.data.length > 65536) {
        console.warn('[WS] Ignored oversized message (>64 KB)');
        return;
      }

      let payload;
      try {
        payload = JSON.parse(event.data);
      } catch {
        console.warn('[WS] Ignored malformed JSON');
        return;
      }

      // Only accept plain objects — no arrays, no primitives
      if (typeof payload !== 'object' || Array.isArray(payload) || payload === null) {
        console.warn('[WS] Ignored unexpected payload type');
        return;
      }

      this._onMessage(payload);
    };

    this._ws.onerror = err => {
      console.error('[WS] Error:', err);
    };

    this._ws.onclose = event => {
      console.log(`[WS] Closed (code ${event.code}). Reason: ${event.reason || 'none'}`);
      this._onStatusChange('disconnected');
      if (!this._stopped) this._scheduleReconnect();
    };
  }

  // ── Exponential backoff ──────────────────────────────────────────────────
  _scheduleReconnect() {
    const delay = this._reconnectDelay;
    console.log(`[WS] Reconnecting in ${delay / 1000}s…`);
    setTimeout(() => this._connect(), delay);

    // Double the delay, cap at maxDelay
    this._reconnectDelay = Math.min(this._reconnectDelay * 2, this._maxDelay);
  }

  // ── Send ─────────────────────────────────────────────────────────────────
  send(data) {
    if (this._ws?.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(data));
    } else {
      console.warn('[WS] Cannot send — socket not open');
    }
  }

  // ── Graceful close ───────────────────────────────────────────────────────
  close() {
    this._stopped = true;
    this._ws?.close(1000, 'Client closed');
  }
}
