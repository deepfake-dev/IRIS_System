const ALLOWED_ORIGINS = [
  'ws://localhost:8080',
  'wss://localhost:8443',
];

export class SecureWebSocket {
  constructor(url, onMessage, onStatusChange, onBinary = null) {
    const allowed = ALLOWED_ORIGINS.some(o => url.startsWith(o));
    if (!allowed) throw new Error(`[WS] Refused non-whitelisted URL: ${url}`);

    this._url            = url;
    this._onMessage      = onMessage;
    this._onStatusChange = onStatusChange;
    this._onBinary       = onBinary;
    this._ws             = null;
    this._reconnectDelay = 1000;
    this._maxDelay       = 30000;
    this._stopped        = false;

    this._connect();
  }

  _connect() {
    if (this._stopped) return;
    this._onStatusChange('reconnecting');

    try {
      this._ws = new WebSocket(this._url);
      this._ws.binaryType = 'arraybuffer';
    } catch (e) {
      console.error('[WS] Failed to create socket:', e);
      this._scheduleReconnect();
      return;
    }

    this._ws.onopen = () => {
      this._reconnectDelay = 1000;
      this._onStatusChange('connected');
    };

    this._ws.onmessage = event => {
      if (event.data instanceof ArrayBuffer) {
        this._onBinary?.(event.data);
        return;
      }

      if (typeof event.data !== 'string') return;
      if (event.data.length > 65536) return;

      let payload;
      try {
        payload = JSON.parse(event.data);
      } catch {
        return;
      }

      if (typeof payload !== 'object' || Array.isArray(payload) || payload === null) return;

      this._onMessage(payload);
    };

    this._ws.onerror = err => console.error('[WS] Error:', err);

    this._ws.onclose = event => {
      this._onStatusChange('disconnected');
      if (!this._stopped) this._scheduleReconnect();
    };
  }

  _scheduleReconnect() {
    const delay = this._reconnectDelay;
    setTimeout(() => this._connect(), delay);
    this._reconnectDelay = Math.min(this._reconnectDelay * 2, this._maxDelay);
  }

  send(data) {
    if (this._ws?.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(data));
    }
  }

  sendBinary(arrayBuffer) {
    if (this._ws?.readyState === WebSocket.OPEN) {
      this._ws.send(arrayBuffer);
    }
  }

  close() {
    this._stopped = true;
    this._ws?.close(1000, 'Client closed');
  }
}