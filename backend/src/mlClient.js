// Talks to the Flask ML service over REST.

class MLError extends Error {
  constructor(message, status = 502) {
    super(message);
    this.status = status;
  }
}

function createMLClient({ baseUrl, key = '', timeoutMs = 15000 }) {
  async function call(method, path, body) {
    let res;
    try {
      res = await fetch(`${baseUrl}${path}`, {
        method,
        headers: { 'Content-Type': 'application/json', ...(key ? { 'X-Service-Key': key } : {}) },
        body: body ? JSON.stringify(body) : undefined,
        signal: AbortSignal.timeout(timeoutMs),
      });
    } catch (err) {
      throw new MLError(`ML service unreachable at ${baseUrl}: ${err.message}`, 503);
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new MLError(data.error || `ML service error ${res.status}`, res.status === 422 ? 422 : 502);
    return data;
  }

  return {
    health: () => call('GET', '/health'),
    register: (visitorId, images, replace = false) =>
      call('POST', '/register', { visitor_id: visitorId, images, replace }),
    identify: (image) => call('POST', '/identify', { image }),
    remove: (visitorId) => call('DELETE', `/visitors/${visitorId}`),
  };
}

module.exports = { createMLClient, MLError };
