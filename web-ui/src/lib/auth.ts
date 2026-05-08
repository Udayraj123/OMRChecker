export const AUTH_USERNAME = "kenji";
export const AUTH_PASSWORD = "12345";
export const SESSION_COOKIE = "omr_session";
export const SESSION_TTL_SECONDS = 60 * 60 * 12;

const encoder = new TextEncoder();

function getAuthSecret() {
  return process.env.AUTH_SECRET || "minsu-omr-kenji-local-session-secret";
}

function base64UrlEncodeBytes(bytes: Uint8Array) {
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }

  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function base64UrlEncode(value: string) {
  return base64UrlEncodeBytes(encoder.encode(value));
}

function base64UrlDecode(value: string) {
  const padded = value.replace(/-/g, "+").replace(/_/g, "/").padEnd(Math.ceil(value.length / 4) * 4, "=");
  const binary = atob(padded);
  const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

async function sign(value: string) {
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(getAuthSecret()),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const signature = await crypto.subtle.sign("HMAC", key, encoder.encode(value));
  return base64UrlEncodeBytes(new Uint8Array(signature));
}

function safeEqual(left: string, right: string) {
  if (left.length !== right.length) return false;

  let mismatch = 0;
  for (let index = 0; index < left.length; index += 1) {
    mismatch |= left.charCodeAt(index) ^ right.charCodeAt(index);
  }

  return mismatch === 0;
}

export function validateCredentials(username: string, password: string) {
  return username === AUTH_USERNAME && password === AUTH_PASSWORD;
}

export async function createSessionToken() {
  const payload = base64UrlEncode(
    JSON.stringify({
      sub: AUTH_USERNAME,
      exp: Date.now() + SESSION_TTL_SECONDS * 1000,
    }),
  );
  const signature = await sign(payload);
  return `${payload}.${signature}`;
}

export async function verifySessionToken(token?: string) {
  if (!token) return false;

  const [payload, signature] = token.split(".");
  if (!payload || !signature) return false;

  const expectedSignature = await sign(payload);
  if (!safeEqual(signature, expectedSignature)) return false;

  try {
    const session = JSON.parse(base64UrlDecode(payload)) as { sub?: string; exp?: number };
    return session.sub === AUTH_USERNAME && typeof session.exp === "number" && session.exp > Date.now();
  } catch {
    return false;
  }
}
