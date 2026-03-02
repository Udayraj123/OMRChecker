/**
 * Migrated from Python: src/utils/serialization.py
 * Recursively serialize a value to a plain JSON-compatible structure.
 * - plain objects → recursively serialized
 * - arrays → recursively serialized items
 * - Map → converted to plain object
 * - primitives (string, number, boolean, null) → pass through
 * - everything else → String(obj)
 */
export function deepSerialize(obj: unknown): unknown {
  if (obj === null || obj === undefined) return obj;
  if (typeof obj === 'string' || typeof obj === 'number' || typeof obj === 'boolean') return obj;
  if (Array.isArray(obj)) return obj.map(deepSerialize);
  if (obj instanceof Map) {
    return Object.fromEntries([...obj.entries()].map(([k, v]) => [k, deepSerialize(v)]));
  }
  if (typeof obj === 'object') {
    return Object.fromEntries(
      Object.entries(obj).map(([k, v]) => [k, deepSerialize(v)])
    );
  }
  try { return String(obj); } catch { return obj; }
}
