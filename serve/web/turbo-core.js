// serve/web/turbo-core.js - Turbo's request-level half, as plain functions (no page, no storage; serve/test_turbo.py runs them
// under node): what a chat's stored settings become when Turbo goes on, and what comes back when it goes off.  The engine's
// half (the context, the expert skipping, the drafting) is the server's: serve/models.py turbo_enable / turbo_disable.

// the fastest thinking level a model has: none where it offers it (GLM, DeepSeek, MiniMax), else its lowest (Qwen's low)
export const fastestThinking = (levels) => (levels.includes("none") ? "none" : levels[0]);

// Turbo on: `stored` is what the browser holds for this model's chat settings (only what its user chose; null for nothing).
// The snapshot keeps it whole, so that going off puts back exactly it; `next` is what Turbo stores: greedy decoding
// (temperature 0 - the other samplers do nothing then) and the fastest thinking level.
export function turboEnter(stored, levels, chatThinking = null) {
  const own = stored && typeof stored === "object" ? stored : null;
  const think = fastestThinking(levels);
  return {
    snapshot: {stored: own ? JSON.parse(JSON.stringify(own)) : null, chatThinking, think},
    next: {...(own || {}), temperature: 0, thinking: think, prefs_v: 3},
  };
}

// Turbo off: the stored settings as they were ({} for none: the model's own defaults).  The open chat's level goes back too,
// unless its user changed it since (it still reads Turbo's level).
export function turboLeave(snapshot, chatNow = null) {
  if (!snapshot) return {stored: null, chatThinking: undefined};
  const back = snapshot.chatThinking != null && chatNow === snapshot.think ? snapshot.chatThinking : undefined;
  return {stored: snapshot.stored ? JSON.parse(JSON.stringify(snapshot.stored)) : {}, chatThinking: back};
}

// the ideas a new chat offers while Turbo is on: coding, all of them
export const CODING_IDEAS = [
  {icon: "code", label: "Write a Python function",
   q: "Write a Python function that merges overlapping intervals in a list of (start, end) pairs, with a few tests."},
  {icon: "box", label: "A three.js scene in one HTML file",
   q: "Write a single-file HTML page with a three.js scene: a slowly rotating, softly lit crystal on a dark background, with orbit controls."},
  {icon: "tool", label: "Explain and improve some code",
   q: "Explain step by step what this code does, then suggest two improvements:\n\n```\n\n```"},
  {icon: "code", label: "Fix a bug",
   q: "This function should return the median of a list but is wrong for even lengths. Find the bug and fix it:\n\n```python\n\n```"},
];
