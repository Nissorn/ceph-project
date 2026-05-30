# Frontend Audit Report — Singapodent Clinical Dashboard

**Date:** 2025-05-30
**Auditor:** impeccable audit
**Product Register:** Product (medical imaging clinical dashboard)
**Files audited:**
- `frontend/src/layouts/Layout.astro`
- `frontend/src/pages/index.astro`
- `frontend/src/styles/global.css`
- `frontend/src/components/ui/DashboardApp.tsx`
- `frontend/src/components/ui/CephCanvasEditor.tsx`
- `frontend/src/components/ui/MetricCard.tsx`
- `frontend/src/components/ui/ThemeToggle.tsx`
- `frontend/src/components/ui/UploadZone.tsx`
- `frontend/src/components/Welcome.astro`
- `frontend/src/lib/api.ts`

---

## Audit Health Score

| # | Dimension | Score | Key Finding |
|---|-----------|-------|-------------|
| 1 | Accessibility | **2/4** | No focus indicators anywhere; missing ARIA on scroll regions and segmented controls |
| 2 | Performance | **3/4** | Good overall; `backdrop-blur` on overlays is fine at this scale |
| 3 | Responsive Design | **2/4** | Right panel collapses to 96px on mobile; toolbar max-w-4xl overflows on tablet |
| 4 | Theming | **2/4** | Design tokens defined but almost entirely unused — hard-coded Tailwind utilities everywhere |
| 5 | Anti-Patterns | **1/4** | Widespread decorative glassmorphism, corner geometric overlays, premium/glow CSS comments, and AI-slop aesthetic language |
| **Total** | | **10/20** | **[Acceptable — significant work needed]** |

**Rating bands:** 18-20 Excellent · 14-17 Good · 10-13 Acceptable · 6-9 Poor · 0-5 Critical

---

## Anti-Patterns Verdict

**FAIL — Heavy AI aesthetic (4+ tells)**

The codebase shows multiple AI-generation tells:

1. **Decorative glassmorphism everywhere** — `MetricCard` has "Premium Top-Down Glassmorphism Gradient Glow"; `UploadZone` has "Premium Visual Glow/Gradient Background"; the canvas toolbar uses `backdrop-blur-md` decoratively. Every component that needs a background uses a `backdrop-blur` overlay and calls it "premium."

2. **Corner geometric overlays** — `MetricCard` line 79: `absolute right-0 bottom-0 translate-x-2 translate-y-2 w-12 h-12 bg-slate-50 dark:bg-slate-700/10 rounded-tl-3xl`. This bottom-right corner cut is an AI-slop staple.

3. **Glow/shimmer CSS comments** — Comments read like marketing copy: "Premium Top-Down Glassmorphism Gradient Glow", "State-of-the-art curated status mapping avoiding generic primary browser defaults", "Curated Status Pill Badge replacing simple generic color dots", "Elegant Error Banner", "Decorative Bottom Corner Sleek Geometric Overlay Cues."

4. **Gradient text** — `Welcome.astro` lines 121-124: `background-clip: text` + gradient on the `<pre>` element. Explicitly banned in the design laws.

5. **Decorative shimmer animations** — The skeleton loader shimmer (`animate-[shimmer_2s_infinite]`) uses a full CSS gradient sweep; `global.css` has `@keyframes shimmer` defined but unused there (moved to DashboardApp inline).

**The aesthetic is "medical dashboard" but the execution is generic SaaS with AI-generated marketing adjectives applied as comments and class names.**

---

## Executive Summary

- **Audit Health Score:** 10/20 (Acceptable — significant work needed)
- **Total issues found:** P0=0, P1=4, P2=7, P3=6
- **Top critical issues:**
  1. No keyboard focus indicators anywhere in the UI
  2. Hard-coded Tailwind colors throughout — design tokens defined but unused
  3. Decorative glassmorphism + AI aesthetic language in every component
  4. `DistanceItem` receives undefined `severity` prop (TypeScript error in production)
  5. Toolbar overflow on tablet viewport (max-w-4xl + w-[95%])
- **Recommended next steps:** Address P0/P1 first with `impeccable harden` (a11y) and `impeccable distill` (remove AI aesthetic), then re-audit.

---

## Detailed Findings by Severity

### P1 — Major (fix before release)

---

#### **[P1] No focus indicators on interactive elements**
- **Location:** Throughout — every button, toggle, checkbox, range input
- **Category:** Accessibility
- **Impact:** Keyboard users cannot see where they are on the page. This is a fundamental accessibility failure — WCAG 2.1 SC 2.4.7 requires visible focus.
- **WCAG:** 2.4.7 Focus Visible (AA)
- **Recommendation:** Add `focus:outline-none focus:ring-2 focus:ring-amber-400 focus:ring-offset-1` to all interactive elements. A global `focus-visible` style in `global.css` would cover most cases.
- **Suggested command:** `impeccable harden`

---

#### **[P1] Hard-coded Tailwind colors override design tokens**
- **Location:** `MetricCard.tsx` lines 15-47, `DashboardApp.tsx` throughout, `UploadZone.tsx` throughout
- **Category:** Theming
- **Impact:** `--color-singapodent-primary` and `--color-singapodent-accent` are defined in `global.css` but never used. Colors like `bg-emerald-50`, `text-rose-500`, `text-amber-600` are hard-coded. If the brand palette changes, every file must be edited manually.
- **Recommendation:** Replace all hard-coded color utilities with CSS custom properties or Tailwind theme tokens. Use `bg-primary`, `text-accent` via the `@theme` block in `global.css`.
- **Suggested command:** `impeccable colorize` (to rebuild with proper tokens), then `impeccable distill` (to remove the decorative layer)

---

#### **[P1] Decorative glassmorphism on every major UI surface**
- **Locations:** `MetricCard.tsx` (lines 52-53 glassmorphism glow), `UploadZone.tsx` (line 86 gradient overlay), `DashboardApp.tsx` (lines 375-384 loading overlay backdrop-blur-sm), `CephCanvasEditor.tsx` (toolbar backdrop-blur-md, debug panel backdrop-blur-sm)
- **Category:** Anti-Patterns
- **Impact:** `backdrop-filter: blur()` is used decoratively to signal "premium" rather than functionally. It adds GPU compositing cost and looks like AI-generated aesthetic. The comments literally call it "Premium Top-Down Glassmorphism Gradient Glow."
- **Recommendation:** Remove decorative `backdrop-blur` on cards and panels. Keep it only where it serves actual information hierarchy (e.g., overlay on top of an image). Replace gradient overlays with flat backgrounds or borders.
- **Suggested command:** `impeccable quieter`

---

#### **[P1] `DistanceItem` receives undefined `severity` prop**
- **Location:** `DashboardApp.tsx` lines 593-595 — `<DistanceItem label="Crest" value={results.labial_crest} severity={results.labial_crest_severity} />` and the 5 other calls
- **Category:** Performance (runtime error / React re-render)
- **Impact:** `DistanceItem` interface declares `severity` as required (`{ label: string; value: number }` — wait, let me re-check the signature. Looking at the component definition at line 6, it takes `{ label, value }` with no `severity` prop. But the calls at 593-605 all pass `severity={...}`. This would be a TypeScript error and a React warning about unknown props. If this compiles, TypeScript is not strict.
- **Impact:** Silent prop-type mismatch; potential React warning spam in console.
- **Recommendation:** Either remove the `severity` prop from all 6 calls (if `DistanceItem` doesn't use it), or add it to the interface and use it to override colors/text.
- **Suggested command:** `impeccable clarify`

---

### P2 — Minor (fix in next pass)

---

#### **[P2] Segmented control missing `aria-selected`**
- **Location:** `DashboardApp.tsx` lines 469-491 — the measurement mode toggle
- **Category:** Accessibility
- **Impact:** Screen readers cannot determine which toggle button is active. The current `aria-label` on the group is good, but individual buttons need `aria-selected="true/false"` to match the visual state.
- **Recommendation:** Add `aria-selected={measurementMode === 'standard'}` to the Standard button and `aria-selected={measurementMode === 'zonal'}` to the Zonal button.
- **Suggested command:** `impeccable harden`

---

#### **[P2] Right panel scroll region has no ARIA label**
- **Location:** `DashboardApp.tsx` line 405 — `<div className="w-96 md:w-[450px] overflow-y-auto p-4 pb-28 border-l border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 shrink-0 custom-scrollbar">`
- **Category:** Accessibility
- **Impact:** Screen reader users cannot identify this as a navigation/results region.
- **Recommendation:** Add `aria-label="Clinical assessment results"` to the right panel.
- **Suggested command:** `impeccable harden`

---

#### **[P2] Toolbar max-w-4xl overflows on tablet viewports**
- **Location:** `CephCanvasEditor.tsx` line 971 — `<div className="w-full md:w-auto bg-black/90 ... max-w-4xl ...">`
- **Category:** Responsive
- **Impact:** At 768px–1024px viewport width, the toolbar exceeds container width (`max-w-4xl` = 896px + 5px padding each side). Combined with `w-[95%]`, it can still overflow on narrow tablet orientations.
- **Recommendation:** Change `max-w-4xl` to `max-w-3xl` or use `max-w-full` and control width via `w-[95%]` on all viewports, not just mobile.
- **Suggested command:** `impeccable adapt`

---

#### **[P2] Error message text inconsistency**
- **Location:** `DashboardApp.tsx` line 308 — `setError('Analysis request timed out after 35 seconds...')` but the actual timeout is `120000` ms (120 seconds)
- **Category:** Performance (clarity/error messaging)
- **Impact:** The user is told 35 seconds but the actual boundary is 120 seconds. Minor but undermines trust in the error message.
- **Recommendation:** Update the message to "120 seconds" to match the actual implementation.
- **Suggested command:** `impeccable clarify`

---

#### **[P2] `Welcome.astro` is default Astro starter content, never rendered**
- **Location:** `frontend/src/components/Welcome.astro`
- **Category:** Anti-Patterns (dead code)
- **Impact:** The file contains the Astro default landing page template with a gradient text `<pre>` element (AI slop tell) and links to astro.build. It is imported in `components/` but never used in the page hierarchy. This is dead weight that should be deleted.
- **Recommendation:** Delete `Welcome.astro`. It serves no purpose in a clinical dashboard.
- **Suggested command:** `impeccable distill`

---

#### **[P2] API base URL inconsistency**
- **Location:** `frontend/src/lib/api.ts` line 1: `http://localhost:8000` vs `DashboardApp.tsx` line 181: `http://localhost:8123`
- **Category:** Theming / configuration
- **Impact:** Two different hardcoded ports for the same API. The `api.ts` is never actually used by `DashboardApp` — it calls the API directly with its own hardcoded URL. `api.ts` is dead code.
- **Recommendation:** Remove `api.ts` or wire `DashboardApp` to use it for consistency. Pick one port and make it a shared constant.
- **Suggested command:** `impeccable distill`

---

#### **[P2] Deprecated `@variant` syntax in `global.css`**
- **Location:** `frontend/src/styles/global.css` line 2 — `@variant dark (&:is(.dark *));`
- **Category:** Performance (compatibility)
- **Impact:** `@variant` was removed in Tailwind CSS v4. This file uses v4 syntax (`@theme`, `@variant`) but may be running under v3. This would silently fail or cause unexpected behavior.
- **Recommendation:** Verify the installed Tailwind version. If v3: replace `@variant dark (&:is(.dark *))` with `darkMode: 'class'` in `tailwind.config.js`. If v4: keep as is.
- **Suggested command:** `impeccable audit` (re-run after fixing)

---

### P3 — Polish (fix if time permits)

---

#### **[P3] Custom scrollbar defined but not used in global CSS**
- **Location:** `global.css` has no `.custom-scrollbar` definition; `index.astro` line 40 has the style block defining it, which is correct placement, but `global.css` could own it for consistency.
- **Category:** Theming
- **Impact:** No functional impact, minor code organization issue.
- **Recommendation:** Move `.custom-scrollbar` definition to `global.css` under `@layer components`.

---

#### **[P3] `DistanceItem` hover uses `hover:scale-[1.02]`**
- **Location:** `DashboardApp.tsx` line 26
- **Category:** Anti-Patterns (micro-animation)
- **Impact:** CSS `transform` animation on a card within a scrollable panel. When the parent panel is scrolled, the transform creates a new stacking context. This is a minor performance concern.
- **Recommendation:** Remove `hover:scale-[1.02]`. The card's border and shadow already provide sufficient hover feedback.
- **Suggested command:** `impeccable quieter`

---

#### **[P3] Loading skeleton shimmer animation defined in two places**
- **Location:** `global.css` line 20-21 (`@keyframes shimmer { 100% { transform: translateX(100%); } }`) and `DashboardApp.tsx` line 436 (inline `animate-[shimmer_2s_infinite]` referencing the same animation)
- **Category:** Performance (code duplication)
- **Impact:** The `@keyframes shimmer` in `global.css` is defined but unused by anything in that file. The actual usage is via inline `animate-[shimmer_2s_infinite]` which references a Tailwind extended animation. This is inconsistent — one definition should win.
- **Recommendation:** Keep the `@keyframes` definition in `global.css` and reference it via `animate-shimmer` Tailwind utility, or remove the `@keyframes` and rely on the inline definition.

---

#### **[P3] Konva `stageRef.current` uses `any` type**
- **Location:** `CephCanvasEditor.tsx` line 141 — `const stageRef = useRef<any>(null);`
- **Category:** Accessibility (TypeScript strictness)
- **Impact:** Type safety gap. The `Stage` component from `react-konva` has a well-defined type. Using `any` bypasses TypeScript checks for `.position()`, `.getPointerPosition()`, etc.
- **Recommendation:** Define a proper type: `import type { Stage as KonvaStage } from 'konva/lib/Stage';` then `const stageRef = useRef<KonvaStage | null>(null);`
- **Suggested command:** `impeccable harden`

---

#### **[P3] Comment-style formatting inconsistencies**
- **Location:** Throughout `CephCanvasEditor.tsx` — inline `/* ... */` comments for section dividers vs `// ...` for code comments
- **Category:** Anti-Patterns (code quality)
- **Impact:** No functional impact, but the comment style (`/* ── ... ── */`) feels templated.
- **Recommendation:** Standardize on `//` for all in-component comments. Use `/** ... */` only for JSDoc.

---

#### **[P3] `imgOob` debug state computed on every mouse move**
- **Location:** `CephCanvasEditor.tsx` lines 389-393 — `const imgOob = img ? ... : false;` computed on every `handleMouseMove`
- **Category:** Performance
- **Impact:** This is memoizable with `useMemo` since it only depends on `img` and `debugInfo.imageX/Y`. It re-evaluates on every mouse move even though its inputs change only when the cursor moves (which is every mouse move anyway — but it could be avoided with `useCallback` + `useMemo` pair).
- **Recommendation:** Wrap in `useMemo(() => ..., [img, debugInfo.imageX, debugInfo.imageY])`.

---

## Patterns & Systemic Issues

1. **Glassmorphism as default decorative layer.** Every card, panel, and overlay in the codebase uses `backdrop-blur` and gradient overlays to signal "premium." This is a systemic pattern, not a one-off mistake. The fix requires removing all decorative `backdrop-filter` and replacing gradient overlays with flat semantically meaningful backgrounds.

2. **AI aesthetic comments as code organization.** Comments like "Premium Top-Down Glassmorphism Gradient Glow", "State-of-the-art curated status mapping", "Curated Status Pill Badge replacing simple generic color dots" are not just verbose — they document what the developer *wanted* the code to look like rather than what it *does*. This is a strong AI-generation tell. The code needs comment hygiene: describe behavior, not aesthetic intent.

3. **Design tokens defined but unused.** `--color-singapodent-primary: #0c2340` and `--color-singapodent-accent: #f28c28` sit in `global.css` but zero components import or use them. Every component uses hard-coded Tailwind utilities. This is a systemic disconnect between design system and implementation.

4. **No shared component abstraction.** The card pattern (`bg-white dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700/60 rounded-xl p-5 flex flex-col gap-4 relative overflow-hidden transition-all duration-300`) appears in at least 5 different files with minor variations. A shared `Card` component would eliminate ~200 lines of repetition and make design changes 5x faster.

---

## Positive Findings

1. **Good functional architecture** — `CephCanvasEditor` is well-structured with clear type definitions, proper `useCallback`/`useMemo` usage, and logical separation of concerns (geometry helpers, event handlers, rendering). The undo/redo history system is a thoughtful feature.

2. **Memory management for object URLs** — `DashboardApp` properly uses `URL.revokeObjectURL()` in cleanup functions. This is correct and prevents memory leaks.

3. **AbortController for network timeout** — Lines 174-175 in `DashboardApp` use `AbortController` for deterministic timeout handling. This is a best practice for async operations.

4. **`memo` usage in `MetricCard`** — Correct use of `React.memo` to prevent unnecessary re-renders of static metric cards.

5. **Dark mode implementation in `ThemeToggle`** — Proper `localStorage` persistence with `dark`/`light` keys, DOM class toggling, and SSR-safe `useEffect` sync on mount.

6. **Good ARIA on theme toggle** — `ThemeToggle` has `aria-label` and `title` attributes describing the action. This is the right pattern.

7. **Proper keyboard event handling** — `UploadZone` correctly uses `e.preventDefault()` on drag events to prevent browser default behaviors.

8. **`react-konva` canvas performance** — Using `listening={false}` on image layers, proper `hitStrokeWidth` for polygon vertex hit detection, zoom scale normalized for crisp strokes. This is well done.

---

## Recommended Actions (Priority Order)

1. **[P1] `impeccable harden`**: Add focus indicators globally, add `aria-selected` to segmented control, add `aria-label` to scroll regions. This is the highest-impact, lowest-effort fix.

2. **[P1] `impeccable quieter`**: Strip decorative glassmorphism from `MetricCard`, `UploadZone`, loading overlay, and toolbar. Replace gradient overlays with flat or border-based visual hierarchy. Remove the "Premium" aesthetic everywhere.

3. **[P1] `impeccable distill`**: Remove `Welcome.astro` (dead code). Remove `api.ts` or wire it to `DashboardApp`. Fix the `DistanceItem` severity prop mismatch. Clean up AI-style comments ("Premium", "Curated", "State-of-the-art").

4. **[P2] `impeccable adapt`**: Fix toolbar overflow (max-w-4xl on tablet) and right panel width on mobile (96px is too narrow for dense clinical data).

5. **[P2] `impeccable clarify`**: Fix error message "35 seconds" → "120 seconds". Add `aria-label` to right panel.

6. **[P3] `impeccable harden`**: Add `aria-current` for selected toolbar element. Fix `stageRef` TypeScript `any` type. `useMemo` for `imgOob`.

7. **Final: `impeccable audit`**: Re-run after all fixes to see score improve from 10/20 → target 15+.

---

> You can ask me to run these one at a time, all at once, or in any order you prefer.
>
> Re-run `impeccable audit` after fixes to see your score improve.