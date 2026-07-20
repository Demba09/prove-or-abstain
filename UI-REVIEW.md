# UI-REVIEW.md — prove-or-abstain Demo Page

**Audited file:** `api/static/index.html` (857 lines, single-file HTML/CSS/JS)  
**Date:** 2026-07-20  
**Auditor:** automated 6-pillar visual audit

---

## Overall Score: 23 / 24

| Pillar | Score | Verdict |
|--------|-------|---------|
| Copywriting | 3 / 4 | Good, minor jargon issues |
| Visuals | 4 / 4 | Excellent |
| Color | 4 / 4 | Excellent |
| Typography | 4 / 4 | Excellent |
| Spacing | 4 / 4 | Excellent |
| Experience Design | 3 / 4 | Good, missing polish on states |

---

## 1. Copywriting — 3/4

### Strengths
- **Header hook**: "It acts when the data proves a cause — and refuses to guess otherwise." Sets the tone in one sentence.
- **Scenario hints**: Each built-in scenario has a clear one-liner (`"one segment collapses → ASSERT"`) that explains the expected behavior before the user clicks.
- **Footer disclaimer**: Explicitly states "Every number comes from pandas/numpy; the LLM only orders..." — critical for trust.
- **Trace labels**: `"investigation trace"`, `"Qwen's tool-call trace"`, `"LLM speculation, not verified"` — clear boundaries between facts and LLM output.
- **Error messages**: ARIA `role="alert"` for screen reader announcements.
- **Empty state**: Dashboard shows `"No alerts yet — run an autonomous check..."` — useful call-to-action.

### Issues
- **"orchestration" label**: `span.mode-label` reads `"orchestration"` — correct but jargon. A more accessible label would be `"investigation mode"` or a tooltip explaining `"graph = fixed pipeline, agent = Qwen drives"`.
- **"Watch a source" text block**: The explanation paragraph (lines 307-310) is dense for a UI label. Consider a tooltip or collapsible detail section like "Advanced".
- **Autonomous check**: Uses `window.alert()` for result display — breaks immersion and uses OS-native dialog. Should render inline like investigation results.

### Recommendation
```html
<!-- line 298: replace "orchestration" -->
<span class="mode-label">investigation mode</span>
```

---

## 2. Visuals — 4/4

### Strengths
- **Zero external dependencies**: All visual elements (badges, chips, stamps, switches, charts) are pure CSS/SVG. No icon library, no framework.
- **SVG bar chart**: Self-contained dataset visualization with baseline/current comparison bars, tooltips via `<title>`, proper axis labels. Responsive viewBox.
- **Gate exhibits**: CSS grid with `auto-fit, minmax(250px, 1fr)` — adapts to viewport width without media queries.
- **Stamp verdict**: `rotate(-2deg)` + `border: 3px double` + colored background — distinctive and memorable. Good visual indicator that doesn't rely on color alone (text says "ASSERT" or "ABSTAIN").
- **Left-border accent**: Assert exhibits get `border-left: 3px solid var(--assert)`, abstain get `var(--abstain)` — subtle but effective differentiation.
- **Toggle switch**: Pure CSS using `::after` pseudo-element with transition — no JavaScript. Smooth.
- **Legend**: Color swatches with labels inline in chart header — matches chart stroke/fill colors.
- **Responsive**: `flex-wrap` on rows, CSS grid auto-fit, no fixed widths on containers. Works from mobile to desktop.

### Issues
None identified.

---

## 3. Color — 4/4

### Palette

| Token | Hex | Usage |
|-------|-----|-------|
| `--accent` | `#2E6F8E` | Links, tabs, buttons, focus |
| `--accent-strong` | `#1F5471` | Hover states, emphasized text |
| `--assert` | `#2F7A50` | ASSERT stamp, chip, exhibit border |
| `--assert-bg` | `#EAF3EC` | ASSERT background |
| `--abstain` | `#A6472E` | ABSTAIN stamp, chip, exhibit border |
| `--abstain-bg` | `#F5EAE6` | ABSTAIN background |
| `--ink` | `#131A20` | Primary text |
| `--muted` | `#5B6670` | Secondary text, labels, borders |
| `--line` | `#D9DEE1` | Borders, dividers |
| `--paper` | `#F4F6F7` | Page background |
| `--card` | `#FFFFFF` | Card/surface background |

### Strengths
- **Full light/dark pair**: Every color has a dark-mode counterpart in the `prefers-color-scheme: dark` block.
- **Manual toggle**: `:root[data-theme="dark"]` and `:root[data-theme="light"]` overrides — user can force theme regardless of system preference.
- **Semantic naming**: `--assert`/`--abstain` rather than `--green`/`--red` — maintainable and readable.
- **Accessible contrast**: Dark ink (#131A20) on light paper (#F4F6F7) ≈ 17:1. Assert text (#2F7A50) on assert-bg (#EAF3EC) still readable.
- **Active tabs**: White text on accent background (`#2E6F8E` → `#FFFFFF`) — strong contrast.
- **Color is not the only signal**: Verdict is conveyed through text ("ASSERT"/"ABSTAIN") + stamp styling + color, not color alone.

### Issues
None identified.

---

## 4. Typography — 4/4

### Scale

| Size | Usage |
|------|-------|
| 9.5px | Axis labels (SVG chart) |
| 10.5px | Tool verdict chips |
| 11px | Eyebrow labels, uppercase tags, labels |
| 11.5px | Scenario hints, LLM tags |
| 12px | Trace logs, agent steps |
| 12.5px | Data tables, routing notes, field labels |
| 13px | Exhibit headers |
| 13.5px | Body text, buttons, verdict meta |
| 15px | Primary text, report body |
| 19px | Dashboard stat numbers |
| 21px | Main heading |
| 22px | Verdict stamp |

### Strengths
- **Two font families**: `ui-sans-serif` for body/UI, `ui-monospace` for data/code — clear separation of content types.
- **Letter-spacing is intentional**: `-0.01em` (headings — tight), `.08em` (uppercase — airy), `.03em` (chips/tags — balanced), `.04em` (dim labels). Not just `0`.
- **Tabular numbers**: `font-variant-numeric: tabular-nums` on data tables, exhibits, trace logs — numbers align vertically.
- **Line heights**: `1.55` (body), `1.65` (report — more breathing room for prose), `1.5` (monospace). Good rhythm.
- **No external fonts**: Self-contained, no render-blocking network requests.
- **Hierarchy**: Heading → subtext → labeled sections → data — clear from size, weight, and color.

### Issues
- The SVG chart axis labels at 9.5px may be difficult to read on high-DPI screens. Consider bumping to 10px.

---

## 5. Spacing — 4/4

### Strengths
- **Consistent border-radius scale**: 6px → 7px (buttons/inputs) → 8px (cards/exhibits) → 10px (panels) → 999px (chips/toggles). Cohesive, not random.
- **Panel rhythm**: `padding: 18px 20px`, `margin-bottom: 20px` — predictable vertical flow.
- **Card padding**: exhibits `13px 15px`, panels `18px 20px` — proportional to content density.
- **Button sizing**: tabs `8px 14px`, secondary `9px 18px`, mode buttons `5px 11px` — appropriate for their hierarchy.
- **Gap system**: flex gaps of 6px, 8px, 10px, 12px, 14px, 16px, 18px, 20px — consistent small-scale increments.
- **Dividers**: `margin: 16px 0` — consistent section breaks.
- **SVG chart padding**: `padL=42, padB=26, padT=10, padR=10` — properly accounts for axis labels.
- **Max-width**: `940px` container — comfortable reading width, not edge-to-edge.

### Issues
None identified.

---

## 6. Experience Design — 3/4

### Strengths
- **ARIA landmarks**: `role="group"`, `role="alert"`, `aria-live="polite"`, `aria-pressed`, `aria-label` on interactive elements. Screen-reader aware.
- **Focus-visible**: Custom `outline: 2px solid var(--focus); outline-offset: 2px` on all interactive elements. Not relying on browser defaults.
- **Keyboard**: Enter triggers "ask" input. Buttons are keyboard-focusable by default. Toggle switch input is visually hidden but focusable.
- **Busy state**: All buttons disabled during API calls (`setBusy(true)`), cursor changes to `wait`.
- **Lazy dashboard**: Loads only when `<details>` is opened, auto-refreshes every 30s while open. Saves server load.
- **Advanced section**: Collapsed by default via `<details>` — reduces cognitive load for first-time users.
- **Results area**: `aria-live="polite"` ensures screen readers announce new results without interrupting.
- **Clear feedback loop**: Scenario → click → busy → result → dashboard refresh. Predictable.
- **Theme toggle**: `data-theme` attribute allows JS to force light/dark regardless of system preference.
- **SVG tooltips**: `<title>` elements inside bar groups provide accessible descriptions on hover.

### Issues
- **No loading indicator**: Buttons get `disabled` + `cursor: wait` but no spinner or skeleton. Users on slow connections see nothing for seconds.
- **`alert()` for autonomous check**: The `runAutonomousCheck()` function uses `window.alert()` which is OS-native, blocks interaction, and breaks the app's visual language. Should render inline like investigation results.
- **No confirmation on resolve**: The dashboard "resolve" button instantly resolves an alert with no confirmation. A misclick could clear an active alert.
- **First observation feedback**: When a "Watch a source" observation is seeded (cold_start=True), the verdict disappears but a message appears in the meta line. The transition is subtle — the stamp disappearing and reappearing as text might be confusing.
- **"Ask" context is implicit**: The query label updates to say "Asking about watched source X" but the user doesn't see which panel/source was selected before the query runs — results appear, then the routing note explains what happened. Consider showing the routing note before the API call (e.g., "Will investigate: clean scenario").

### Recommendations

1. **Add a spinner**:
```css
button:disabled::after {
  content: "…";
  margin-left: 4px;
  animation: pulse 1s infinite;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
```

2. **Replace `alert()` with inline result**:
Instead of `alert(...)`, render autonomous check results in a temporary panel within the dashboard section, matching the investigation result styling.

3. **Add resolve confirmation**:
```js
btn.onclick = function() {
  if (confirm('Resolve this alert?')) resolveAlert(a.id);
};
```

4. **Show "will investigate" preview**:
When the user types in the query field, show a pre-request note: "Will route to the {context} scenario" before they click "ask".

---

## Summary

| # | Finding | Pillar | Severity | Action |
|---|---------|--------|----------|--------|
| 1 | "orchestration" label is jargon | Copywriting | Low | Rename to "investigation mode" |
| 2 | Watch-a-source explanation is dense | Copywriting | Low | Move to collapsible detail |
| 3 | No loading spinner | Experience | Medium | Add CSS pulse animation on disabled buttons |
| 4 | `alert()` for autonomous check | Experience | Medium | Replace with inline result rendering |
| 5 | No confirmation on alert resolve | Experience | Medium | Add `confirm()` dialog |
| 6 | Cold-start feedback is subtle | Experience | Low | Make cold-start stamp visible (not hidden) |
| 7 | "Ask" context shown after, not before | Experience | Low | Show routing preview pre-request |

---

**Bottom line**: This is an exceptionally well-crafted single-file frontend. No external dependencies, complete dark mode, clean typography, and solid accessibility. The 3-point gap from 24/24 comes from missing UX polish (spinners, confirmation dialogs, progressive feedback on async states) — not structural issues. The remaining fixes are ~30 lines of code changes.
