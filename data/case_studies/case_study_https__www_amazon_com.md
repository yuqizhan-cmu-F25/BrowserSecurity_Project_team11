# Case Study: https://www.amazon.com

**Generated:** 2025-12-01T13:53:06.536922

---

## Executive Summary

**Overall Accuracy Grade:** F (Poor)

**Accuracy Score:** 57.5%


### Key Findings
- High-severity tracking detected: canvas, webgl, audio, font, geolocation

---

## VV8 Trace Statistics

| Metric | Value |
|--------|-------|
| Total Unique Features | 317 |
| Total API Calls | 5,465 |
| Unique Scripts | 5 |
| First-Party Scripts | 0 |
| Third-Party Scripts | 5 |

### Top 10 Most-Used APIs

| API | Call Count |
|-----|------------|
| `Text.nodeType` | 248 |
| `Window.ue` | 248 |
| `Text.csm_node_id` | 226 |
| `NodeList.length` | 226 |
| `Window.ue_t0` | 179 |
| `NamedNodeMap.length` | 150 |
| `Window.ue_sid` | 148 |
| `Window.ue_mkt` | 147 |
| `Window.ue_mid` | 146 |
| `Window.ue_sn` | 146 |

---

## Detected Tracking Techniques


### 🔴 Canvas

**Severity:** HIGH

**Total API Calls:** 28

**Description:** Canvas fingerprinting uses HTML5 canvas to draw invisible shapes/text, then extracts the rendered pixels. Subtle hardware/driver differences create unique signatures.


**Evidence APIs:**
- `MutationObserver.` (3 calls)
- `HTMLCanvasElement.getContext` (1 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)

### 🔴 Webgl

**Severity:** HIGH

**Total API Calls:** 18

**Description:** WebGL fingerprinting queries GPU renderer/vendor strings and shader precision, creating device-specific signatures based on graphics hardware.


**Evidence APIs:**
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)

### 🔴 Audio

**Severity:** HIGH

**Total API Calls:** 33

**Description:** Audio fingerprinting generates sound through Web Audio API and measures output variations caused by hardware/driver differences.


**Evidence APIs:**
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `PerformanceNavigation.type` (6 calls)
- `MutationObserver.` (3 calls)

### 🟡 Navigator

**Severity:** MEDIUM

**Total API Calls:** 60

**Description:** Collects browser and device properties via Navigator API to build unique profile.


**Evidence APIs:**
- `MutationObserver.` (3 calls)
- `Navigator.userAgent` (1 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)

### 🟡 Screen

**Severity:** MEDIUM

**Total API Calls:** 642

**Description:** Screen dimensions and display properties help identify device type and configuration.


**Evidence APIs:**
- `Screen.width` (49 calls)
- `MutationObserver.` (3 calls)
- `Screen.height` (49 calls)
- `MutationObserver.` (3 calls)
- `Screen.width` (49 calls)

### 🔴 Font

**Severity:** HIGH

**Total API Calls:** 12

**Description:** Detects installed fonts by measuring text rendering differences, highly unique per system.


**Evidence APIs:**
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)

### 🟡 Storage

**Severity:** MEDIUM

**Total API Calls:** 36

**Description:** Uses browser storage APIs to persist tracking identifiers across sessions.


**Evidence APIs:**
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)

### 🟡 Cookie

**Severity:** MEDIUM

**Total API Calls:** 18

**Description:** Traditional cookie-based tracking and session management.


**Evidence APIs:**
- `MutationObserver.` (3 calls)
- `HTMLDocument.cookie` (3 calls)
- `MutationObserver.` (3 calls)
- `HTMLDocument.cookie` (3 calls)
- `MutationObserver.` (3 calls)

### 🟡 Input Tracking

**Severity:** MEDIUM

**Total API Calls:** 155

**Description:** Captures user input events for behavioral analysis and bot detection.


**Evidence APIs:**
- `MutationObserver.` (3 calls)
- `HTMLBodyElement.addEventListener` (1 calls)
- `HTMLDocument.addEventListener` (10 calls)
- `Window.addEventListener` (21 calls)
- `MutationObserver.` (3 calls)

### 🟢 Timing

**Severity:** LOW

**Total API Calls:** 181

**Description:** Collects precise timing data for performance analytics and potential fingerprinting.


**Evidence APIs:**
- `Performance.now` (13 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `PerformanceNavigation.type` (6 calls)

### 🟢 Visibility

**Severity:** LOW

**Total API Calls:** 24

**Description:** Tracks page/element visibility for viewability measurement and engagement analytics.


**Evidence APIs:**
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)

### 🔴 Geolocation

**Severity:** HIGH

**Total API Calls:** 13

**Description:** Requests precise geographic location - high privacy sensitivity.


**Evidence APIs:**
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `Window.location` (4 calls)
- `MutationObserver.` (3 calls)

### 🟡 Beacon

**Severity:** MEDIUM

**Total API Calls:** 11

**Description:** Sends tracking data reliably even during page unload.


**Evidence APIs:**
- `Navigator.sendBeacon` (5 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)

### 🟢 Xhr Fetch

**Severity:** LOW

**Total API Calls:** 18

**Description:** Network requests potentially transmitting tracking data to servers.


**Evidence APIs:**
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)

### 🟡 Websocket

**Severity:** MEDIUM

**Total API Calls:** 9

**Description:** Persistent connection for real-time tracking and session monitoring.


**Evidence APIs:**
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)
- `MutationObserver.` (3 calls)

### 🟡 Pixel Tracking

**Severity:** MEDIUM

**Total API Calls:** 15

**Description:** Tracking pixels for cross-site user identification.


**Evidence APIs:**
- `MutationObserver.` (3 calls)
- `HTMLScriptElement.src` (4 calls)
- `MutationObserver.` (3 calls)
- `Image` (5 calls)

---

## LLM Classification Analysis

### Existing Classification

- **Primary Category:** analytics
- **Confidence:** high
- **Privacy Risk:** high
- **Tracking Intensity:** heavy

**Explanation:** Amazon's tracking implementation combines analytics infrastructure (Window.ue_* properties for user event tracking, Navigator.sendBeacon for data transmission) with advertising capabilities and device fingerprinting. The heavy usage of custom tracking objects (ue_t0, ue_mid, ue_mkt, ue_sn, ue_sid) indicates comprehensive behavioral analytics, while the presence of fingerprinting features and 67 advertising-related APIs suggests multi-purpose tracking for both performance measurement and ad targeting.

---

## Validation Results

### Heuristic Validation

**Agreement Rate:** 25.0%

**Missed Techniques:** canvas, webgl, audio, font, storage, cookie, input_tracking, visibility, geolocation, xhr_fetch, websocket, pixel_tracking


### Ground Truth Validation

- **Ground Truth Source:** adblock_team
- **Is Known Tracker:** True
- **Tracker Agreement:** ✅ Yes
- **Category Agreement:** ✅ Yes
- **Validation Result:** correct

### Internal Consistency

**Consistent:** ✅ Yes

**Consistency Score:** 100.0%


### Error Analysis


**Potential False Positives (1):**
- fingerprinting: LLM may have over-interpreted ambiguous API patterns

**Potential False Negatives (5):**
- canvas (high severity)
- webgl (high severity)
- audio (high severity)
- font (high severity)
- geolocation (high severity)

---

## Accuracy Assessment

### Overall Grade: F (Poor)


**Strengths:**
- ✅ Internal consistency in classification
- ✅ Correct tracker/non-tracker classification

**Weaknesses:**
- ⚠️ Missed techniques: canvas, webgl, audio
- ⚠️ Potential false positives: 1
- ⚠️ Potential false negatives: 5

---

## Recommendations

- Consider adding explicit detection for: canvas, webgl, audio, font, storage, cookie, input_tracking, visibility, geolocation, xhr_fetch, websocket, pixel_tracking
- High-severity technique 'canvas' was not prominently identified
- High-severity technique 'webgl' was not prominently identified

---

*Report generated by VV8 Case Study Generator*