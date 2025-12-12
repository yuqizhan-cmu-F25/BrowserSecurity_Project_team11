#!/usr/bin/env python3
"""
Deep Case Study Generator for Single Website VV8 Trace Analysis

This module generates comprehensive case studies for individual websites,
focusing on validating the accuracy of LLM interpretations of VV8 (VisibleV8) traces.

Key Features:
1. Deep-dive analysis of all JavaScript API calls captured by VV8
2. Script-by-script breakdown with source attribution
3. Tracking technique identification and classification
4. Cross-validation of LLM interpretations against:
   - Known tracking signatures/heuristics
   - Ground truth databases (EasyPrivacy, EasyList, OISD)
   - Industry-standard fingerprinting patterns
5. Accuracy assessment with confidence scoring
6. Detection of potential LLM misinterpretations

Usage:
    python generate_case_study.py www.example.com
    python generate_case_study.py www.example.com --output-dir ./reports
    python generate_case_study.py www.example.com --format markdown
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from urllib.parse import urlparse
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import anthropic

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "case_studies"
GROUND_TRUTH_FILE = DATA_DIR / "authoritative_ground_truth.csv"
CLASSIFICATIONS_FILE = DATA_DIR / "llm_classifications.csv"

# ============================================================================
# TRACKING SIGNATURE DATABASE
# These are well-documented JavaScript API patterns associated with tracking
# ============================================================================

FINGERPRINTING_SIGNATURES = {
    # Canvas Fingerprinting - draws shapes/text and extracts pixel data
    "canvas": {
        "apis": [
            "HTMLCanvasElement.getContext",
            "CanvasRenderingContext2D.fillText",
            "CanvasRenderingContext2D.fillRect",
            "CanvasRenderingContext2D.strokeText",
            "CanvasRenderingContext2D.arc",
            "CanvasRenderingContext2D.fill",
            "HTMLCanvasElement.toDataURL",
            "HTMLCanvasElement.toBlob",
            "CanvasRenderingContext2D.getImageData",
        ],
        "description": "Canvas fingerprinting uses HTML5 canvas to draw invisible shapes/text, "
                      "then extracts the rendered pixels. Subtle hardware/driver differences create unique signatures.",
        "severity": "high",
        "references": ["https://browserleaks.com/canvas", "Mowery & Shacham 2012"],
    },
    # WebGL Fingerprinting - GPU/driver enumeration
    "webgl": {
        "apis": [
            "WebGLRenderingContext.getParameter",
            "WebGLRenderingContext.getSupportedExtensions",
            "WebGLRenderingContext.getExtension",
            "WebGLRenderingContext.getShaderPrecisionFormat",
            "WebGL2RenderingContext.getParameter",
            "WEBGL_debug_renderer_info",
        ],
        "description": "WebGL fingerprinting queries GPU renderer/vendor strings and shader precision, "
                      "creating device-specific signatures based on graphics hardware.",
        "severity": "high",
        "references": ["https://browserleaks.com/webgl"],
    },
    # Audio Fingerprinting - AudioContext oscillator output
    "audio": {
        "apis": [
            "AudioContext",
            "OfflineAudioContext",
            "OscillatorNode.frequency",
            "OscillatorNode.type",
            "AudioContext.createOscillator",
            "AudioContext.createDynamicsCompressor",
            "AudioContext.createAnalyser",
            "AnalyserNode.getFloatFrequencyData",
            "AudioBuffer.getChannelData",
        ],
        "description": "Audio fingerprinting generates sound through Web Audio API and measures output variations "
                      "caused by hardware/driver differences.",
        "severity": "high",
        "references": ["Englehardt & Narayanan 2016"],
    },
    # Navigator/Browser Fingerprinting
    "navigator": {
        "apis": [
            "Navigator.userAgent",
            "Navigator.userAgentData",
            "Navigator.platform",
            "Navigator.language",
            "Navigator.languages",
            "Navigator.plugins",
            "Navigator.mimeTypes",
            "Navigator.hardwareConcurrency",
            "Navigator.deviceMemory",
            "Navigator.maxTouchPoints",
            "Navigator.vendor",
            "Navigator.cookieEnabled",
            "Navigator.doNotTrack",
            "Navigator.connection",
            "Navigator.getBattery",
            "Navigator.getGamepads",
            "Navigator.mediaDevices",
            "Navigator.permissions",
            "Navigator.storage",
        ],
        "description": "Collects browser and device properties via Navigator API to build unique profile.",
        "severity": "medium",
        "references": ["https://panopticlick.eff.org/"],
    },
    # Screen Fingerprinting
    "screen": {
        "apis": [
            "Screen.width",
            "Screen.height",
            "Screen.availWidth",
            "Screen.availHeight",
            "Screen.colorDepth",
            "Screen.pixelDepth",
            "Screen.orientation",
            "Window.devicePixelRatio",
            "Window.innerWidth",
            "Window.innerHeight",
            "Window.outerWidth",
            "Window.outerHeight",
            "Window.screenX",
            "Window.screenY",
            "Window.screenLeft",
            "Window.screenTop",
        ],
        "description": "Screen dimensions and display properties help identify device type and configuration.",
        "severity": "medium",
        "references": [],
    },
    # Font Fingerprinting
    "font": {
        "apis": [
            "Document.fonts",
            "FontFaceSet.check",
            "FontFaceSet.load",
            "FontFace",
        ],
        "description": "Detects installed fonts by measuring text rendering differences, highly unique per system.",
        "severity": "high",
        "references": ["Fifield & Egelman 2015"],
    },
    # Storage Fingerprinting / Supercookies
    "storage": {
        "apis": [
            "Window.localStorage",
            "Window.sessionStorage",
            "Storage.getItem",
            "Storage.setItem",
            "Storage.removeItem",
            "Window.indexedDB",
            "IDBFactory.open",
            "IDBObjectStore.put",
            "IDBObjectStore.get",
            "CacheStorage.open",
            "Cache.put",
            "Cache.match",
        ],
        "description": "Uses browser storage APIs to persist tracking identifiers across sessions.",
        "severity": "medium",
        "references": [],
    },
    # Cookie Tracking
    "cookie": {
        "apis": [
            "Document.cookie",
            "HTMLDocument.cookie",
            "CookieStore.get",
            "CookieStore.set",
        ],
        "description": "Traditional cookie-based tracking and session management.",
        "severity": "medium",
        "references": [],
    },
}

BEHAVIORAL_TRACKING_SIGNATURES = {
    # Mouse/Keyboard Tracking
    "input_tracking": {
        "apis": [
            "EventTarget.addEventListener",
            "Window.addEventListener",
            "Document.addEventListener",
            "Element.addEventListener",
            "MouseEvent",
            "KeyboardEvent",
            "TouchEvent",
            "PointerEvent",
            "WheelEvent",
        ],
        "events": ["click", "mousedown", "mouseup", "mousemove", "keydown", "keyup", "touchstart", "touchmove", "scroll"],
        "description": "Captures user input events for behavioral analysis and bot detection.",
        "severity": "medium",
    },
    # Timing/Performance Tracking
    "timing": {
        "apis": [
            "Performance.now",
            "Performance.timing",
            "Performance.getEntries",
            "Performance.getEntriesByType",
            "Performance.getEntriesByName",
            "PerformanceNavigationTiming",
            "PerformanceResourceTiming",
            "PerformanceObserver",
            "Date.now",
            "Window.requestAnimationFrame",
        ],
        "description": "Collects precise timing data for performance analytics and potential fingerprinting.",
        "severity": "low",
    },
    # Visibility Tracking
    "visibility": {
        "apis": [
            "Document.visibilityState",
            "Document.hidden",
            "IntersectionObserver",
            "IntersectionObserverEntry",
            "ResizeObserver",
            "ResizeObserverEntry",
            "MutationObserver",
        ],
        "description": "Tracks page/element visibility for viewability measurement and engagement analytics.",
        "severity": "low",
    },
    # Geolocation
    "geolocation": {
        "apis": [
            "Geolocation.getCurrentPosition",
            "Geolocation.watchPosition",
            "Navigator.geolocation",
        ],
        "description": "Requests precise geographic location - high privacy sensitivity.",
        "severity": "high",
    },
}

NETWORK_TRACKING_SIGNATURES = {
    # Beacon API (fire-and-forget tracking requests)
    "beacon": {
        "apis": [
            "Navigator.sendBeacon",
            "Beacon",
        ],
        "description": "Sends tracking data reliably even during page unload.",
        "severity": "medium",
    },
    # XHR/Fetch for data transmission
    "xhr_fetch": {
        "apis": [
            "XMLHttpRequest.open",
            "XMLHttpRequest.send",
            "XMLHttpRequest.setRequestHeader",
            "Window.fetch",
            "Request",
            "Response",
        ],
        "description": "Network requests potentially transmitting tracking data to servers.",
        "severity": "low",
    },
    # WebSocket for real-time tracking
    "websocket": {
        "apis": [
            "WebSocket",
            "WebSocket.send",
            "WebSocket.close",
        ],
        "description": "Persistent connection for real-time tracking and session monitoring.",
        "severity": "medium",
    },
    # Image/Pixel tracking
    "pixel_tracking": {
        "apis": [
            "HTMLImageElement.src",
            "Image",
        ],
        "description": "Tracking pixels for cross-site user identification.",
        "severity": "medium",
    },
}


def load_mega_features(domain: str) -> Optional[pd.DataFrame]:
    """Load mega_features data for a specific domain."""
    # Try different filename patterns
    safe_domain = domain.replace("https://", "").replace("http://", "").replace("/", "_").replace(":", "_")
    
    patterns = [
        f"mega_features_{safe_domain}.csv",
        f"mega_features_www.{safe_domain}.csv",
        f"mega_features_{safe_domain.replace('www.', '')}.csv",
    ]
    
    for pattern in patterns:
        filepath = DATA_DIR / pattern
        if filepath.exists():
            return pd.read_csv(filepath)
    
    # List available files for debugging
    available = list(DATA_DIR.glob("mega_features_*.csv"))
    print(f"Available mega_features files: {[f.name for f in available]}")
    return None


def load_ground_truth(domain: str) -> Optional[Dict]:
    """Load ground truth data for a domain."""
    if not GROUND_TRUTH_FILE.exists():
        return None
    
    gt_df = pd.read_csv(GROUND_TRUTH_FILE)
    
    # Try to find matching entry
    for _, row in gt_df.iterrows():
        if domain in row["url"] or domain in str(row.get("domain", "")):
            return row.to_dict()
    
    return None


def load_llm_classification(domain: str) -> Optional[Dict]:
    """Load existing LLM classification for a domain."""
    if not CLASSIFICATIONS_FILE.exists():
        return None
    
    class_df = pd.read_csv(CLASSIFICATIONS_FILE)
    
    for _, row in class_df.iterrows():
        if domain in row["url"]:
            return row.to_dict()
    
    return None


class VV8TraceAnalyzer:
    """Deep analyzer for VV8 JavaScript API traces."""
    
    def __init__(self, features_df: pd.DataFrame, domain: str):
        self.features_df = features_df
        self.domain = domain
        self.scripts = self._group_by_script()
        self.feature_summary = self._summarize_features()
        
    def _group_by_script(self) -> Dict[str, pd.DataFrame]:
        """Group features by source script URL."""
        scripts = {}
        if "script_url" in self.features_df.columns:
            for script_url, group in self.features_df.groupby("script_url"):
                scripts[str(script_url)] = group
        return scripts
    
    def _summarize_features(self) -> Dict[str, Dict]:
        """Create feature usage summary."""
        summary = {}
        for _, row in self.features_df.iterrows():
            feature_name = row.get("feature_name", "unknown")
            if feature_name not in summary:
                summary[feature_name] = {
                    "total_count": 0,
                    "scripts": set(),
                    "usage_modes": set(),
                    "receiver": row.get("receiver_name", ""),
                    "member": row.get("member_name", ""),
                }
            summary[feature_name]["total_count"] += row.get("usage_count", 1)
            summary[feature_name]["scripts"].add(str(row.get("script_url", "unknown")))
            summary[feature_name]["usage_modes"].add(str(row.get("usage_mode", "")))
        return summary
    
    def detect_tracking_techniques(self) -> Dict[str, Dict]:
        """Detect tracking techniques based on API signatures."""
        detected = {}
        all_signatures = {
            **FINGERPRINTING_SIGNATURES,
            **BEHAVIORAL_TRACKING_SIGNATURES,
            **NETWORK_TRACKING_SIGNATURES,
        }
        
        for technique_name, signature in all_signatures.items():
            matches = []
            for api in signature["apis"]:
                # Check various matching patterns
                for feature_name, data in self.feature_summary.items():
                    if self._api_matches(feature_name, api):
                        matches.append({
                            "api": feature_name,
                            "pattern": api,
                            "count": data["total_count"],
                            "scripts": list(data["scripts"])[:5],  # Limit to 5 scripts
                        })
            
            if matches:
                detected[technique_name] = {
                    "matches": matches,
                    "total_calls": sum(m["count"] for m in matches),
                    "unique_apis": len(matches),
                    "description": signature["description"],
                    "severity": signature.get("severity", "unknown"),
                }
        
        return detected
    
    def _api_matches(self, feature_name: str, pattern: str) -> bool:
        """Check if a feature name matches an API pattern."""
        feature_lower = feature_name.lower()
        pattern_lower = pattern.lower()
        
        # Exact match
        if feature_lower == pattern_lower:
            return True
        
        # Pattern ends with feature (e.g., "Navigator.userAgent" matches "userAgent")
        if pattern_lower.endswith(feature_lower.split(".")[-1]):
            return True
        
        # Feature contains pattern components
        pattern_parts = pattern_lower.split(".")
        feature_parts = feature_lower.split(".")
        if len(pattern_parts) >= 2 and len(feature_parts) >= 2:
            if pattern_parts[-1] == feature_parts[-1]:  # Same member name
                return True
        
        return False
    
    def analyze_scripts(self) -> List[Dict]:
        """Analyze each script's behavior."""
        script_analyses = []
        
        for script_url, features in self.scripts.items():
            analysis = {
                "url": script_url,
                "is_first_party": self._is_first_party(script_url),
                "is_third_party": not self._is_first_party(script_url),
                "feature_count": len(features),
                "total_api_calls": features["usage_count"].sum() if "usage_count" in features.columns else len(features),
                "top_features": [],
                "suspected_purpose": [],
                "risk_level": "low",
            }
            
            # Get top features for this script
            if "feature_name" in features.columns and "usage_count" in features.columns:
                top = features.nlargest(10, "usage_count")[["feature_name", "usage_count"]]
                analysis["top_features"] = top.to_dict("records")
            
            # Determine suspected purpose
            purposes = self._infer_script_purpose(features)
            analysis["suspected_purpose"] = purposes
            
            # Determine risk level
            analysis["risk_level"] = self._assess_script_risk(features, purposes)
            
            script_analyses.append(analysis)
        
        # Sort by risk and call count
        risk_order = {"high": 0, "medium": 1, "low": 2}
        script_analyses.sort(key=lambda x: (risk_order.get(x["risk_level"], 3), -x["total_api_calls"]))
        
        return script_analyses
    
    def _is_first_party(self, script_url: str) -> bool:
        """Check if script is first-party."""
        if not script_url or script_url == "?" or pd.isna(script_url):
            return True  # Inline scripts are first-party
        
        try:
            parsed = urlparse(script_url)
            script_domain = parsed.netloc.lower()
            site_domain = self.domain.lower().replace("www.", "")
            
            # First party if same domain or subdomain
            return site_domain in script_domain or script_domain in site_domain
        except:
            return True
    
    def _infer_script_purpose(self, features: pd.DataFrame) -> List[str]:
        """Infer the purpose of a script based on its API usage."""
        purposes = set()
        
        if "feature_name" not in features.columns:
            return ["unknown"]
        
        feature_names = features["feature_name"].str.lower().tolist()
        feature_text = " ".join(feature_names)
        
        # Check for fingerprinting patterns
        if any(f in feature_text for f in ["canvas", "webgl", "audicontext", "getparameter"]):
            purposes.add("fingerprinting")
        
        # Check for analytics patterns
        if any(f in feature_text for f in ["performance", "timing", "navigation", "beacon"]):
            purposes.add("analytics")
        
        # Check for advertising patterns
        if any(f in feature_text for f in ["ad", "pixel", "track", "doubleclick", "googletag"]):
            purposes.add("advertising")
        
        # Check for storage patterns
        if any(f in feature_text for f in ["cookie", "localstorage", "indexeddb", "storage"]):
            purposes.add("persistence")
        
        # Check for DOM manipulation (often benign)
        if any(f in feature_text for f in ["queryselector", "createelement", "appendchild", "classlist"]):
            purposes.add("dom_manipulation")
        
        # Check for event handling
        if any(f in feature_text for f in ["addeventlistener", "removeeventlistener", "event"]):
            purposes.add("event_handling")
        
        return list(purposes) if purposes else ["general"]
    
    def _assess_script_risk(self, features: pd.DataFrame, purposes: List[str]) -> str:
        """Assess risk level of a script."""
        high_risk_purposes = {"fingerprinting"}
        medium_risk_purposes = {"analytics", "advertising", "persistence"}
        
        if high_risk_purposes & set(purposes):
            return "high"
        elif medium_risk_purposes & set(purposes):
            return "medium"
        return "low"
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the trace."""
        return {
            "total_features": len(self.feature_summary),
            "total_api_calls": sum(f["total_count"] for f in self.feature_summary.values()),
            "unique_scripts": len(self.scripts),
            "first_party_scripts": sum(1 for s in self.scripts if self._is_first_party(s)),
            "third_party_scripts": sum(1 for s in self.scripts if not self._is_first_party(s)),
            "top_features_by_usage": sorted(
                [(k, v["total_count"]) for k, v in self.feature_summary.items()],
                key=lambda x: x[1],
                reverse=True
            )[:20],
        }


class LLMAccuracyValidator:
    """Validates LLM interpretation accuracy against known patterns and ground truth."""
    
    def __init__(self, analyzer: VV8TraceAnalyzer, llm_classification: Optional[Dict], ground_truth: Optional[Dict]):
        self.analyzer = analyzer
        self.llm_classification = llm_classification or {}
        self.ground_truth = ground_truth or {}
        self.validation_results = {}
        
    def validate(self) -> Dict:
        """Run full validation suite."""
        self.validation_results = {
            "heuristic_validation": self._validate_against_heuristics(),
            "ground_truth_validation": self._validate_against_ground_truth(),
            "consistency_check": self._check_internal_consistency(),
            "false_positive_analysis": self._analyze_false_positives(),
            "false_negative_analysis": self._analyze_false_negatives(),
            "confidence_assessment": self._assess_confidence(),
            "overall_accuracy_score": 0.0,
        }
        
        # Calculate overall score
        self.validation_results["overall_accuracy_score"] = self._calculate_overall_score()
        
        return self.validation_results
    
    def _validate_against_heuristics(self) -> Dict:
        """Validate LLM classification against signature-based heuristics."""
        detected_techniques = self.analyzer.detect_tracking_techniques()
        llm_category = self.llm_classification.get("primary_category", "").lower()
        llm_indicators = str(self.llm_classification.get("key_indicators", "")).lower()
        
        validations = []
        agreements = 0
        disagreements = 0
        
        # Check if LLM detected the same techniques we found
        for technique, data in detected_techniques.items():
            technique_found_by_llm = (
                technique in llm_category or 
                technique in llm_indicators or
                any(m["api"].lower() in llm_indicators for m in data["matches"][:3])
            )
            
            validation = {
                "technique": technique,
                "detected_by_heuristics": True,
                "detected_by_llm": technique_found_by_llm,
                "api_evidence": [m["api"] for m in data["matches"][:5]],
                "call_count": data["total_calls"],
                "severity": data["severity"],
                "agreement": technique_found_by_llm,
            }
            
            if technique_found_by_llm:
                agreements += 1
            else:
                disagreements += 1
            
            validations.append(validation)
        
        return {
            "validations": validations,
            "agreements": agreements,
            "disagreements": disagreements,
            "agreement_rate": agreements / (agreements + disagreements) if (agreements + disagreements) > 0 else 1.0,
            "missed_techniques": [v["technique"] for v in validations if not v["detected_by_llm"]],
        }
    
    def _validate_against_ground_truth(self) -> Dict:
        """Validate against authoritative ground truth databases."""
        if not self.ground_truth:
            return {"status": "no_ground_truth_available", "validation": None}
        
        gt_is_tracker = self.ground_truth.get("is_known_tracker", False)
        gt_category = self.ground_truth.get("tracker_category", "")
        gt_sources = self.ground_truth.get("tracker_sources", "")
        
        llm_category = self.llm_classification.get("primary_category", "").lower()
        llm_is_tracker = llm_category in ["advertising", "analytics", "fingerprinting", "tracking", "social"]
        
        # Check agreement
        tracker_agreement = gt_is_tracker == llm_is_tracker
        
        category_agreement = False
        if gt_category and llm_category:
            # Fuzzy category matching
            category_mappings = {
                "advertising": ["advertising", "ad", "marketing"],
                "analytics": ["analytics", "tracking", "measurement"],
                "fingerprinting": ["fingerprinting", "device identification"],
                "tracking": ["tracking", "analytics", "advertising", "fingerprinting"],
            }
            gt_cat_lower = gt_category.lower()
            for key, values in category_mappings.items():
                if gt_cat_lower in values and llm_category in values:
                    category_agreement = True
                    break
        
        return {
            "status": "validated",
            "ground_truth": {
                "is_tracker": gt_is_tracker,
                "category": gt_category,
                "sources": gt_sources,
            },
            "llm_classification": {
                "is_tracker": llm_is_tracker,
                "category": llm_category,
            },
            "tracker_agreement": tracker_agreement,
            "category_agreement": category_agreement,
            "validation_result": "correct" if tracker_agreement else ("false_positive" if llm_is_tracker else "false_negative"),
        }
    
    def _check_internal_consistency(self) -> Dict:
        """Check if LLM's classification is internally consistent with its explanation."""
        inconsistencies = []
        
        llm_category = self.llm_classification.get("primary_category", "").lower()
        llm_explanation = str(self.llm_classification.get("classification_explanation", "")).lower()
        llm_indicators = str(self.llm_classification.get("key_indicators", "")).lower()
        llm_risk = self.llm_classification.get("privacy_risk", "").lower()
        llm_intensity = self.llm_classification.get("tracking_intensity", "").lower()
        
        # Check category vs explanation consistency
        if llm_category == "benign" and any(w in llm_explanation for w in ["fingerprint", "track", "advertis"]):
            inconsistencies.append({
                "type": "category_explanation_mismatch",
                "description": "Classified as benign but explanation mentions tracking behaviors",
            })
        
        if llm_category in ["fingerprinting", "advertising"] and "benign" in llm_explanation:
            inconsistencies.append({
                "type": "category_explanation_mismatch",
                "description": "Classified as tracking but explanation suggests benign behavior",
            })
        
        # Check risk vs category consistency
        if llm_risk == "high" and llm_category == "benign":
            inconsistencies.append({
                "type": "risk_category_mismatch",
                "description": "High privacy risk but benign classification",
            })
        
        if llm_risk == "low" and llm_category == "fingerprinting":
            inconsistencies.append({
                "type": "risk_category_mismatch",
                "description": "Low privacy risk for fingerprinting classification",
            })
        
        # Check intensity vs evidence
        stats = self.analyzer.get_statistics()
        if llm_intensity == "heavy" and stats["total_api_calls"] < 100:
            inconsistencies.append({
                "type": "intensity_evidence_mismatch",
                "description": f"Heavy tracking claimed but only {stats['total_api_calls']} API calls",
            })
        
        return {
            "is_consistent": len(inconsistencies) == 0,
            "inconsistencies": inconsistencies,
            "consistency_score": 1.0 - (len(inconsistencies) * 0.2),  # Deduct 20% per inconsistency
        }
    
    def _analyze_false_positives(self) -> Dict:
        """Analyze potential false positive detections by LLM."""
        potential_fps = []
        detected = self.analyzer.detect_tracking_techniques()
        llm_indicators = str(self.llm_classification.get("key_indicators", ""))
        
        # Check if LLM claims techniques not supported by evidence
        claimed_techniques = []
        if "fingerprint" in llm_indicators.lower():
            claimed_techniques.append("fingerprinting")
        if "canvas" in llm_indicators.lower():
            claimed_techniques.append("canvas")
        if "webgl" in llm_indicators.lower():
            claimed_techniques.append("webgl")
        
        for technique in claimed_techniques:
            if technique not in detected and not any(technique in k for k in detected.keys()):
                potential_fps.append({
                    "claimed_technique": technique,
                    "evidence_found": False,
                    "possible_reason": "LLM may have over-interpreted ambiguous API patterns",
                })
        
        return {
            "potential_false_positives": potential_fps,
            "count": len(potential_fps),
        }
    
    def _analyze_false_negatives(self) -> Dict:
        """Analyze potential false negative detections by LLM."""
        potential_fns = []
        detected = self.analyzer.detect_tracking_techniques()
        llm_category = self.llm_classification.get("primary_category", "").lower()
        llm_indicators = str(self.llm_classification.get("key_indicators", "")).lower()
        
        # Check for high-severity techniques not mentioned by LLM
        high_severity_detected = {k: v for k, v in detected.items() if v["severity"] == "high"}
        
        for technique, data in high_severity_detected.items():
            if technique not in llm_indicators and technique not in llm_category:
                potential_fns.append({
                    "missed_technique": technique,
                    "severity": "high",
                    "evidence": [m["api"] for m in data["matches"][:3]],
                    "call_count": data["total_calls"],
                    "possible_reason": "Technique may have been overlooked or deprioritized",
                })
        
        return {
            "potential_false_negatives": potential_fns,
            "count": len(potential_fns),
        }
    
    def _assess_confidence(self) -> Dict:
        """Assess confidence in the overall validation."""
        confidence_factors = []
        
        # Factor 1: Amount of data
        stats = self.analyzer.get_statistics()
        if stats["total_api_calls"] > 1000:
            confidence_factors.append(("data_volume", 1.0, "Sufficient API call data"))
        elif stats["total_api_calls"] > 100:
            confidence_factors.append(("data_volume", 0.8, "Moderate API call data"))
        else:
            confidence_factors.append(("data_volume", 0.5, "Limited API call data"))
        
        # Factor 2: Ground truth availability
        if self.ground_truth and self.ground_truth.get("is_known_tracker") is not None:
            confidence_factors.append(("ground_truth", 1.0, "Ground truth available"))
        else:
            confidence_factors.append(("ground_truth", 0.6, "No ground truth available"))
        
        # Factor 3: LLM classification completeness
        if self.llm_classification.get("primary_category") and self.llm_classification.get("key_indicators"):
            confidence_factors.append(("llm_completeness", 1.0, "Complete LLM classification"))
        else:
            confidence_factors.append(("llm_completeness", 0.5, "Incomplete LLM classification"))
        
        # Factor 4: Clear signature matches
        detected = self.analyzer.detect_tracking_techniques()
        if len(detected) > 0:
            confidence_factors.append(("signature_matches", 0.9, f"Found {len(detected)} tracking signatures"))
        else:
            confidence_factors.append(("signature_matches", 0.7, "No clear tracking signatures"))
        
        overall_confidence = sum(f[1] for f in confidence_factors) / len(confidence_factors)
        
        return {
            "factors": confidence_factors,
            "overall_confidence": overall_confidence,
            "confidence_level": "high" if overall_confidence > 0.8 else ("medium" if overall_confidence > 0.6 else "low"),
        }
    
    def _calculate_overall_score(self) -> float:
        """Calculate overall accuracy score."""
        scores = []
        
        # Heuristic agreement
        heuristic = self.validation_results.get("heuristic_validation", {})
        scores.append(heuristic.get("agreement_rate", 0.5) * 0.3)  # 30% weight
        
        # Ground truth agreement
        gt = self.validation_results.get("ground_truth_validation", {})
        if gt.get("status") == "validated":
            gt_score = 1.0 if gt.get("tracker_agreement") else 0.0
            gt_score += 0.5 if gt.get("category_agreement") else 0.0
            scores.append(min(gt_score, 1.0) * 0.35)  # 35% weight
        else:
            scores.append(0.5 * 0.35)  # Neutral if no ground truth
        
        # Consistency
        consistency = self.validation_results.get("consistency_check", {})
        scores.append(consistency.get("consistency_score", 0.5) * 0.15)  # 15% weight
        
        # False positive/negative penalty
        fp = self.validation_results.get("false_positive_analysis", {})
        fn = self.validation_results.get("false_negative_analysis", {})
        error_penalty = (fp.get("count", 0) + fn.get("count", 0)) * 0.05
        scores.append(max(0, 0.2 - error_penalty))  # 20% weight minus penalties
        
        return min(sum(scores), 1.0)


class CaseStudyGenerator:
    """Generates comprehensive case study reports."""
    
    def __init__(self, domain: str, api_key: Optional[str] = None):
        self.domain = domain
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.features_df = None
        self.analyzer = None
        self.validator = None
        self.llm_classification = None
        self.ground_truth = None
        
    def load_data(self) -> bool:
        """Load all required data."""
        print(f"Loading data for {self.domain}...")
        
        # Load features
        self.features_df = load_mega_features(self.domain)
        if self.features_df is None:
            print(f"❌ Could not load mega_features for {self.domain}")
            return False
        
        print(f"  ✓ Loaded {len(self.features_df)} feature entries")
        
        # Load existing classification
        self.llm_classification = load_llm_classification(self.domain)
        if self.llm_classification:
            print(f"  ✓ Loaded existing LLM classification")
        else:
            print(f"  ⚠ No existing LLM classification found")
        
        # Load ground truth
        self.ground_truth = load_ground_truth(self.domain)
        if self.ground_truth:
            print(f"  ✓ Loaded ground truth data")
        else:
            print(f"  ⚠ No ground truth data found")
        
        # Initialize analyzer
        self.analyzer = VV8TraceAnalyzer(self.features_df, self.domain)
        
        # Initialize validator
        self.validator = LLMAccuracyValidator(
            self.analyzer, 
            self.llm_classification, 
            self.ground_truth
        )
        
        return True
    
    def generate_fresh_llm_analysis(self) -> Optional[Dict]:
        """Generate a fresh LLM analysis for comparison."""
        if not self.api_key:
            print("⚠ No API key available for fresh LLM analysis")
            return None
        
        print("\n🤖 Generating fresh LLM analysis...")
        
        client = anthropic.Anthropic(api_key=self.api_key)
        
        # Prepare comprehensive feature data
        stats = self.analyzer.get_statistics()
        detected = self.analyzer.detect_tracking_techniques()
        scripts = self.analyzer.analyze_scripts()
        
        # Build detailed prompt
        prompt = self._build_analysis_prompt(stats, detected, scripts)
        
        try:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=4096,
                system="""You are an expert web privacy researcher specializing in JavaScript tracking analysis.
Your task is to analyze VV8 (VisibleV8) JavaScript API traces and provide a detailed, accurate classification.
Be precise and evidence-based. Cite specific APIs when making claims about tracking techniques.
Always respond in valid JSON format.""",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            
            response_text = response.content[0].text
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(response_text)
            
        except Exception as e:
            print(f"❌ Error during LLM analysis: {e}")
            return None
    
    def _build_analysis_prompt(self, stats: Dict, detected: Dict, scripts: List[Dict]) -> str:
        """Build comprehensive analysis prompt for LLM."""
        
        # Format detected techniques
        techniques_text = []
        for technique, data in detected.items():
            techniques_text.append(f"\n### {technique.upper()} ({data['severity']} severity)")
            techniques_text.append(f"   Total API calls: {data['total_calls']}")
            techniques_text.append(f"   APIs detected: {', '.join(m['api'] for m in data['matches'][:5])}")
        
        # Format top features
        top_features_text = "\n".join([
            f"  - {feat[0]}: {feat[1]} calls" 
            for feat in stats["top_features_by_usage"][:15]
        ])
        
        # Format script analysis
        scripts_text = []
        for script in scripts[:10]:  # Top 10 scripts
            party = "1st-party" if script["is_first_party"] else "3rd-party"
            scripts_text.append(
                f"\n  - [{party}] {script['url'][:80]}..."
                f"\n    Purpose: {', '.join(script['suspected_purpose'])}"
                f"\n    Risk: {script['risk_level']}, Calls: {script['total_api_calls']}"
            )
        
        return f"""Analyze the following VV8 JavaScript API traces from {self.domain} and provide a detailed classification.

=== TRACE STATISTICS ===
- Total unique features: {stats['total_features']}
- Total API calls: {stats['total_api_calls']}
- Unique scripts: {stats['unique_scripts']}
- First-party scripts: {stats['first_party_scripts']}
- Third-party scripts: {stats['third_party_scripts']}

=== TOP 15 MOST-USED APIs ===
{top_features_text}

=== DETECTED TRACKING TECHNIQUES (by signature matching) ===
{''.join(techniques_text) if techniques_text else 'No clear tracking signatures detected'}

=== SCRIPT ANALYSIS (top 10 by risk/volume) ===
{''.join(scripts_text)}

=== YOUR TASK ===
Provide a comprehensive analysis in the following JSON format:
{{
    "primary_category": "one of: fingerprinting, analytics, advertising, social, benign",
    "secondary_categories": ["list of other applicable categories"],
    "confidence": "high, medium, or low",
    "privacy_risk": "low, medium, or high",
    "tracking_intensity": "none, light, moderate, or heavy",
    
    "detected_techniques": {{
        "fingerprinting": {{
            "present": true/false,
            "techniques": ["list of specific techniques, e.g., 'canvas', 'webgl'"],
            "evidence": ["specific APIs that prove this"]
        }},
        "behavioral_tracking": {{
            "present": true/false,
            "techniques": ["e.g., 'mouse tracking', 'scroll tracking'"],
            "evidence": ["specific APIs"]
        }},
        "data_collection": {{
            "present": true/false,
            "methods": ["e.g., 'cookies', 'localStorage', 'beacons'"],
            "evidence": ["specific APIs"]
        }}
    }},
    
    "script_assessment": {{
        "most_suspicious_scripts": ["URLs of scripts with highest tracking risk"],
        "third_party_risk": "assessment of third-party script risk"
    }},
    
    "classification_explanation": "Detailed 3-5 sentence explanation of your classification, citing specific evidence",
    "key_indicators": ["List 5-10 specific APIs/features that most influenced your classification"],
    
    "uncertainty_notes": ["Any aspects where you are uncertain or where evidence is ambiguous"],
    "alternative_interpretations": ["Possible alternative explanations for the observed behavior"]
}}

Be thorough and precise. If you detect fingerprinting, specify exactly which type (canvas, WebGL, audio, etc.).
If you're uncertain about something, say so explicitly."""
    
    def generate_case_study(self, include_fresh_analysis: bool = True) -> Dict:
        """Generate the complete case study."""
        print(f"\n{'='*70}")
        print(f"GENERATING CASE STUDY FOR: {self.domain}")
        print(f"{'='*70}\n")
        
        case_study = {
            "metadata": {
                "domain": self.domain,
                "generated_at": datetime.now().isoformat(),
                "vv8_data_available": self.features_df is not None,
                "existing_classification_available": self.llm_classification is not None,
                "ground_truth_available": self.ground_truth is not None,
            },
            "trace_statistics": {},
            "detected_tracking_techniques": {},
            "script_analysis": [],
            "existing_llm_classification": self.llm_classification,
            "ground_truth": self.ground_truth,
            "fresh_llm_analysis": None,
            "validation_results": {},
            "accuracy_assessment": {},
            "recommendations": [],
        }
        
        if self.analyzer:
            # Get statistics
            case_study["trace_statistics"] = self.analyzer.get_statistics()
            print(f"📊 Trace Statistics:")
            print(f"   - Total features: {case_study['trace_statistics']['total_features']}")
            print(f"   - Total API calls: {case_study['trace_statistics']['total_api_calls']}")
            print(f"   - Scripts: {case_study['trace_statistics']['unique_scripts']}")
            
            # Detect tracking techniques
            case_study["detected_tracking_techniques"] = self.analyzer.detect_tracking_techniques()
            print(f"\n🔍 Detected Tracking Techniques:")
            for technique, data in case_study["detected_tracking_techniques"].items():
                print(f"   - {technique}: {data['total_calls']} calls ({data['severity']} severity)")
            
            # Analyze scripts
            case_study["script_analysis"] = self.analyzer.analyze_scripts()
            print(f"\n📜 Analyzed {len(case_study['script_analysis'])} scripts")
        
        # Generate fresh LLM analysis if requested
        if include_fresh_analysis:
            fresh_analysis = self.generate_fresh_llm_analysis()
            case_study["fresh_llm_analysis"] = fresh_analysis
            if fresh_analysis:
                print(f"\n✅ Fresh LLM analysis complete")
                print(f"   Category: {fresh_analysis.get('primary_category')}")
                print(f"   Confidence: {fresh_analysis.get('confidence')}")
        
        # Run validation
        if self.validator:
            print(f"\n🔬 Running validation...")
            case_study["validation_results"] = self.validator.validate()
            
            print(f"   - Heuristic agreement: {case_study['validation_results']['heuristic_validation']['agreement_rate']:.1%}")
            
            gt_validation = case_study["validation_results"]["ground_truth_validation"]
            if gt_validation.get("status") == "validated":
                print(f"   - Ground truth agreement: {gt_validation['tracker_agreement']}")
            
            print(f"   - Consistency score: {case_study['validation_results']['consistency_check']['consistency_score']:.1%}")
            print(f"   - Overall accuracy: {case_study['validation_results']['overall_accuracy_score']:.1%}")
        
        # Generate accuracy assessment
        case_study["accuracy_assessment"] = self._generate_accuracy_assessment(case_study)
        
        # Generate recommendations
        case_study["recommendations"] = self._generate_recommendations(case_study)
        
        return case_study
    
    def _generate_accuracy_assessment(self, case_study: Dict) -> Dict:
        """Generate overall accuracy assessment."""
        validation = case_study.get("validation_results", {})
        
        assessment = {
            "overall_accuracy_grade": "",
            "accuracy_score": validation.get("overall_accuracy_score", 0),
            "strengths": [],
            "weaknesses": [],
            "key_findings": [],
        }
        
        score = assessment["accuracy_score"]
        if score >= 0.9:
            assessment["overall_accuracy_grade"] = "A (Excellent)"
        elif score >= 0.8:
            assessment["overall_accuracy_grade"] = "B (Good)"
        elif score >= 0.7:
            assessment["overall_accuracy_grade"] = "C (Acceptable)"
        elif score >= 0.6:
            assessment["overall_accuracy_grade"] = "D (Needs Improvement)"
        else:
            assessment["overall_accuracy_grade"] = "F (Poor)"
        
        # Identify strengths
        heuristic = validation.get("heuristic_validation", {})
        if heuristic.get("agreement_rate", 0) > 0.8:
            assessment["strengths"].append("Strong agreement with signature-based heuristics")
        
        consistency = validation.get("consistency_check", {})
        if consistency.get("is_consistent"):
            assessment["strengths"].append("Internal consistency in classification")
        
        gt = validation.get("ground_truth_validation", {})
        if gt.get("tracker_agreement"):
            assessment["strengths"].append("Correct tracker/non-tracker classification")
        
        # Identify weaknesses
        if heuristic.get("missed_techniques"):
            assessment["weaknesses"].append(
                f"Missed techniques: {', '.join(heuristic['missed_techniques'][:3])}"
            )
        
        fp = validation.get("false_positive_analysis", {})
        if fp.get("count", 0) > 0:
            assessment["weaknesses"].append(
                f"Potential false positives: {fp['count']}"
            )
        
        fn = validation.get("false_negative_analysis", {})
        if fn.get("count", 0) > 0:
            assessment["weaknesses"].append(
                f"Potential false negatives: {fn['count']}"
            )
        
        # Key findings
        techniques = case_study.get("detected_tracking_techniques", {})
        if techniques:
            high_severity = [k for k, v in techniques.items() if v["severity"] == "high"]
            if high_severity:
                assessment["key_findings"].append(
                    f"High-severity tracking detected: {', '.join(high_severity)}"
                )
        
        return assessment
    
    def _generate_recommendations(self, case_study: Dict) -> List[str]:
        """Generate recommendations for improving classification accuracy."""
        recommendations = []
        validation = case_study.get("validation_results", {})
        
        # Check for missed techniques
        heuristic = validation.get("heuristic_validation", {})
        if heuristic.get("missed_techniques"):
            recommendations.append(
                "Consider adding explicit detection for: " + 
                ", ".join(heuristic["missed_techniques"])
            )
        
        # Check consistency
        consistency = validation.get("consistency_check", {})
        if not consistency.get("is_consistent"):
            recommendations.append(
                "Review classification logic for internal consistency"
            )
        
        # Check for false negatives
        fn = validation.get("false_negative_analysis", {})
        if fn.get("potential_false_negatives"):
            for fn_item in fn["potential_false_negatives"][:2]:
                recommendations.append(
                    f"High-severity technique '{fn_item['missed_technique']}' was not prominently identified"
                )
        
        # Check confidence
        confidence = validation.get("confidence_assessment", {})
        if confidence.get("overall_confidence", 1.0) < 0.7:
            recommendations.append(
                "Low validation confidence - consider gathering more data or improving ground truth coverage"
            )
        
        if not recommendations:
            recommendations.append("Classification appears accurate - no specific improvements needed")
        
        return recommendations
    
    def save_case_study(self, case_study: Dict, output_dir: Optional[Path] = None, 
                        format: str = "both") -> List[Path]:
        """Save case study to files."""
        output_dir = output_dir or OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        safe_domain = self.domain.replace(".", "_").replace("/", "_")
        
        saved_files = []
        
        # Save JSON version
        if format in ["json", "both"]:
            json_path = output_dir / f"case_study_{safe_domain}.json"
            with open(json_path, "w") as f:
                # Convert sets to lists for JSON serialization
                json.dump(self._prepare_for_json(case_study), f, indent=2, default=str)
            saved_files.append(json_path)
            print(f"\n💾 Saved JSON: {json_path}")
        
        # Save Markdown version
        if format in ["markdown", "both"]:
            md_path = output_dir / f"case_study_{safe_domain}.md"
            with open(md_path, "w") as f:
                f.write(self._generate_markdown_report(case_study))
            saved_files.append(md_path)
            print(f"💾 Saved Markdown: {md_path}")
        
        return saved_files
    
    def _prepare_for_json(self, obj: Any) -> Any:
        """Prepare object for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(v) for v in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        elif pd.isna(obj):
            return None
        return obj
    
    def _generate_markdown_report(self, case_study: Dict) -> str:
        """Generate a detailed Markdown report."""
        md = []
        
        # Header
        md.append(f"# Case Study: {self.domain}")
        md.append(f"\n**Generated:** {case_study['metadata']['generated_at']}")
        md.append(f"\n---\n")
        
        # Executive Summary
        md.append("## Executive Summary\n")
        assessment = case_study.get("accuracy_assessment", {})
        md.append(f"**Overall Accuracy Grade:** {assessment.get('overall_accuracy_grade', 'N/A')}")
        md.append(f"\n**Accuracy Score:** {assessment.get('accuracy_score', 0):.1%}\n")
        
        if assessment.get("key_findings"):
            md.append("\n### Key Findings")
            for finding in assessment["key_findings"]:
                md.append(f"- {finding}")
        
        md.append("\n---\n")
        
        # Trace Statistics
        md.append("## VV8 Trace Statistics\n")
        stats = case_study.get("trace_statistics", {})
        md.append(f"| Metric | Value |")
        md.append(f"|--------|-------|")
        md.append(f"| Total Unique Features | {stats.get('total_features', 0):,} |")
        md.append(f"| Total API Calls | {stats.get('total_api_calls', 0):,} |")
        md.append(f"| Unique Scripts | {stats.get('unique_scripts', 0)} |")
        md.append(f"| First-Party Scripts | {stats.get('first_party_scripts', 0)} |")
        md.append(f"| Third-Party Scripts | {stats.get('third_party_scripts', 0)} |")
        
        md.append("\n### Top 10 Most-Used APIs\n")
        md.append("| API | Call Count |")
        md.append("|-----|------------|")
        for api, count in stats.get("top_features_by_usage", [])[:10]:
            md.append(f"| `{api}` | {count:,} |")
        
        md.append("\n---\n")
        
        # Detected Tracking Techniques
        md.append("## Detected Tracking Techniques\n")
        techniques = case_study.get("detected_tracking_techniques", {})
        if techniques:
            for technique, data in techniques.items():
                severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(data["severity"], "⚪")
                md.append(f"\n### {severity_emoji} {technique.replace('_', ' ').title()}")
                md.append(f"\n**Severity:** {data['severity'].upper()}")
                md.append(f"\n**Total API Calls:** {data['total_calls']:,}")
                md.append(f"\n**Description:** {data['description']}")
                md.append(f"\n\n**Evidence APIs:**")
                for match in data["matches"][:5]:
                    md.append(f"- `{match['api']}` ({match['count']} calls)")
        else:
            md.append("*No clear tracking techniques detected via signature matching.*")
        
        md.append("\n---\n")
        
        # LLM Classification Comparison
        md.append("## LLM Classification Analysis\n")
        
        existing = case_study.get("existing_llm_classification")
        fresh = case_study.get("fresh_llm_analysis")
        
        if existing:
            md.append("### Existing Classification\n")
            md.append(f"- **Primary Category:** {existing.get('primary_category', 'N/A')}")
            md.append(f"- **Confidence:** {existing.get('confidence', 'N/A')}")
            md.append(f"- **Privacy Risk:** {existing.get('privacy_risk', 'N/A')}")
            md.append(f"- **Tracking Intensity:** {existing.get('tracking_intensity', 'N/A')}")
            if existing.get("classification_explanation"):
                md.append(f"\n**Explanation:** {existing['classification_explanation']}")
        
        if fresh:
            md.append("\n### Fresh Analysis (for comparison)\n")
            md.append(f"- **Primary Category:** {fresh.get('primary_category', 'N/A')}")
            md.append(f"- **Confidence:** {fresh.get('confidence', 'N/A')}")
            md.append(f"- **Privacy Risk:** {fresh.get('privacy_risk', 'N/A')}")
            if fresh.get("classification_explanation"):
                md.append(f"\n**Explanation:** {fresh['classification_explanation']}")
            if fresh.get("uncertainty_notes"):
                md.append(f"\n**Uncertainties:** {', '.join(fresh['uncertainty_notes'])}")
        
        md.append("\n---\n")
        
        # Validation Results
        md.append("## Validation Results\n")
        validation = case_study.get("validation_results", {})
        
        # Heuristic validation
        heuristic = validation.get("heuristic_validation", {})
        md.append("### Heuristic Validation\n")
        md.append(f"**Agreement Rate:** {heuristic.get('agreement_rate', 0):.1%}\n")
        if heuristic.get("missed_techniques"):
            md.append(f"**Missed Techniques:** {', '.join(heuristic['missed_techniques'])}\n")
        
        # Ground truth validation
        gt = validation.get("ground_truth_validation", {})
        md.append("\n### Ground Truth Validation\n")
        if gt.get("status") == "validated":
            md.append(f"- **Ground Truth Source:** {gt['ground_truth'].get('sources', 'N/A')}")
            md.append(f"- **Is Known Tracker:** {gt['ground_truth'].get('is_tracker', 'N/A')}")
            md.append(f"- **Tracker Agreement:** {'✅ Yes' if gt.get('tracker_agreement') else '❌ No'}")
            md.append(f"- **Category Agreement:** {'✅ Yes' if gt.get('category_agreement') else '❌ No'}")
            md.append(f"- **Validation Result:** {gt.get('validation_result', 'N/A')}")
        else:
            md.append("*No ground truth available for validation.*")
        
        # Consistency check
        consistency = validation.get("consistency_check", {})
        md.append("\n### Internal Consistency\n")
        md.append(f"**Consistent:** {'✅ Yes' if consistency.get('is_consistent') else '❌ No'}")
        md.append(f"\n**Consistency Score:** {consistency.get('consistency_score', 0):.1%}\n")
        if consistency.get("inconsistencies"):
            md.append("\n**Inconsistencies Found:**")
            for inc in consistency["inconsistencies"]:
                md.append(f"- {inc['type']}: {inc['description']}")
        
        # False positive/negative analysis
        fp = validation.get("false_positive_analysis", {})
        fn = validation.get("false_negative_analysis", {})
        
        if fp.get("potential_false_positives") or fn.get("potential_false_negatives"):
            md.append("\n### Error Analysis\n")
            if fp.get("potential_false_positives"):
                md.append(f"\n**Potential False Positives ({fp['count']}):**")
                for item in fp["potential_false_positives"]:
                    md.append(f"- {item['claimed_technique']}: {item['possible_reason']}")
            if fn.get("potential_false_negatives"):
                md.append(f"\n**Potential False Negatives ({fn['count']}):**")
                for item in fn["potential_false_negatives"]:
                    md.append(f"- {item['missed_technique']} ({item['severity']} severity)")
        
        md.append("\n---\n")
        
        # Accuracy Assessment
        md.append("## Accuracy Assessment\n")
        md.append(f"### Overall Grade: {assessment.get('overall_accuracy_grade', 'N/A')}\n")
        
        if assessment.get("strengths"):
            md.append("\n**Strengths:**")
            for s in assessment["strengths"]:
                md.append(f"- ✅ {s}")
        
        if assessment.get("weaknesses"):
            md.append("\n**Weaknesses:**")
            for w in assessment["weaknesses"]:
                md.append(f"- ⚠️ {w}")
        
        md.append("\n---\n")
        
        # Recommendations
        md.append("## Recommendations\n")
        for rec in case_study.get("recommendations", []):
            md.append(f"- {rec}")
        
        md.append("\n---\n")
        md.append(f"*Report generated by VV8 Case Study Generator*")
        
        return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive case study for a single website's VV8 traces"
    )
    parser.add_argument(
        "domain",
        help="Domain to analyze (e.g., www.example.com)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for case study files"
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "both"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    parser.add_argument(
        "--fresh-analysis",
        action="store_true",
        help="Generate fresh LLM analysis (default: uses existing classification only)"
    )
    parser.add_argument(
        "--api-key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = CaseStudyGenerator(args.domain, api_key=args.api_key)
    
    # Load data
    if not generator.load_data():
        print("\n❌ Failed to load data. Cannot generate case study.")
        sys.exit(1)
    
    # Generate case study
    case_study = generator.generate_case_study(
        include_fresh_analysis=args.fresh_analysis
    )
    
    # Save results
    saved_files = generator.save_case_study(
        case_study, 
        output_dir=args.output_dir,
        format=args.format
    )
    
    print(f"\n{'='*70}")
    print("CASE STUDY GENERATION COMPLETE")
    print(f"{'='*70}")
    
    assessment = case_study.get("accuracy_assessment", {})
    print(f"\n📊 Overall Accuracy Grade: {assessment.get('overall_accuracy_grade', 'N/A')}")
    print(f"📈 Accuracy Score: {assessment.get('accuracy_score', 0):.1%}")
    
    print(f"\n📁 Output Files:")
    for f in saved_files:
        print(f"   - {f}")


if __name__ == "__main__":
    main()
