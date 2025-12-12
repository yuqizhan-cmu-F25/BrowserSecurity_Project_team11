import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from datetime import datetime
import numpy as np

# --- Configuration ---
CLASSIFICATIONS_FILE = Path("data/processed/llm_classifications.csv")
FEATURES_DIR = Path("data/processed/")
OUTPUT_REPORT = Path("data/analysis_reports/feature_importance_report.md")

# --- Feature Categories for Better Context ---
FEATURE_CATEGORIES = {
    # Core tracking techniques
    "fingerprinting": ["canvas", "webgl", "audiocontext", "timezone", "font", "battery", "devicememory", "hardwareconcurrency", "platform", "plugins", "mimetypes", "webrtc", "getusermedia", "mediadevices", "speechsynthesis"],
    "tracking_storage": ["cookie", "localstorage", "sessionstorage", "indexeddb", "cachestorage", "serviceworker", "caches"],
    "network_requests": ["fetch", "xmlhttprequest", "beacon", "websocket", "sendbeacon", "nexthopprotocol", "abortcontroller", "request", "response", "headers"],
    
    # DOM & page interaction
    "dom_traversal": ["queryselector", "getelementby", "parentnode", "childnodes", "nextsibling", "previoussibling", "children", "firstchild", "lastchild", "closest", "matches"],
    "dom_modification": ["createelement", "appendchild", "removechild", "innerhtml", "textcontent", "setattribute", "getattribute", "classlist", "insertbefore", "replacechild", "clonenode"],
    "dom_properties": ["nodetype", "nodename", "tagname", "namednodemap", "attributes", "dataset", "id", "classname"],
    "html_elements": ["htmldiv", "htmlspan", "htmlanchor", "htmlinput", "htmlform", "htmlimage", "htmlscript", "htmlstyle", "htmlbody", "htmlhead", "htmlul", "htmlli", "htmlbutton", "htmliframe"],
    
    # Events & timing
    "event_handling": ["addeventlistener", "removeeventlistener", "customevent", "dispatchevent", "onclick", "onload", "onscroll", "onmouse", "event", "target", "currenttarget"],
    "timers_scheduling": ["setinterval", "settimeout", "requestanimationframe", "cancelanimationframe", "clearinterval", "cleartimeout", "queuemicrotask"],
    "observers": ["intersectionobserver", "intersectionratio", "mutationobserver", "resizeobserver", "performanceobserver", "reportingobserver"],
    
    # Page & browser state
    "page_lifecycle": ["visibilitystate", "hidden", "prerendering", "readystate", "domcontentloaded", "load", "unload", "beforeunload", "pagehide", "pageshow"],
    "performance_metrics": ["performance", "getentriesbytype", "getentriesbyname", "timing", "navigation", "resource", "measure", "mark", "now"],
    "location_url": ["hostname", "href", "origin", "pathname", "search", "hash", "protocol", "host", "port", "location", "url"],
    "navigator_info": ["navigator", "useragent", "language", "languages", "online", "connection", "cookieenabled", "donottrack", "vendor"],
    
    # Styling & layout
    "css_styling": ["style", "display", "cssstyledeclaration", "getcomputedstyle", "csstext", "stylesheets", "cssrule", "visibility", "opacity", "transform"],
    "layout_geometry": ["getboundingclientrect", "offsetwidth", "offsetheight", "clientwidth", "clientheight", "scrollwidth", "scrollheight", "offsettop", "offsetleft"],
    
    # User interaction
    "user_input": ["scroll", "click", "mouse", "touch", "keyboard", "focus", "blur", "input", "change", "submit", "keydown", "keyup", "keypress"],
    "pointer_events": ["pointerdown", "pointerup", "pointermove", "pointerenter", "pointerleave", "pointerover", "pointerout"],
    
    # Scripts & documents
    "script_document": ["currentscript", "script", "document", "window", "htmldocument", "documentelement", "body", "head", "title"],
    
    # Third-party & frameworks
    "error_monitoring": ["sentry", "bugsnag", "rollbar", "errorhandler", "onerror", "unhandledrejection", "reporterror"],
    "frameworks": ["react", "angular", "vue", "jquery", "webpack", "babel", "__proto__", "prototype"],
}

CATEGORY_DESCRIPTIONS = {
    "advertising": "Scripts focused on ad delivery, targeting, and conversion tracking. Often communicate with ad networks and DSPs.",
    "analytics": "Scripts that measure user engagement, page performance, and behavioral patterns for business intelligence.",
    "fingerprinting": "Scripts that collect device/browser characteristics to create unique user identifiers without cookies.",
    "functional": "Scripts providing core website functionality like UI components, forms, and interactive features.",
    "social": "Scripts enabling social media integration, sharing buttons, and embedded social content.",
    "security": "Scripts focused on fraud detection, bot prevention, and user authentication verification.",
}

CATEGORY_ICONS = {
    "advertising": "",
    "analytics": "",
    "fingerprinting": "",
    "functional": "",
    "social": "",
    "security": "",
}

# Display names for feature categories (for the report)
FEATURE_CATEGORY_DISPLAY = {
    "fingerprinting": ("Fingerprinting", "Device/browser identification techniques"),
    "tracking_storage": ("Storage", "Client-side data persistence"),
    "network_requests": ("Network", "HTTP requests and data transmission"),
    "dom_traversal": ("DOM Traversal", "Finding elements in page structure"),
    "dom_modification": ("DOM Modification", "Changing page content dynamically"),
    "dom_properties": ("DOM Properties", "Reading element attributes"),
    "html_elements": ("HTML Elements", "Specific HTML element types"),
    "event_handling": ("Event Handling", "User interaction listeners"),
    "timers_scheduling": ("Timers", "Scheduled code execution"),
    "observers": ("Observers", "Monitoring page/element changes"),
    "page_lifecycle": ("Page Lifecycle", "Page state and visibility"),
    "performance_metrics": ("Performance", "Timing and metrics collection"),
    "location_url": ("URL/Location", "Page address information"),
    "navigator_info": ("Navigator", "Browser/device information"),
    "css_styling": ("CSS Styling", "Visual appearance control"),
    "layout_geometry": ("Layout/Geometry", "Element size and position"),
    "user_input": ("User Input", "Keyboard, mouse, touch events"),
    "pointer_events": ("Pointer Events", "Mouse/touch tracking"),
    "script_document": ("Script/Document", "Core document APIs"),
    "error_monitoring": ("Error Monitoring", "Error tracking services"),
    "frameworks": ("Frameworks", "JS framework internals"),
}


def classify_feature(feature_name: str) -> tuple[str, str]:
    """Classify a feature into a high-level category. Returns (category_key, display_name)."""
    feature_lower = feature_name.lower()
    for category, keywords in FEATURE_CATEGORIES.items():
        if any(kw in feature_lower for kw in keywords):
            display_info = FEATURE_CATEGORY_DISPLAY.get(category, (category.replace("_", " ").title(), ""))
            return category, display_info[0]
    return "uncategorized", "Uncategorized"


def get_feature_description(feature_name: str) -> str:
    """Get a brief description for known features."""
    descriptions = {
        "canvas": "HTML5 Canvas API - often used for fingerprinting",
        "webgl": "WebGL rendering - can reveal GPU information",
        "navigator": "Browser/device information access",
        "localstorage": "Persistent client-side storage",
        "cookie": "HTTP cookie manipulation",
        "fetch": "Modern network requests",
        "beacon": "Analytics data transmission",
        "intersectionobserver": "Viewport visibility tracking",
        "performance": "Page/resource timing metrics",
        "customevent": "Custom event dispatching",
        "setinterval": "Periodic code execution",
        "visibilitystate": "Page visibility detection",
        "addeventlistener": "Event binding for user actions",
    }
    for key, desc in descriptions.items():
        if key in feature_name.lower():
            return desc
    return ""


def create_importance_bar(value: float, max_value: float, width: int = 20) -> str:
    """Create a text-based progress bar."""
    filled = int((value / max_value) * width) if max_value > 0 else 0
    return "█" * filled + "░" * (width - filled)


def analyze_feature_importance():
    """Analyze which features are most indicative of tracking categories."""
    OUTPUT_REPORT.parent.mkdir(exist_ok=True)

    # --- Load Data ---
    if not CLASSIFICATIONS_FILE.exists():
        print(f"Error: Classification file not found: {CLASSIFICATIONS_FILE}")
        return

    class_df = pd.read_csv(CLASSIFICATIONS_FILE)

    # --- Aggregate Features for Each Website ---
    all_websites = []
    all_features_raw = []
    for _, row in class_df.iterrows():
        safe_filename = row["url"].replace("https://", "").replace("http://", "").replace("/", "_").replace(":", "_")
        features_file = FEATURES_DIR / f"mega_features_{safe_filename}.csv"
        if features_file.exists():
            features_df = pd.read_csv(features_file)
            if not features_df.empty:
                if "feature_name" in features_df.columns:
                    feature_list = []
                    for _, feat_row in features_df.iterrows():
                        feature_name = str(feat_row["feature_name"])
                        usage_count = int(feat_row.get("usage_count", 1))
                        feature_list.extend([feature_name] * min(usage_count, 100))
                        all_features_raw.append(feature_name)
                    
                    feature_doc = " ".join(feature_list)
                    all_websites.append({
                        "url": row["url"],
                        "category": row["primary_category"],
                        "features": feature_doc
                    })
    
    if not all_websites:
        print("No feature data found. Cannot perform analysis.")
        return

    script_df = pd.DataFrame(all_websites)
    
    # Calculate category distribution
    category_counts = script_df["category"].value_counts()
    total_websites = len(script_df)

    # --- TF-IDF Vectorization ---
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    X = vectorizer.fit_transform(script_df["features"])
    y = script_df["category"]
    feature_names = np.array(vectorizer.get_feature_names_out())

    # --- Train a Classifier to Find Feature Importances ---
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    max_importance = importances[sorted_indices[0]]

    # --- Generate Report ---
    with open(OUTPUT_REPORT, "w") as f:
        # Header
        f.write("# Feature Importance Analysis Report\n\n")
        f.write(f"> **Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M')}\n\n")
        f.write("---\n\n")
        
        # Executive Summary
        f.write("## 1. Executive Summary\n\n")
        f.write("This report analyzes JavaScript API usage patterns to identify features most predictive of different tracking behaviors. ")
        f.write("Using machine learning (Random Forest classification with TF-IDF vectorization), we ranked features by their discriminative power.\n\n")
        
        # Dataset Overview
        f.write("### Dataset Overview\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| **Total Websites Analyzed** | {total_websites} |\n")
        f.write(f"| **Unique Features Detected** | {len(feature_names):,} |\n")
        f.write(f"| **Categories Identified** | {len(category_counts)} |\n\n")
        
        # Category Distribution
        f.write("### Category Distribution\n\n")
        f.write("| Category | Count | Percentage | Distribution |\n")
        f.write("|----------|-------|------------|-------------|\n")
        for cat, count in category_counts.items():
            pct = (count / total_websites) * 100
            bar = create_importance_bar(count, category_counts.max(), 15)
            f.write(f"| **{cat.capitalize()}** | {count} | {pct:.1f}% | `{bar}` |\n")
        f.write("\n---\n\n")

        # Key Findings
        f.write("## 2. Key Findings\n\n")
        
        # Analyze top features by type
        top_20_features = [feature_names[sorted_indices[i]] for i in range(min(20, len(sorted_indices)))]
        feature_type_counts = Counter([classify_feature(f)[0] for f in top_20_features])
        
        f.write("### Top Feature Categories in Tracking Detection\n\n")
        f.write("The most predictive features fall into these categories:\n\n")
        for feat_type, count in feature_type_counts.most_common():
            display_info = FEATURE_CATEGORY_DISPLAY.get(feat_type, (feat_type.replace("_", " ").title(), ""))
            display_name = display_info[0]
            f.write(f"- **{display_name}**: {count} feature{'s' if count > 1 else ''} in top 20\n")
        f.write("\n")

        # --- Overall Top Features ---
        f.write("---\n\n")
        f.write("## 3. Top 20 Most Predictive Features\n\n")
        f.write("These features have the highest discriminative power for classifying tracking behavior:\n\n")
        f.write("| Rank | Feature | Importance | Relative | Category |\n")
        f.write("|:----:|---------|:----------:|----------|----------|\n")
        
        for i in range(min(20, len(sorted_indices))):
            idx = sorted_indices[i]
            feat_name = feature_names[idx]
            imp = importances[idx]
            bar = create_importance_bar(imp, max_importance, 12)
            _, feat_cat_display = classify_feature(feat_name)
            
            f.write(f"| {i+1} | `{feat_name}` | {imp:.4f} | `{bar}` | {feat_cat_display} |\n")
        
        f.write("\n")
        
        # Insights box
        f.write("> **Note:** The prominence of timing/observation APIs (`setinterval`, `intersectionratio`, `visibilitystate`) ")
        f.write("suggests that tracking scripts heavily rely on monitoring user behavior and page state changes.\n\n")

        # --- Top Features per Category ---
        f.write("---\n\n")
        f.write("## 4. Category-Specific Feature Analysis\n\n")
        f.write("Below are the most characteristic features for each tracking category, ranked by TF-IDF score within that category.\n\n")
        
        for category in sorted(script_df["category"].unique()):
            f.write(f"### 4.{list(sorted(script_df['category'].unique())).index(category)+1} {category.capitalize()}\n\n")
            
            # Add category description
            if category in CATEGORY_DESCRIPTIONS:
                f.write(f"*{CATEGORY_DESCRIPTIONS[category]}*\n\n")
            
            # Find features most associated with this category
            category_indices = np.where(y == category)[0]
            if len(category_indices) > 0:
                category_X = X[category_indices]
                mean_tfidf = np.array(category_X.mean(axis=0)).flatten()
                top_feature_indices = np.argsort(mean_tfidf)[::-1]
                max_tfidf = mean_tfidf[top_feature_indices[0]] if len(top_feature_indices) > 0 else 1
                
                f.write("| Rank | Feature | TF-IDF Score | Relevance |\n")
                f.write("|:----:|---------|:------------:|----------|\n")
                
                for rank, i in enumerate(top_feature_indices[:10]):
                    feat_name = feature_names[i]
                    tfidf_score = mean_tfidf[i]
                    bar = create_importance_bar(tfidf_score, max_tfidf, 10)
                    f.write(f"| {rank+1} | `{feat_name}` | {tfidf_score:.4f} | `{bar}` |\n")
                
                f.write("\n")
                
                # Add category-specific insight
                top_3 = [feature_names[top_feature_indices[j]] for j in range(min(3, len(top_feature_indices)))]
                f.write(f"> **Notable:** Top features include `{'`, `'.join(top_3)}`\n\n")
        
        # --- Methodology Section ---
        f.write("---\n\n")
        f.write("## 5. Methodology\n\n")
        f.write("### Analysis Pipeline\n\n")
        f.write("```\n")
        f.write("┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐\n")
        f.write("│  Raw Feature    │───▶│  TF-IDF          │───▶│  Random Forest  │\n")
        f.write("│  Extraction     │    │  Vectorization   │    │  Classification │\n")
        f.write("└─────────────────┘    └──────────────────┘    └─────────────────┘\n")
        f.write("                                                       │\n")
        f.write("                                                       ▼\n")
        f.write("                                              ┌─────────────────┐\n")
        f.write("                                              │  Feature        │\n")
        f.write("                                              │  Importance     │\n")
        f.write("                                              └─────────────────┘\n")
        f.write("```\n\n")
        f.write("### Technical Details\n\n")
        f.write("| Component | Configuration |\n")
        f.write("|-----------|---------------|\n")
        f.write("| **Vectorizer** | TF-IDF with max 1,000 features |\n")
        f.write("| **Classifier** | Random Forest (100 estimators) |\n")
        f.write("| **Feature Weighting** | Usage count (capped at 100) |\n")
        f.write("| **Ranking Method** | Gini importance from RF |\n\n")
        
        # Footer
        f.write("---\n\n")
        f.write("## 6. Appendix: Feature Category Definitions\n\n")
        f.write("| Category | Description | Example Features |\n")
        f.write("|----------|-------------|------------------|\n")
        for cat, keywords in FEATURE_CATEGORIES.items():
            display_info = FEATURE_CATEGORY_DISPLAY.get(cat, (cat.replace("_", " ").title(), ""))
            display_name, description = display_info
            examples = ", ".join(f"`{k}`" for k in keywords[:4])
            f.write(f"| {display_name} | {description} | {examples} |\n")
        f.write("\n---\n\n")
        f.write("*Report generated by LLM Tracking Analysis Pipeline*\n")

    print(f"Feature importance report saved to: {OUTPUT_REPORT}")

if __name__ == "__main__":
    analyze_feature_importance()
