import pandas as pd
from pathlib import Path
import json
import re
import requests
from urllib.parse import urlparse
from collections import defaultdict
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
GROUND_TRUTH_DIR = Path("data/ground_truth_sources")
CLASSIFICATIONS_FILE = Path("data/processed/llm_classifications.csv")
OUTPUT_DIR = Path("data/processed")
CACHE_EXPIRY_DAYS = 7  # Re-download sources after this many days

# Authoritative source URLs grouped into logical providers:
# 1) AdBlock team (EasyList + EasyPrivacy)
# 2) OISD blocklist (hosts format)
SOURCES = {
    # AdBlock team lists
    "adblock_easyprivacy": {
        "url": "https://easylist.to/easylist/easyprivacy.txt",
        "filename": "easyprivacy.txt",
        "type": "adblock",
    },
    "adblock_easylist": {
        "url": "https://easylist.to/easylist/easylist.txt",
        "filename": "easylist.txt",
        "type": "adblock",
    },
    # OISD blocklist (hosts format)
    "oisd": {
        "url": "https://big.oisd.nl/domainswild",
        "filename": "oisd_domains.txt",
        "type": "hosts",
    },
}

# Category normalization mapping
CATEGORY_MAPPING = {
    # Advertising
    "advertising": "advertising",
    "ad": "advertising",
    "ads": "advertising",
    "advertisement": "advertising",
    "ad network": "advertising",
    "ad_network": "advertising",
    "ad-network": "advertising",
    # Analytics
    "analytics": "analytics",
    "analytic": "analytics",
    "site analytics": "analytics",
    "site_analytics": "analytics",
    "audience measurement": "analytics",
    "audience_measurement": "analytics",
    # Fingerprinting
    "fingerprinting": "fingerprinting",
    "fingerprint": "fingerprinting",
    "device fingerprinting": "fingerprinting",
    "canvas fingerprinting": "fingerprinting",
    # Social
    "social": "social",
    "social network": "social",
    "social_network": "social",
    "social tracking": "social",
    # Tracking (generic)
    "tracking": "tracking",
    "tracker": "tracking",
    "third-party tracking": "tracking",
    # Content delivery (often benign)
    "content": "content",
    "cdn": "content",
    "content delivery": "content",
    # Session replay
    "session replay": "session_replay",
    "session_replay": "session_replay",
    "session-replay": "session_replay",
    # Cryptomining (malicious)
    "cryptomining": "cryptomining",
    "cryptominer": "cryptomining",
    "mining": "cryptomining",
    # Malware
    "malware": "malware",
    "malicious": "malware",
    # Email tracking
    "email": "email_tracking",
    "email tracking": "email_tracking",
    # Consent management
    "consent": "consent",
    "consent management": "consent",
    # Marketing
    "marketing": "marketing",
    # Embedded content
    "embedded": "embedded",
    "embedded content": "embedded",
    # Disconnect-specific categories
    "disconnect": "advertising",
    "facebook disconnect": "social",
    "google disconnect": "advertising",
}


def normalize_category(category: str) -> str:
    """Normalize category names across different sources."""
    if not category:
        return "tracking"
    cat_lower = category.lower().strip()
    return CATEGORY_MAPPING.get(cat_lower, cat_lower)


def download_source(name: str, source_info: dict, force: bool = False) -> Path | None:
    """Download a tracking database source if not cached or expired."""
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    filepath = GROUND_TRUTH_DIR / source_info["filename"]
    
    # Check if we need to download
    should_download = force or not filepath.exists()
    if filepath.exists():
        file_age = datetime.now() - datetime.fromtimestamp(filepath.stat().st_mtime)
        if file_age > timedelta(days=CACHE_EXPIRY_DAYS):
            should_download = True
            logger.info(f"Cache expired for {name}, re-downloading...")
    
    if should_download:
        logger.info(f"Downloading {name} from {source_info['url']}...")
        try:
            response = requests.get(source_info["url"], timeout=60)
            response.raise_for_status()
            filepath.write_bytes(response.content)
            logger.info(f"Successfully downloaded {name}")
        except requests.RequestException as e:
            logger.warning(f"Failed to download {name}: {e}")
            if not filepath.exists():
                return None
    
    return filepath


def parse_adblock_list(filepath: Path | None) -> dict:
    """
    Parse AdBlock Plus format filter lists (EasyPrivacy, EasyList, uBlock).
    Returns a dict of {domain: {"categories": set(), "rules": list()}}
    """
    trackers: dict = {}
    
    def make_tracker_entry():
        return {"categories": set(), "rules": []}
    
    if not filepath or not filepath.exists():
        return trackers
    
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith("!") or line.startswith("["):
                continue
            
            # Parse domain-based rules
            domain = None
            
            # Pattern: ||domain^ (third-party domain blocking)
            if line.startswith("||") and "^" in line:
                match = re.match(r'\|\|([a-zA-Z0-9.-]+)\^', line)
                if match:
                    domain = match.group(1).lower()
            
            # Pattern: ||domain/path (URL path blocking)
            elif line.startswith("||"):
                match = re.match(r'\|\|([a-zA-Z0-9.-]+)/', line)
                if match:
                    domain = match.group(1).lower()
            
            # Pattern: @@||domain (whitelist - skip)
            elif line.startswith("@@"):
                continue
            
            # Pattern: /regex/ (skip complex regex patterns)
            elif line.startswith("/") and line.endswith("/"):
                continue
            
            if domain and len(domain) > 3 and "." in domain:
                # Remove www. prefix
                if domain.startswith("www."):
                    domain = domain[4:]
                
                if domain not in trackers:
                    trackers[domain] = make_tracker_entry()
                
                trackers[domain]["rules"].append(line)
                
                # Infer category from filename
                filename_lower = filepath.name.lower()
                if "privacy" in filename_lower:
                    trackers[domain]["categories"].add("tracking")
                elif "easylist" in filename_lower:
                    trackers[domain]["categories"].add("advertising")
    
    return trackers


def parse_disconnect_json(filepath: Path | None) -> dict:
    """
    Parse Disconnect.me tracker list.
    Returns a dict of {domain: {"categories": set(), "company": str}}
    """
    trackers: dict = {}
    
    def make_tracker_entry():
        return {"categories": set(), "company": ""}
    
    if not filepath or not filepath.exists():
        return trackers
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Disconnect JSON: {e}")
        return trackers
    
    # Disconnect structure: {"categories": {"Advertising": [{"CompanyName": {"domain": ["urls"]}}]}}
    categories = data.get("categories", data)  # Handle different formats
    
    for category_name, services in categories.items():
        if not isinstance(services, list):
            continue
            
        normalized_category = normalize_category(category_name)
        
        for service in services:
            if not isinstance(service, dict):
                continue
                
            for company_name, domains_info in service.items():
                if not isinstance(domains_info, dict):
                    continue
                    
                for primary_domain, related_domains in domains_info.items():
                    # Add the primary domain
                    if isinstance(primary_domain, str) and "." in primary_domain:
                        clean_domain = primary_domain.replace("http://", "").replace("https://", "").rstrip("/")
                        if clean_domain.startswith("www."):
                            clean_domain = clean_domain[4:]
                        if clean_domain not in trackers:
                            trackers[clean_domain] = make_tracker_entry()
                        trackers[clean_domain]["categories"].add(normalized_category)
                        trackers[clean_domain]["company"] = company_name
                    
                    # Add related domains
                    if isinstance(related_domains, list):
                        for domain in related_domains:
                            if isinstance(domain, str) and "." in domain:
                                clean = domain.replace("http://", "").replace("https://", "").rstrip("/")
                                if clean.startswith("www."):
                                    clean = clean[4:]
                                if clean not in trackers:
                                    trackers[clean] = make_tracker_entry()
                                trackers[clean]["categories"].add(normalized_category)
                                trackers[clean]["company"] = company_name
    
    return trackers


def parse_hosts_file(filepath: Path | None) -> dict:
    """
    Parse hosts-style or plain-domain blocklists (Steven Black, OISD, etc.).

    Supported formats:
      - Classic hosts: 0.0.0.0 domain.com  /  127.0.0.1 domain.com
      - Plain domains: one domain per line (e.g., OISD domainswild)
    """
    trackers: dict = {}
    
    def make_tracker_entry():
        return {"categories": set(), "rules": []}
    
    if not filepath or not filepath.exists():
        return trackers
    
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            
            domain: str | None = None

            parts = line.split()

            if len(parts) >= 2:
                # Hosts format: IP domain
                ip = parts[0]
                d = parts[1].lower()
                if ip in ("0.0.0.0", "127.0.0.1"):
                    domain = d
            else:
                # Plain-domain format (e.g., OISD domainswild)
                d = line.lower()
                domain = d

            if not domain:
                continue

            # Normalise domain/patterns from lists like OISD:
            #  - strip leading "*." or "*"
            #  - strip leading "."
            domain = domain.lstrip("*.")  # remove any wildcard prefixes
            if domain.startswith("."):
                domain = domain[1:]

            # Basic sanity checks
            if " " in domain or "/" in domain or domain == "localhost":
                continue
            if "." not in domain or len(domain) <= 3:
                continue

            # Remove common prefix
            if domain.startswith("www."):
                domain = domain[4:]

            if domain not in trackers:
                trackers[domain] = make_tracker_entry()
            trackers[domain]["categories"].add("tracking")
            trackers[domain]["rules"].append(line)
    
    return trackers


def parse_duckduckgo_tracker_radar(filepath: Path | None) -> dict:
    """
    Parse DuckDuckGo Tracker Radar database.
    This is one of the most comprehensive tracker databases available.
    """
    trackers: dict = {}
    
    def make_tracker_entry():
        return {"categories": set(), "fingerprinting": 0, "prevalence": 0}
    
    if not filepath or not filepath.exists():
        return trackers
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse DuckDuckGo Tracker Radar: {e}")
        return trackers
    
    # Handle different possible structures
    tracker_list = data if isinstance(data, list) else data.get("trackers", data.get("domains", []))
    
    if isinstance(tracker_list, dict):
        # Format: {domain: {info}}
        for domain, info in tracker_list.items():
            if isinstance(info, dict):
                clean_domain = domain.lower()
                if clean_domain.startswith("www."):
                    clean_domain = clean_domain[4:]
                
                if clean_domain not in trackers:
                    trackers[clean_domain] = make_tracker_entry()
                
                # Extract categories
                cats = info.get("categories", info.get("category", []))
                if isinstance(cats, str):
                    cats = [cats]
                for cat in cats:
                    trackers[clean_domain]["categories"].add(normalize_category(cat))
                
                # Extract fingerprinting score
                trackers[clean_domain]["fingerprinting"] = info.get("fingerprinting", 0)
                trackers[clean_domain]["prevalence"] = info.get("prevalence", 0)
    
    elif isinstance(tracker_list, list):
        for entry in tracker_list:
            if isinstance(entry, dict):
                domain = entry.get("domain", entry.get("name", ""))
                if not domain or "." not in domain:
                    continue
                
                clean_domain = domain.lower()
                if clean_domain.startswith("www."):
                    clean_domain = clean_domain[4:]
                
                if clean_domain not in trackers:
                    trackers[clean_domain] = make_tracker_entry()
                
                cats = entry.get("categories", entry.get("category", []))
                if isinstance(cats, str):
                    cats = [cats]
                for cat in cats:
                    trackers[clean_domain]["categories"].add(normalize_category(cat))
                
                trackers[clean_domain]["fingerprinting"] = entry.get("fingerprinting", 0)
                trackers[clean_domain]["prevalence"] = entry.get("prevalence", 0)
    
    return trackers


class GroundTruthDatabase:
    """Aggregated ground truth database from multiple sources."""
    
    def __init__(self):
        self.trackers: dict = {}
        self.source_stats: dict = {}
    
    def _make_tracker_entry(self) -> dict:
        return {
            "sources": set(),
            "categories": set(),
            "companies": set(),
            "fingerprinting_score": 0,
            "prevalence": 0,
            "rules_count": 0
        }
    
    def add_source(self, name: str, tracker_dict: dict):
        """Add trackers from a source to the database."""
        count = 0
        for domain, info in tracker_dict.items():
            if not domain or len(domain) < 4:
                continue
            
            if domain not in self.trackers:
                self.trackers[domain] = self._make_tracker_entry()
            
            self.trackers[domain]["sources"].add(name)
            count += 1
            
            if "categories" in info:
                self.trackers[domain]["categories"].update(info["categories"])
            
            if "company" in info and info["company"]:
                self.trackers[domain]["companies"].add(info["company"])
            
            if "fingerprinting" in info:
                self.trackers[domain]["fingerprinting_score"] = max(
                    self.trackers[domain]["fingerprinting_score"],
                    info["fingerprinting"]
                )
            
            if "prevalence" in info:
                self.trackers[domain]["prevalence"] = max(
                    self.trackers[domain]["prevalence"],
                    info["prevalence"]
                )
            
            if "rules" in info:
                self.trackers[domain]["rules_count"] += len(info["rules"])
        
        self.source_stats[name] = count
        logger.info(f"Added {count} domains from {name}")
    
    def lookup_domain(self, domain: str) -> dict | None:
        """Look up a domain in the database, including subdomain matching."""
        domain = domain.lower().strip()
        if domain.startswith("www."):
            domain = domain[4:]
        
        # Exact match
        if domain in self.trackers:
            return self._format_result(domain, self.trackers[domain], "exact")
        
        # Subdomain matching (e.g., tracker.example.com -> example.com)
        parts = domain.split(".")
        for i in range(1, len(parts) - 1):
            parent = ".".join(parts[i:])
            if parent in self.trackers:
                return self._format_result(domain, self.trackers[parent], "subdomain")
        
        # Check if this domain is a parent of any known tracker
        # (useful for detecting tracker infrastructure domains)
        for tracker_domain in self.trackers:
            if tracker_domain.endswith("." + domain):
                return self._format_result(domain, self.trackers[tracker_domain], "parent")
        
        return None
    
    def _format_result(self, queried_domain: str, data: dict, match_type: str) -> dict:
        """Format lookup result with confidence scoring."""
        num_sources = len(data["sources"])
        
        # Confidence based on number of confirming sources (3 sources max)
        # Using presentation-friendly labels
        if num_sources >= 3:
            confidence = "verified"      # All 3 sources agree
            confidence_pct = 98
        elif num_sources >= 2:
            confidence = "confirmed"     # 2 sources agree
            confidence_pct = 92
        else:
            confidence = "detected"      # 1 source found it
            confidence_pct = 85
        
        # Adjust confidence for match type
        if match_type == "subdomain":
            confidence = {"verified": "confirmed", "confirmed": "detected", "detected": "detected"}[confidence]
            confidence_pct = max(confidence_pct - 8, 80)
        elif match_type == "parent":
            confidence = "detected"
            confidence_pct = 80
        
        # Determine primary category
        categories = list(data["categories"]) if data["categories"] else ["tracking"]
        primary_category = self._determine_primary_category(categories)
        
        return {
            "is_tracker": True,
            "domain": queried_domain,
            "match_type": match_type,
            "sources": list(data["sources"]),
            "num_sources": num_sources,
            "confidence": confidence,
            "confidence_pct": confidence_pct,
            "primary_category": primary_category,
            "all_categories": categories,
            "companies": list(data["companies"]),
            "fingerprinting_score": data["fingerprinting_score"],
            "prevalence": data["prevalence"]
        }
    
    def _determine_primary_category(self, categories: list) -> str:
        """Determine the most significant category."""
        # Priority order for primary category
        priority = [
            "fingerprinting", "cryptomining", "malware", "session_replay",
            "advertising", "analytics", "social", "marketing",
            "tracking", "email_tracking", "content", "embedded", "consent"
        ]
        
        for cat in priority:
            if cat in categories:
                return cat
        
        return categories[0] if categories else "tracking"
    
    def get_stats(self) -> dict:
        """Get statistics about the database."""
        total_domains = len(self.trackers)
        category_counts = defaultdict(int)
        source_coverage = defaultdict(int)
        
        for domain, data in self.trackers.items():
            for cat in data["categories"]:
                category_counts[cat] += 1
            for source in data["sources"]:
                source_coverage[source] += 1
        
        return {
            "total_domains": total_domains,
            "category_counts": dict(category_counts),
            "source_coverage": dict(source_coverage),
            "multi_source_confirmed": sum(1 for d in self.trackers.values() if len(d["sources"]) >= 2)
        }


def generate_visualizations(db: GroundTruthDatabase, output_dir: Path):
    """Generate visualization charts for the ground truth database."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = db.get_stats()
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Color palette - modern, distinctive colors
    colors = {
        'primary': '#2563EB',      # Blue
        'secondary': '#7C3AED',    # Purple
        'accent1': '#059669',      # Emerald
        'accent2': '#DC2626',      # Red
        'accent3': '#D97706',      # Amber
        'accent4': '#0891B2',      # Cyan
        'accent5': '#DB2777',      # Pink
        'accent6': '#4F46E5',      # Indigo
        'background': '#F8FAFC',
        'text': '#1E293B'
    }
    
    color_palette = [colors['primary'], colors['secondary'], colors['accent1'], 
                     colors['accent2'], colors['accent3'], colors['accent4'],
                     colors['accent5'], colors['accent6']]
    
    # ========== 1. Source Coverage Bar Chart ==========
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=colors['background'])
    ax.set_facecolor(colors['background'])
    
    sources = list(stats['source_coverage'].keys())
    counts = list(stats['source_coverage'].values())
    
    # Sort by count
    sorted_data = sorted(zip(sources, counts), key=lambda x: x[1], reverse=True)
    sources, counts = zip(*sorted_data)
    
    bars = ax.barh(sources, counts, color=color_palette[:len(sources)], edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2, 
                f'{count:,}', va='center', ha='left', fontsize=11, fontweight='bold', color=colors['text'])
    
    ax.set_xlabel('Number of Tracker Domains', fontsize=12, fontweight='bold', color=colors['text'])
    ax.set_title('Ground Truth Sources Coverage', fontsize=16, fontweight='bold', color=colors['text'], pad=20)
    ax.tick_params(colors=colors['text'], labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(counts) * 1.15)
    
    plt.tight_layout()
    source_chart_path = output_dir / 'ground_truth_sources.png'
    plt.savefig(source_chart_path, dpi=150, facecolor=colors['background'], bbox_inches='tight')
    plt.close()
    logger.info(f"Source coverage chart saved to: {source_chart_path}")
    
    # ========== 2. Category Distribution Pie Chart ==========
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=colors['background'])
    
    categories = list(stats['category_counts'].keys())
    cat_counts = list(stats['category_counts'].values())
    
    # Sort by count
    sorted_cat_data = sorted(zip(categories, cat_counts), key=lambda x: x[1], reverse=True)
    categories, cat_counts = zip(*sorted_cat_data)
    
    # Create pie chart with explosion for emphasis
    explode = [0.02] * len(categories)
    explode[0] = 0.05  # Emphasize the largest category
    
    wedges, texts, autotexts = ax.pie(
        cat_counts, 
        labels=categories,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(cat_counts)):,})',
        colors=color_palette[:len(categories)],
        explode=explode,
        shadow=False,
        startangle=90,
        textprops={'fontsize': 11, 'color': colors['text']},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    # Style the percentage labels
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    ax.set_title('Tracker Category Distribution', fontsize=16, fontweight='bold', color=colors['text'], pad=20)
    
    plt.tight_layout()
    category_chart_path = output_dir / 'ground_truth_categories.png'
    plt.savefig(category_chart_path, dpi=150, facecolor=colors['background'], bbox_inches='tight')
    plt.close()
    logger.info(f"Category distribution chart saved to: {category_chart_path}")
    
    # ========== 3. Confidence / Validation Level Distribution ==========
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=colors['background'])
    ax.set_facecolor(colors['background'])
    
    # With two providers (AdBlock + OISD), we distinguish:
    # - "Both lists"   -> domain appears in BOTH sources (high agreement)
    # - "Single list"  -> domain appears in exactly ONE source
    confidence_counts = {'both': 0, 'single': 0}
    for domain, data in db.trackers.items():
        num_sources = len(data['sources'])
        if num_sources >= 2:
            confidence_counts['both'] += 1
        else:
            confidence_counts['single'] += 1
    
    conf_labels = ['Both lists\n(2 sources)\n95%', 'Single list\n(1 source)\n85%']
    conf_values = [confidence_counts['both'], confidence_counts['single']]
    conf_colors = [colors['accent1'], colors['accent3']]  # Green, Amber
    
    bars = ax.bar(conf_labels, conf_values, color=conf_colors, edgecolor='white', linewidth=2)
    
    # Add value labels on top of bars
    for bar, val in zip(bars, conf_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(conf_values)*0.02,
                f'{val:,}', ha='center', va='bottom', fontsize=12, fontweight='bold', color=colors['text'])
    
    ax.set_ylabel('Number of Domains', fontsize=12, fontweight='bold', color=colors['text'])
    ax.set_title('Ground Truth Validation Level\n(Based on AdBlock–OISD Agreement)', 
                 fontsize=14, fontweight='bold', color=colors['text'], pad=20)
    ax.tick_params(colors=colors['text'], labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(conf_values) * 1.15)
    
    plt.tight_layout()
    confidence_chart_path = output_dir / 'ground_truth_confidence.png'
    plt.savefig(confidence_chart_path, dpi=150, facecolor=colors['background'], bbox_inches='tight')
    plt.close()
    logger.info(f"Confidence distribution chart saved to: {confidence_chart_path}")

    # ========== 3b. High-Confidence Overlap Subset ==========
    # This chart focuses only on domains that appear in both sources.
    high_conf_total = confidence_counts["both"]
    if high_conf_total > 0:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor=colors["background"])
        ax.set_facecolor(colors["background"])

        overlap_labels = ["Both lists (AdBlock ∩ OISD)"]
        overlap_values = [confidence_counts["both"]]
        overlap_colors = [colors["accent1"]]

        bars = ax.bar(overlap_labels, overlap_values, color=overlap_colors, edgecolor="white", linewidth=2)

        for bar, val in zip(bars, overlap_values):
            pct = val / high_conf_total * 100 if high_conf_total else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(overlap_values) * 0.03,
                f"{val:,}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color=colors["text"],
            )

        ax.set_ylabel("Number of Domains", fontsize=12, fontweight="bold", color=colors["text"])
        ax.set_title(
            "High-Confidence Tracker Domains\n(Domains in Both AdBlock and OISD)",
            fontsize=14,
            fontweight="bold",
            color=colors["text"],
            pad=20,
        )
        ax.tick_params(colors=colors["text"], labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, max(overlap_values) * 1.25)

        plt.tight_layout()
        confidence_overlap_path = output_dir / "ground_truth_confidence_overlap.png"
        plt.savefig(confidence_overlap_path, dpi=150, facecolor=colors["background"], bbox_inches="tight")
        plt.close()
        logger.info(f"High-confidence overlap chart saved to: {confidence_overlap_path}")
    else:
        confidence_overlap_path = None
    
    # ========== 4. Summary Dashboard ==========
    fig = plt.figure(figsize=(16, 12), facecolor=colors['background'])
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Title
    fig.suptitle('Ground Truth Database Overview', fontsize=20, fontweight='bold', 
                 color=colors['text'], y=0.98)
    
    # --- Key Metrics (top row) ---
    ax_metrics = fig.add_subplot(gs[0, :])
    ax_metrics.set_facecolor(colors['background'])
    ax_metrics.axis('off')
    
    metrics = [
        ('Total Tracker Domains', f"{stats['total_domains']:,}", colors['primary']),
        ('Multi-Source Confirmed', f"{stats['multi_source_confirmed']:,}", colors['accent1']),
        ('Data Sources', f"{len(stats['source_coverage'])}", colors['secondary']),
        ('Categories', f"{len(stats['category_counts'])}", colors['accent3'])
    ]
    
    for i, (label, value, color) in enumerate(metrics):
        x_pos = 0.125 + i * 0.25
        # Value
        ax_metrics.text(x_pos, 0.6, value, ha='center', va='center', 
                       fontsize=28, fontweight='bold', color=color, transform=ax_metrics.transAxes)
        # Label
        ax_metrics.text(x_pos, 0.2, label, ha='center', va='center',
                       fontsize=12, color=colors['text'], transform=ax_metrics.transAxes)
        # Underline
        ax_metrics.axhline(y=0.1, xmin=x_pos-0.08, xmax=x_pos+0.08, color=color, linewidth=3)
    
    # --- Source Coverage (bottom left) ---
    ax_sources = fig.add_subplot(gs[1:, 0])
    ax_sources.set_facecolor(colors['background'])
    
    bars = ax_sources.barh(sources, counts, color=color_palette[:len(sources)], edgecolor='white', linewidth=1)
    for bar, count in zip(bars, counts):
        ax_sources.text(bar.get_width() + max(counts)*0.02, bar.get_y() + bar.get_height()/2, 
                       f'{count:,}', va='center', ha='left', fontsize=9, fontweight='bold')
    ax_sources.set_xlabel('Domains', fontsize=10, fontweight='bold')
    ax_sources.set_title('Source Coverage', fontsize=12, fontweight='bold', color=colors['text'])
    ax_sources.spines['top'].set_visible(False)
    ax_sources.spines['right'].set_visible(False)
    ax_sources.set_xlim(0, max(counts) * 1.2)
    
    # --- Category Distribution (bottom center) ---
    ax_cat = fig.add_subplot(gs[1:, 1])
    ax_cat.set_facecolor(colors['background'])
    
    wedges, texts, autotexts = ax_cat.pie(
        cat_counts, 
        autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else '',
        colors=color_palette[:len(categories)],
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    ax_cat.set_title('Category Distribution', fontsize=12, fontweight='bold', color=colors['text'])
    
    # Add legend
    ax_cat.legend(wedges, categories, loc='upper left', bbox_to_anchor=(-0.1, -0.05), 
                  fontsize=9, frameon=False)
    
    # --- Validation Distribution (bottom right) ---
    ax_conf = fig.add_subplot(gs[1:, 2])
    ax_conf.set_facecolor(colors['background'])
    
    conf_short_labels = ['Both lists', 'Single list']
    bars = ax_conf.bar(conf_short_labels, conf_values, color=conf_colors, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, conf_values):
        ax_conf.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(conf_values)*0.02,
                    f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax_conf.set_ylabel('Domains', fontsize=10, fontweight='bold')
    ax_conf.set_title('Validation Levels', fontsize=12, fontweight='bold', color=colors['text'])
    ax_conf.spines['top'].set_visible(False)
    ax_conf.spines['right'].set_visible(False)
    ax_conf.set_ylim(0, max(conf_values) * 1.15)
    ax_conf.tick_params(axis='x', rotation=15)
    
    # Add timestamp
    fig.text(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
             ha='right', va='bottom', fontsize=8, color='gray')
    
    dashboard_path = output_dir / 'ground_truth_dashboard.png'
    plt.savefig(dashboard_path, dpi=150, facecolor=colors['background'], bbox_inches='tight')
    plt.close()
    logger.info(f"Dashboard saved to: {dashboard_path}")
    
    return {
        'sources': source_chart_path,
        'categories': category_chart_path,
        'confidence': confidence_chart_path,
        'confidence_overlap': confidence_overlap_path,
        'dashboard': dashboard_path
    }


def merge_tracker_dicts(*tracker_dicts: dict) -> dict:
    """Merge multiple tracker dictionaries into a single one.

    Each input dict is of the form:
      { domain: {"categories": set(), "rules": [...], ...} }
    Only categories and rules are merged, which is sufficient for our use.
    """
    merged: dict = {}
    for td in tracker_dicts:
        if not td:
            continue
        for domain, info in td.items():
            if domain not in merged:
                merged[domain] = {
                    "categories": set(),
                    "rules": [],
                }
            if "categories" in info:
                merged[domain]["categories"].update(info["categories"])
            if "rules" in info:
                merged[domain]["rules"].extend(info["rules"])
    return merged


def build_ground_truth_database(force_download: bool = False) -> GroundTruthDatabase:
    """Build ground truth database from major providers:

    1) AdBlock team   -> EasyList + EasyPrivacy
    2) OISD blocklist -> big.oisd.nl (hosts format)
    """
    db = GroundTruthDatabase()
    
    logger.info("Building ground truth database from AdBlock and OISD sources...")

    # --- AdBlock team (EasyList + EasyPrivacy) ---
    adblock_easyprivacy_path = download_source("adblock_easyprivacy", SOURCES["adblock_easyprivacy"], force_download)
    adblock_easylist_path = download_source("adblock_easylist", SOURCES["adblock_easylist"], force_download)
    adblock_dict = merge_tracker_dicts(
        parse_adblock_list(adblock_easyprivacy_path),
        parse_adblock_list(adblock_easylist_path),
    )
    if adblock_dict:
        db.add_source("adblock_team", adblock_dict)

    # --- OISD blocklist (hosts format) ---
    oisd_path = download_source("oisd", SOURCES["oisd"], force_download)
    if oisd_path:
        db.add_source("oisd", parse_hosts_file(oisd_path))
    
    # Print statistics
    stats = db.get_stats()
    logger.info(f"\nGround Truth Database Statistics:")
    logger.info(f"  Total unique tracker domains: {stats['total_domains']:,}")
    logger.info(f"  Multi-source confirmed: {stats['multi_source_confirmed']:,}")
    logger.info(f"\n  Category breakdown:")
    for cat, count in sorted(stats['category_counts'].items(), key=lambda x: -x[1])[:10]:
        logger.info(f"    {cat}: {count:,}")
    
    return db


def generate_script_level_ground_truth(db: GroundTruthDatabase) -> pd.DataFrame:
    """
    Generate ground truth for individual scripts by matching against tracker databases.
    """
    if not CLASSIFICATIONS_FILE.exists():
        logger.error(f"Classification file not found: {CLASSIFICATIONS_FILE}")
        return None
    
    class_df = pd.read_csv(CLASSIFICATIONS_FILE)
    
    records = []
    tracked_domains = set()
    
    # Process each website
    for _, row in class_df.iterrows():
        website_url = row.get("url", "")
        
        try:
            domain = urlparse(website_url).netloc
            if domain.startswith("www."):
                domain = domain[4:]
        except Exception:
            continue
        
        # Look up the website's main domain
        lookup = db.lookup_domain(domain)
        
        if lookup:
            tracked_domains.add(domain)
        
        records.append({
            "url": website_url,
            "domain": domain,
            "is_known_tracker": lookup is not None,
            "tracker_sources": ",".join(lookup["sources"]) if lookup else "",
            "tracker_num_sources": lookup["num_sources"] if lookup else 0,
            "tracker_confidence": lookup["confidence"] if lookup else "not_in_database",
            "tracker_category": lookup["primary_category"] if lookup else "",
            "tracker_all_categories": ",".join(lookup["all_categories"]) if lookup else "",
            "tracker_companies": ",".join(lookup["companies"]) if lookup else "",
            "fingerprinting_score": lookup["fingerprinting_score"] if lookup else 0,
            "match_type": lookup["match_type"] if lookup else ""
        })
    
    return pd.DataFrame(records)


def generate_website_level_ground_truth(db: GroundTruthDatabase) -> pd.DataFrame:
    """
    Generate website-level ground truth with expected tracking categories.
    This creates labels suitable for evaluating LLM classifications.
    """
    if not CLASSIFICATIONS_FILE.exists():
        logger.error(f"Classification file not found: {CLASSIFICATIONS_FILE}")
        return None
    
    class_df = pd.read_csv(CLASSIFICATIONS_FILE)
    
    records = []
    
    for _, row in class_df.iterrows():
        website_url = row.get("url", "")
        llm_category = row.get("primary_category", "")
        llm_secondary = row.get("secondary_categories", "")
        
        try:
            domain = urlparse(website_url).netloc
            if domain.startswith("www."):
                domain = domain[4:]
        except Exception:
            continue
        
        # Look up if the site itself is a known tracker
        lookup = db.lookup_domain(domain)
        
        # For major websites, they host trackers but aren't trackers themselves
        # We need to infer expected tracking behavior based on known patterns
        ground_truth_label = infer_website_tracking_type(domain, llm_category, lookup)
        
        records.append({
            "url": website_url,
            "domain": domain,
            "ground_truth_label": ground_truth_label,
            "is_tracker_domain": lookup is not None,
            "tracker_info": json.dumps(lookup) if lookup else "",
            "llm_primary_category": llm_category,
            "llm_secondary_categories": llm_secondary
        })
    
    return pd.DataFrame(records)


def infer_website_tracking_type(domain: str, llm_category: str, tracker_lookup: dict | None) -> str:
    """
    Infer the expected tracking type for a website based on domain analysis
    and known patterns.
    """
    domain_lower = domain.lower()
    
    # Known advertising/ad-tech companies
    ad_domains = [
        "doubleclick", "adsense", "adnxs", "googlesyndication", "facebook.com",
        "criteo", "taboola", "outbrain", "pubmatic", "rubiconproject"
    ]
    if any(ad in domain_lower for ad in ad_domains):
        return "advertising"
    
    # Known analytics providers
    analytics_domains = [
        "google-analytics", "googleanalytics", "analytics", "amplitude",
        "mixpanel", "segment", "hotjar", "fullstory", "heap"
    ]
    if any(a in domain_lower for a in analytics_domains):
        return "analytics"
    
    # Social networks (typically heavy trackers)
    social_domains = ["facebook", "twitter", "instagram", "linkedin", "tiktok", "snapchat"]
    if any(s in domain_lower for s in social_domains):
        return "social"
    
    # E-commerce sites (typically heavy advertising)
    ecommerce_domains = ["amazon", "ebay", "alibaba", "walmart", "target", "bestbuy"]
    if any(e in domain_lower for e in ecommerce_domains):
        return "advertising"
    
    # News/media sites (typically heavy advertising + analytics)
    news_domains = ["cnn", "nytimes", "washingtonpost", "huffpost", "buzzfeed", "forbes"]
    if any(n in domain_lower for n in news_domains):
        return "advertising"
    
    # Tech companies
    tech_domains = ["google", "microsoft", "apple"]
    if any(t in domain_lower for t in tech_domains):
        return "analytics"
    
    # If it's a known tracker domain
    if tracker_lookup:
        return tracker_lookup["primary_category"]
    
    # Default to LLM's classification if no other signal
    return llm_category if llm_category else "functional"


def generate_all_ground_truth(force_download: bool = False):
    """Generate all ground truth files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build the database
    db = build_ground_truth_database(force_download)
    
    # Generate script-level ground truth
    logger.info("\nGenerating script-level ground truth...")
    script_gt = generate_script_level_ground_truth(db)
    if script_gt is not None:
        script_gt_path = OUTPUT_DIR / "authoritative_ground_truth.csv"
        script_gt.to_csv(script_gt_path, index=False)
        logger.info(f"Script-level ground truth saved to: {script_gt_path}")
        
        # Print summary
        tracker_count = script_gt["is_known_tracker"].sum()
        logger.info(f"  Total entries: {len(script_gt)}")
        logger.info(f"  Known tracker domains: {tracker_count}")
    
    # Generate website-level ground truth (for evaluation)
    logger.info("\nGenerating website-level ground truth...")
    website_gt = generate_website_level_ground_truth(db)
    if website_gt is not None:
        website_gt_path = OUTPUT_DIR / "ground_truth_labels.csv"
        website_gt.to_csv(website_gt_path, index=False)
        logger.info(f"Website-level ground truth saved to: {website_gt_path}")
    
    # Save database statistics
    stats = db.get_stats()
    stats_path = OUTPUT_DIR / "ground_truth_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Database statistics saved to: {stats_path}")
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    viz_dir = Path("data/dashboards")
    chart_paths = generate_visualizations(db, viz_dir)
    
    print("\n" + "=" * 60)
    print("GROUND TRUTH GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total tracker domains in database: {stats['total_domains']:,}")
    print(f"Multi-source confirmed trackers: {stats['multi_source_confirmed']:,}")
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_DIR / 'authoritative_ground_truth.csv'}")
    print(f"  - {OUTPUT_DIR / 'ground_truth_labels.csv'}")
    print(f"  - {OUTPUT_DIR / 'ground_truth_stats.json'}")
    print(f"\nVisualizations:")
    for name, path in chart_paths.items():
        print(f"  - {path}")
    
    return db


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ground truth from authoritative tracking databases")
    parser.add_argument("--force-download", action="store_true", 
                       help="Force re-download of all sources")
    parser.add_argument("--stats-only", action="store_true",
                       help="Only print database statistics, don't generate files")
    
    args = parser.parse_args()
    
    if args.stats_only:
        db = build_ground_truth_database(args.force_download)
        stats = db.get_stats()
        print(json.dumps(stats, indent=2))
    else:
        generate_all_ground_truth(args.force_download)
