#!/usr/bin/env python3
"""
LLM-based tracking script classifier using Mfeatures (mega_features) data.
"""
import os
import anthropic
import pandas as pd
from pathlib import Path
import json
from typing import Dict
import time
import re

class TrackingScriptClassifier:
    """Classify tracking scripts using LLM based on Mfeatures data."""
    
    def __init__(self, model="claude-haiku-4-5", api_key=None):
        """Initialize classifier with cost-effective model."""
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv('ANTHROPIC_API_KEY'))
        self.model = model
        
    def analyze_features(self, features_df: pd.DataFrame) -> Dict:
        """Analyze feature data and extract key patterns."""
        if len(features_df) == 0:
            return {}
        
        # Get top features by usage count
        top_features = features_df.nlargest(30, 'usage_count')[
            ['feature_name', 'receiver_name', 'member_name', 'usage_count', 'usage_mode']
        ].to_dict('records')
        
        # Count unique features
        unique_features = features_df['feature_name'].nunique()
        total_uses = features_df['usage_count'].sum()
        
        # Identify tracking-related patterns
        tracking_keywords = {
            'fingerprinting': ['Canvas', 'WebGL', 'Navigator', 'Screen', 'Battery', 'Font', 'Audio'],
            'analytics': ['Performance', 'Timing', 'Beacon', 'Analytics', 'Track'],
            'advertising': ['Ad', 'Cookie', 'Storage', 'Pixel'],
            'privacy_invasive': ['Geolocation', 'Camera', 'Microphone', 'Bluetooth']
        }
        
        detected_patterns = {}
        for category, keywords in tracking_keywords.items():
            matches = features_df[
                features_df['feature_name'].str.contains('|'.join(keywords), case=False, na=False)
            ]
            if len(matches) > 0:
                detected_patterns[category] = {
                    'count': len(matches),
                    'examples': matches['feature_name'].unique()[:5].tolist()
                }
        
        return {
            'top_features': top_features,
            'unique_features': unique_features,
            'total_uses': int(total_uses),
            'detected_patterns': detected_patterns
        }
    
    def create_classification_prompt(self, analysis: Dict, url: str) -> str:
        """Create prompt for classification based on feature analysis."""
        
        if not analysis:
            return ""
        
        # Format top features
        features_text = []
        for feat in analysis['top_features'][:20]:
            features_text.append(
                f"- {feat['feature_name']} (used {feat['usage_count']} times, mode: {feat['usage_mode']})"
            )
        
        # Format detected patterns
        patterns_text = []
        for category, data in analysis.get('detected_patterns', {}).items():
            patterns_text.append(f"- {category.upper()}: {data['count']} features detected")
            patterns_text.append(f"  Examples: {', '.join(data['examples'][:3])}")
        
        prompt = f"""You are an expert in web privacy and tracking script analysis.
Analyze the following JavaScript API usage from a website and classify the tracking behavior.

URL: {url}

=== FEATURE USAGE SUMMARY ===
Total unique features: {analysis['unique_features']}
Total feature uses: {analysis['total_uses']}

=== TOP 20 MOST USED FEATURES ===
{chr(10).join(features_text)}

=== DETECTED TRACKING PATTERNS ===
{chr(10).join(patterns_text) if patterns_text else 'No obvious tracking patterns detected'}

=== CLASSIFICATION TASK ===
Based on these JavaScript API calls, classify the tracking behavior into one or more categories:

1. **Fingerprinting**: Collecting device/browser characteristics for unique identification
   - Look for: Canvas, WebGL, Navigator properties, Screen properties, Battery, Fonts, Audio fingerprinting
   
2. **Analytics**: Tracking user behavior for analytics purposes
   - Look for: Performance timing, Page visibility, Mouse/keyboard events, Scroll tracking, Beacons
   
3. **Advertising**: Tracking for ad targeting or measurement
   - Look for: Third-party cookies, Tracking pixels, Ad-related APIs, Storage APIs for tracking
   
4. **Benign**: Minimal or no tracking behavior
   - Look for: Basic functionality, No privacy-invasive APIs, Standard web features

Provide your response in the following JSON format:
{{
    "primary_category": "one of: fingerprinting, analytics, advertising, benign",
    "secondary_categories": ["list of other applicable categories"],
    "confidence": "high, medium, or low",
    "explanation": "Brief explanation of why you classified it this way (2-3 sentences)",
    "key_indicators": ["List 3-5 specific features that indicate tracking"],
    "privacy_risk": "low, medium, or high",
    "tracking_intensity": "none, light, moderate, heavy"
}}
"""
        return prompt
    
    def classify(self, features_df: pd.DataFrame, url: str) -> Dict:
        """Classify tracking behavior based on features."""
        
        if len(features_df) == 0:
            return {
                "primary_category": "unknown",
                "secondary_categories": [],
                "confidence": "low",
                "explanation": "No feature data available for classification",
                "key_indicators": [],
                "privacy_risk": "unknown",
                "tracking_intensity": "unknown"
            }
        
        # Analyze features first
        analysis = self.analyze_features(features_df)
        
        # Create prompt
        prompt = self.create_classification_prompt(analysis, url)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system="You are an expert in web privacy and tracking script analysis. Always respond with valid JSON only.",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            # Extract JSON from response
            response_text = response.content[0].text
            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response_text)
            return result
            
        except Exception as e:
            print(f"    Error during classification: {e}")
            return {
                "primary_category": "error",
                "secondary_categories": [],
                "confidence": "low",
                "explanation": f"Error: {str(e)}",
                "key_indicators": [],
                "privacy_risk": "unknown",
                "tracking_intensity": "unknown"
            }
    
    def generate_explanation(self, features_df: pd.DataFrame, url: str) -> str:
        """Generate detailed explanation of tracking behavior."""
        
        if len(features_df) == 0:
            return "No feature data available to generate explanation."
        
        analysis = self.analyze_features(features_df)
        
        # Format top features
        features_text = []
        for feat in analysis['top_features'][:15]:
            features_text.append(f"- {feat['feature_name']} ({feat['usage_count']} uses)")
        
        prompt = f"""You are an expert in web privacy. Explain in simple, human-readable language what tracking is happening on this website based on the JavaScript features being used.

URL: {url}

Total unique features: {analysis['unique_features']}
Total feature uses: {analysis['total_uses']}

Top JavaScript Features Used:
{chr(10).join(features_text)}

Provide a clear, concise explanation (2-3 paragraphs) that a non-technical user could understand. Focus on:
1. What data is being collected
2. How it's being collected (which APIs/features)
3. Potential privacy implications
4. Whether this is typical for this type of website

Be specific about which features indicate which types of tracking."""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system="You are an expert in web privacy who explains technical concepts clearly.",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error generating explanation: {str(e)}"

def process_dataset(data_dir='data/processed', output_file='data/processed/llm_classifications.csv', limit=None):
    """Process entire dataset with LLM classifier."""
    
    # Check API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set!")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return None
    
    classifier = TrackingScriptClassifier()
    
    # Load crawls summary
    crawls_df = pd.read_csv(Path(data_dir) / 'crawls_summary.csv')
    
    # Filter to only crawls with Mfeatures post-processor
    crawls_df = crawls_df[crawls_df['postprocessor_used'].str.contains('Mfeatures', na=False)].copy()
    
    # Deduplicate by URL (keep the most recent crawl so dashboards don't double count)
    original_count = len(crawls_df)
    crawls_df['start_time'] = pd.to_datetime(crawls_df['start_time'], errors='coerce')
    crawls_df = (
        crawls_df
        .sort_values(['start_time', 'id'])
        .drop_duplicates(subset='url', keep='last')
        .reset_index(drop=True)
    )
    deduped_count = len(crawls_df)
    if deduped_count != original_count:
        print(f"ℹ️  Removed {original_count - deduped_count} duplicate crawl entries (kept latest per URL)")
    
    if limit:
        crawls_df = crawls_df.head(limit)
    
    results = []
    
    print(f"Processing {len(crawls_df)} websites...\n")
    print("="*80)
    
    for idx, row in crawls_df.iterrows():
        url = row['url']
        submission_id = row['id']
        
        print(f"\n[{idx+1}/{len(crawls_df)}] {url}")
        print("-"*80)
        
        # Load mega_features for this URL
        safe_filename = url.replace('https://', '').replace('http://', '').replace('/', '_').replace(':', '_')
        features_file = Path(data_dir) / f'mega_features_{safe_filename}.csv'
        
        if not features_file.exists():
            print(f"  ⚠️  Features file not found: {features_file.name}")
            continue
        
        try:
            features_df = pd.read_csv(features_file)
            
            if len(features_df) == 0:
                print(f"  ⚠️  No features in file")
                continue
            
            print(f"  📊 Loaded {len(features_df)} features, {features_df['usage_count'].sum():.0f} total uses")
            
            # Classify
            print(f"  🤖 Classifying with LLM...")
            classification = classifier.classify(features_df, url)
            
            # Generate explanation
            print(f"  📝 Generating explanation...")
            explanation = classifier.generate_explanation(features_df, url)
            
            # Store results
            result = {
                'url': url,
                'submission_id': submission_id,
                'primary_category': classification.get('primary_category', 'unknown'),
                'secondary_categories': ','.join(classification.get('secondary_categories', [])),
                'confidence': classification.get('confidence', 'unknown'),
                'privacy_risk': classification.get('privacy_risk', 'unknown'),
                'tracking_intensity': classification.get('tracking_intensity', 'unknown'),
                'classification_explanation': classification.get('explanation', ''),
                'key_indicators': ','.join(classification.get('key_indicators', [])),
                'detailed_explanation': explanation,
                'num_features': len(features_df),
                'total_feature_uses': int(features_df['usage_count'].sum())
            }
            
            results.append(result)
            
            print(f"  ✅ Result: {classification.get('primary_category')} "
                  f"({classification.get('confidence')} confidence, "
                  f"{classification.get('privacy_risk')} risk, "
                  f"{classification.get('tracking_intensity')} intensity)")
            
            # Rate limiting - be nice to the API
            time.sleep(2)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"\n{'='*80}")
        print(f"✅ Saved {len(results)} classifications to {output_file}")
        print(f"{'='*80}")
        
        # Print summary
        print("\n" + "="*80)
        print("CLASSIFICATION SUMMARY")
        print("="*80)
        print(f"\nTotal websites classified: {len(results)}")
        
        print(f"\n📊 Category Distribution:")
        print(results_df['primary_category'].value_counts().to_string())
        
        print(f"\n🎯 Confidence Distribution:")
        print(results_df['confidence'].value_counts().to_string())
        
        print(f"\n⚠️  Privacy Risk Distribution:")
        print(results_df['privacy_risk'].value_counts().to_string())
        
        print(f"\n🔥 Tracking Intensity Distribution:")
        print(results_df['tracking_intensity'].value_counts().to_string())
        
        print(f"\n📈 Statistics:")
        print(f"  Average features per site: {results_df['num_features'].mean():.0f}")
        print(f"  Average feature uses per site: {results_df['total_feature_uses'].mean():.0f}")
        print(f"  Max features: {results_df['num_features'].max()}")
        print(f"  Min features: {results_df['num_features'].min()}")
        
        print("\n" + "="*80)
        
        return results_df
    else:
        print("\n⚠️  No results to save!")
        return None

if __name__ == '__main__':
    import sys
    
    # Optional: limit number of sites for testing
    # Usage: python llm_classifier_final.py 5  (to process only 5 sites)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    
    if limit:
        print(f"🧪 TEST MODE: Processing only {limit} websites\n")
    
    results_df = process_dataset(limit=limit)
