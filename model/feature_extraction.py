#!/usr/bin/env python3
"""
Feature Extraction Module for Phishing Website Detection
Extracts various features from URLs for model training and prediction.
"""

import re
import socket
import ssl
from urllib.parse import urlparse
import random  # Used for simulating some features in demo

def extract_features_from_url(url):
    """
    Extract features from a URL for phishing detection.
    
    Args:
        url (str): The URL to analyze
        
    Returns:
        list: A list of extracted features
    """
    # Initialize features with default values
    features = {
        'url_length': 0,
        'domain_length': 0,
        'has_ip': 0,
        'has_at': 0,
        'redirects': 0,
        'prefix_suffix': 0,
        'dots_count': 0,
        'suspicious_tld': 0,
        'has_https': 0,
        'domain_age_days': 0,
        'has_form': 0,
        'has_iframe': 0
    }
    
    try:
        # Basic URL features
        features['url_length'] = len(url)
        
        # Parse the URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Domain features
        features['domain_length'] = len(domain)
        features['has_https'] = 1 if parsed_url.scheme == 'https' else 0
        
        # Count dots in domain
        features['dots_count'] = domain.count('.')
        
        # Check for @ symbol in URL (common in phishing)
        features['has_at'] = 1 if '@' in url else 0
        
        # Check for IP address instead of domain
        try:
            socket.inet_aton(domain)
            features['has_ip'] = 1
        except:
            # Check if domain has digits and dashes (potential IP obfuscation)
            if bool(re.search(r'\d+\.\d+\.\d+\.\d+', domain)):
                features['has_ip'] = 1
        
        # Check for suspicious TLDs
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.top', '.xyz']
        features['suspicious_tld'] = 1 if any(tld in domain for tld in suspicious_tlds) else 0
        
        # Check for prefix/suffix in domain (e.g., paypal-secure.com)
        common_brands = ['paypal', 'apple', 'microsoft', 'amazon', 'google', 'facebook', 'netflix']
        features['prefix_suffix'] = 1 if any(brand in domain and not domain.startswith(f"www.{brand}.") and not domain == f"{brand}.com" for brand in common_brands) else 0
        
        # Check for redirects (simplified for demo)
        features['redirects'] = 1 if 'redirect' in url or 'forward' in url else 0
        
        # Domain age (simplified for demo)
        # In a real implementation, you'd use WHOIS data
        # Here we're simulating for demonstration purposes
        if features['has_ip'] or features['suspicious_tld'] or features['prefix_suffix']:
            # Suspicious domains tend to be newer
            features['domain_age_days'] = random.randint(1, 30)
        else:
            # Legitimate domains tend to be older
            features['domain_age_days'] = random.randint(100, 1000)
        
        # HTML/JS features (simplified for demo)
        # In a real implementation, you'd fetch the page and analyze its HTML
        # Here we're simulating for demonstration purposes
        if features['has_ip'] or features['suspicious_tld'] or features['prefix_suffix']:
            # Phishing sites often have forms to steal credentials
            features['has_form'] = random.choice([0, 1, 1, 1])
            features['has_iframe'] = random.choice([0, 1, 1])
        else:
            features['has_form'] = random.choice([0, 1, 0])
            features['has_iframe'] = random.choice([0, 0, 1])
            
    except Exception as e:
        print(f"Error extracting features from URL {url}: {e}")
    
    # Return features as a list in a consistent order
    return [
        features['url_length'],
        features['domain_length'],
        features['has_ip'],
        features['has_at'],
        features['redirects'],
        features['prefix_suffix'],
        features['dots_count'],
        features['suspicious_tld'],
        features['has_https'],
        features['domain_age_days'],
        features['has_form'],
        features['has_iframe']
    ]

def extract_features_dict(url):
    """
    Extract features from a URL and return as a dictionary.
    This is used by the API for predictions.
    
    Args:
        url (str): The URL to analyze
        
    Returns:
        dict: A dictionary of extracted features
    """
    feature_list = extract_features_from_url(url)
    feature_names = [
        'url_length', 'domain_length', 'has_ip', 'has_at', 'redirects',
        'prefix_suffix', 'dots_count', 'suspicious_tld', 'has_https',
        'domain_age_days', 'has_form', 'has_iframe'
    ]
    
    return dict(zip(feature_names, feature_list))

# Test the feature extraction
if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "http://malicious-phishing-site.com",
        "https://paypal-secure.suspicious-domain.com"
    ]
    
    for url in test_urls:
        features = extract_features_dict(url)
        print(f"\nFeatures for {url}:")
        for feature, value in features.items():
            print(f"  {feature}: {value}")