"""Tests for trusted domains functionality"""
import pytest
from pathlib import Path
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verifai.utils.trusted_domains import load_trusted_domains


class TestTrustedDomains:
    def test_load_trusted_domains_returns_list(self):
        """Test that load_trusted_domains returns a list"""
        domains = load_trusted_domains()
        assert isinstance(domains, list)
    
    def test_load_trusted_domains_not_empty(self):
        """Test that we have at least some trusted domains"""
        domains = load_trusted_domains()
        assert len(domains) > 0
    
    def test_trusted_domains_are_valid(self):
        """Test that all domains are valid strings without protocols"""
        domains = load_trusted_domains()
        for domain in domains:
            assert isinstance(domain, str)
            assert "https://" not in domain
            assert "http://" not in domain
            assert len(domain) > 0
    
    def test_trusted_domains_include_expected_sources(self):
        """Test that expected sources are in the trusted domains list"""
        domains = load_trusted_domains()
        
        # Expected domains from the debank-sources.json
        expected_sources = [
            "stopfake.org",
            "bellingcat.com",
            "dfrlab.org"
        ]
        
        for expected in expected_sources:
            assert expected in domains, f"Expected domain {expected} not found in trusted domains"
    
    def test_trusted_domains_have_no_duplicates(self):
        """Test that there are no duplicate domains"""
        domains = load_trusted_domains()
        assert len(domains) == len(set(domains))

