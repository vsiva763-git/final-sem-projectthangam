#!/usr/bin/env python3
"""Test script to verify the web application is running"""

import urllib.request
import json
import sys

def test_health():
    """Test the health endpoint"""
    try:
        print("🔍 Testing web application health endpoint...")
        with urllib.request.urlopen('http://localhost:5000/health', timeout=5) as response:
            data = json.loads(response.read().decode())
            
            print("\n✅ Application is running successfully!")
            print("=" * 60)
            print(f"📊 Status: {data['status']}")
            print(f"🤖 Model loaded: {data['model_loaded']}")
            print(f"💻 Device: {data['device']}")
            print("=" * 60)
            
            return True
    except urllib.error.URLError as e:
        print(f"\n❌ Failed to connect to application: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def test_main_page():
    """Test the main page"""
    try:
        print("\n🔍 Testing main page...")
        with urllib.request.urlopen('http://localhost:5000/', timeout=5) as response:
            html = response.read().decode()
            if 'Speech Enhancement' in html:
                print("✅ Main page is accessible")
                return True
            else:
                print("⚠️  Main page returned unexpected content")
                return False
    except Exception as e:
        print(f"❌ Failed to access main page: {e}")
        return False

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🧪 Web Application Test Suite")
    print("=" * 60)
    
    health_ok = test_health()
    main_page_ok = test_main_page()
    
    print("\n" + "=" * 60)
    print("📋 Test Results:")
    print("=" * 60)
    print(f"Health endpoint: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Main page: {'✅ PASS' if main_page_ok else '❌ FAIL'}")
    print("=" * 60)
    
    if health_ok and main_page_ok:
        print("\n🎉 All tests passed! Application is running correctly.")
        print("🌐 You can access it at: http://localhost:5000")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please check the application.")
        sys.exit(1)
