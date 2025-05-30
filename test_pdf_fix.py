#!/usr/bin/env python3
"""
Test PDF functionality fix
"""

import requests

def test_pdf_status():
    """Check if PDF is loaded in memory"""
    try:
        response = requests.get("http://localhost:8000/pdf_info")
        if response.status_code == 200:
            data = response.json()
            if data.get('has_pdf'):
                print(f"✅ PDF loaded: {data['filename']}")
                print(f"   Pages: {data['pages']}")
                print(f"   Words: {data['word_count']}")
                return True
            else:
                print("❌ No PDF currently loaded")
                return False
    except Exception as e:
        print(f"Error checking PDF status: {e}")
        return False

def test_pdf_query(query):
    """Test a query against the API"""
    try:
        response = requests.post("http://localhost:8000/query", json={"query": query})
        if response.status_code == 200:
            data = response.json()
            answer = data.get('answer', '')
            print(f"\nQuery: {query}")
            print(f"Response preview: {answer[:200]}...")
            
            # Check if PDF context was used
            if 'Analyzing:' in answer or 'PDF' in answer:
                print("✅ PDF context detected in response")
            else:
                print("⚠️  No clear PDF context in response")
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("🔍 Testing PDF Fix...")
    print("=" * 50)
    
    # Check if PDF is loaded
    has_pdf = test_pdf_status()
    
    if has_pdf:
        print("\n📝 Testing follow-up queries...")
        
        # Test various query formats
        test_queries = [
            "tell me about this pdf",
            "what is in the file?",
            "summarize the document",
            "what does it say?",
            "tell me about the resume",
            "what are the key points?"
        ]
        
        for query in test_queries:
            test_pdf_query(query)
            print("-" * 30)
    else:
        print("\n⚠️  No PDF loaded. Please upload a PDF first.")
    
    print("\n💡 Troubleshooting tips:")
    print("1. Make sure PyPDF2 is installed: pip3 install PyPDF2")
    print("2. Check API logs for debug messages")
    print("3. Restart the API after making changes")
    print("4. Ensure the PDF file is valid and not corrupted")