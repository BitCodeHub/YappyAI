#!/usr/bin/env python3
"""
Test PDF Q&A functionality
"""

import requests
import base64

# Create a simple test PDF content (you would normally read from a file)
def create_test_pdf_base64():
    # This is a placeholder - in real usage, you'd read an actual PDF file
    # and convert it to base64
    with open("test.pdf", "rb") as pdf_file:
        return base64.b64encode(pdf_file.read()).decode()

def test_pdf_qa():
    url = "http://localhost:8000/query"
    
    print("🚀 Testing PDF Q&A Functionality...")
    print("=" * 50)
    
    # Test 1: Initial PDF upload with summary request
    print("\n1. Testing PDF Upload with Summary Request")
    
    # For this test, we'll simulate a PDF upload
    # In real usage, you'd have actual PDF content
    test_pdf_content = """SAMPLE PDF CONTENT
    
Company Annual Report 2023

Executive Summary:
Our company achieved record revenue of $150 million in 2023, representing a 25% increase from 2022. 
Key achievements include:
- Launched 3 new products
- Expanded to 5 new markets
- Increased customer base by 40%

Financial Highlights:
- Revenue: $150 million
- Profit: $30 million
- R&D Investment: $20 million

Future Outlook:
We expect continued growth in 2024 with projected revenue of $200 million.
    """
    
    # Simulate base64 encoded PDF (in reality, this would be actual PDF binary data)
    fake_pdf_base64 = base64.b64encode(test_pdf_content.encode()).decode()
    
    payload = {
        "query": "Please summarize this PDF and highlight the key financial data",
        "file_data": {
            "name": "annual_report_2023.pdf",
            "type": "application/pdf",
            "content": f"data:application/pdf;base64,{fake_pdf_base64}"
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            print("✅ PDF uploaded successfully!")
            print(f"Response preview: {data['answer'][:200]}...")
        else:
            print(f"❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Follow-up question about the PDF
    print("\n\n2. Testing Follow-up Question")
    
    payload2 = {
        "query": "What was the profit mentioned in the PDF?"
    }
    
    try:
        response = requests.post(url, json=payload2)
        if response.status_code == 200:
            data = response.json()
            print("✅ Follow-up question answered!")
            print(f"Response: {data['answer'][:200]}...")
        else:
            print(f"❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Search within PDF
    print("\n\n3. Testing Search Functionality")
    
    payload3 = {
        "query": "Search for all mentions of 'revenue' in the document"
    }
    
    try:
        response = requests.post(url, json=payload3)
        if response.status_code == 200:
            data = response.json()
            print("✅ Search completed!")
            print(f"Response: {data['answer'][:200]}...")
        else:
            print(f"❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("📄 PDF Q&A Testing Complete!")
    print("\n💡 PDF Q&A Features:")
    print("- Upload any PDF and ask questions")
    print("- Automatic text extraction and analysis")
    print("- Page-by-page content access")
    print("- Metadata extraction (title, author, etc.)")
    print("- Persistent context for follow-up questions")
    print("- Search within document")
    print("- Summary generation")

if __name__ == "__main__":
    test_pdf_qa()