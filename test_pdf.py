#!/usr/bin/env python3
"""Test PDF processing functionality"""

import base64
import io

# Test data structure similar to what the API receives
class FileData:
    def __init__(self, name, type, content):
        self.name = name
        self.type = type
        self.content = content

def test_pdf_extraction():
    """Test the PDF extraction function"""
    try:
        # Import the function from api_clean
        import sys
        sys.path.append('/Users/jimmylam/Downloads/agenticSeek-main')
        from api_clean import extract_pdf_content
        
        # Create a simple test PDF (base64 encoded)
        # This is a minimal valid PDF with "Hello World" text
        test_pdf_base64 = "JVBERi0xLjQKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwovUGFnZXMgMiAwIFIKPj4KZW5kb2JqCjIgMCBvYmoKPDwKL1R5cGUgL1BhZ2VzCi9LaWRzIFszIDAgUl0KL0NvdW50IDEKL01lZGlhQm94IFswIDAgNjEyIDc5Ml0KPj4KZW5kb2JqCjMgMCBvYmoKPDwKL1R5cGUgL1BhZ2UKL1BhcmVudCAyIDAgUgovUmVzb3VyY2VzIDw8Ci9Gb250IDw8Ci9GMSA0IDAgUgo+Pgo+PgovQ29udGVudHMgNSAwIFIKPj4KZW5kb2JqCjQgMCBvYmoKPDwKL1R5cGUgL0ZvbnQKL1N1YnR5cGUgL1R5cGUxCi9CYXNlRm9udCAvSGVsdmV0aWNhCj4+CmVuZG9iago1IDAgb2JqCjw8Ci9MZW5ndGggNDQKPj4Kc3RyZWFtCkJUCi9GMSAxMiBUZgo1MCA3MDAgVGQKKEhlbGxvIFdvcmxkKSBUagpFVAplbmRzdHJlYW0KZW5kb2JqCnhyZWYKMCA2CjAwMDAwMDAwMDAgNjU1MzUgZiAKMDAwMDAwMDAwOSAwMDAwMCBuIAowMDAwMDAwMDU2IDAwMDAwIG4gCjAwMDAwMDAxMzAgMDAwMDAgbiAKMDAwMDAwMDIyOSAwMDAwMCBuIAowMDAwMDAwMzE3IDAwMDAwIG4gCnRyYWlsZXIKPDwKL1NpemUgNgovUm9vdCAxIDAgUgo+PgpzdGFydHhyZWYKNDA5CiUlRU9G"
        
        # Create test file data
        test_file = FileData(
            name="test.pdf",
            type="application/pdf", 
            content=test_pdf_base64
        )
        
        print("Testing PDF extraction...")
        result = extract_pdf_content(test_file)
        
        if result:
            print("\n✅ PDF extraction successful!")
            print(f"Total pages: {result['total_pages']}")
            print(f"Word count: {result['word_count']}")
            print(f"Has text: {result['has_text']}")
            print(f"Text preview: {result['full_text'][:200]}...")
        else:
            print("\n❌ PDF extraction failed!")
            
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pdf_extraction()