#!/usr/bin/env python3
"""
Test advanced chart generation capabilities
"""

import requests
import json

# Test data for different diagram types
test_cases = [
    {
        "name": "Knowledge Graph Test",
        "query": "show me a knowledge graph of relationships in this data",
        "csv_content": """Person,Company,Role,Department
John,TechCorp,Manager,Engineering
Mary,TechCorp,Developer,Engineering
Bob,DataCorp,Analyst,Analytics
Alice,TechCorp,Designer,UX
Charlie,DataCorp,Manager,Analytics"""
    },
    {
        "name": "Flow Diagram Test", 
        "query": "create a process flow diagram",
        "csv_content": """Step,Process,Duration,Owner
1,Planning,2 days,PM
2,Development,5 days,Engineering
3,Testing,2 days,QA
4,Deployment,1 day,DevOps
5,Monitoring,Ongoing,Operations"""
    },
    {
        "name": "Cause-Effect Test",
        "query": "generate a cause and effect fishbone diagram",
        "csv_content": """Category,Factor,Impact,Frequency
People,Training,High,Often
Process,Documentation,Medium,Sometimes
Technology,Outdated Systems,High,Always
Environment,Workspace,Low,Rarely"""
    },
    {
        "name": "Timeline Test",
        "query": "show this data as a timeline",
        "csv_content": """Date,Event,Type,Impact
2024-01,Project Start,Milestone,High
2024-02,Requirements,Phase,Medium
2024-03,Development,Phase,High
2024-04,Testing,Phase,Medium
2024-05,Launch,Milestone,High"""
    },
    {
        "name": "Heatmap Test",
        "query": "create a correlation heatmap",
        "csv_content": """Product,Sales,Marketing,Support,Development
A,100,50,20,80
B,150,70,30,90
C,200,90,40,100
D,120,60,25,85"""
    }
]

def test_advanced_charts():
    url = "http://localhost:8000/query"
    
    print("🚀 Testing Advanced Chart Generation...")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        
        payload = {
            "query": test_case['query'],
            "file_data": {
                "name": f"test_{i}.csv",
                "type": "text/csv", 
                "content": test_case['csv_content']
            }
        }
        
        try:
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get('answer', '')
                
                # Count images
                image_count = answer.count('data:image/png;base64,')
                
                print(f"✅ Success! Generated {image_count} visualizations")
                
                # Check for specific diagram types mentioned
                if 'Knowledge Graph' in answer:
                    print("   🔗 Knowledge Graph detected")
                if 'Flow Diagram' in answer:
                    print("   🔄 Flow Diagram detected")
                if 'Cause & Effect' in answer or 'Fishbone' in answer:
                    print("   🐟 Cause-Effect Diagram detected")
                if 'Timeline' in answer:
                    print("   ⏰ Timeline detected")
                if 'Heatmap' in answer:
                    print("   🔥 Heatmap detected")
                
            else:
                print(f"❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Advanced Chart Testing Complete!")
    print("\n💡 Available Diagram Types:")
    print("- 🔗 Knowledge Graphs (relationships, network, connections)")
    print("- 🔄 Flow Diagrams (process, workflow, sequence)")
    print("- 🐟 Cause-Effect Diagrams (fishbone, ishikawa, causal)")
    print("- 🌳 Hierarchy Diagrams (tree, organization, structure)")
    print("- ⏰ Timeline Diagrams (chronology, sequence, history)")
    print("- 🔥 Heatmaps (correlation, matrix)")
    print("- 💧 Sankey Diagrams (flow, allocation)")

if __name__ == "__main__":
    test_advanced_charts()