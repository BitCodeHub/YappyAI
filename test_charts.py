#!/usr/bin/env python3
"""
Test chart generation directly
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

# Test data
test_data = [
    {"Name": "John", "Age": 25, "Salary": 50000},
    {"Name": "Mary", "Age": 30, "Salary": 60000},
    {"Name": "Bob", "Age": 35, "Salary": 70000}
]

numeric_columns = ["Age", "Salary"]

def test_chart_generation():
    try:
        print("Testing chart generation...")
        
        # Create simple bar chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Get data for chart
        ages = [row["Age"] for row in test_data]
        names = [row["Name"] for row in test_data]
        
        colors = sns.color_palette("viridis", len(ages))
        bars = ax.bar(names, ages, color=colors)
        ax.set_title('Age Distribution', fontweight='bold', fontsize=14)
        ax.set_ylabel('Age')
        
        # Add value labels
        for bar, age in zip(bars, ages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{age}', ha='center', va='bottom', fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode()
        plt.close()
        
        print(f"Chart generated successfully! Base64 length: {len(img_data)}")
        print(f"Base64 starts with: {img_data[:50]}...")
        
        return f"data:image/png;base64,{img_data}"
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_chart_generation()
    if result:
        print("✅ Chart generation working!")
        # Write to HTML file for testing
        html_content = f"""
<html>
<body>
<h1>Test Chart</h1>
<img src="{result}" alt="Test Chart" />
</body>
</html>
        """
        with open("/Users/jimmylam/Downloads/agenticSeek-main/test_chart.html", "w") as f:
            f.write(html_content)
        print("Test chart saved to test_chart.html")
    else:
        print("❌ Chart generation failed!")