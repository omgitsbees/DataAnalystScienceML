import pandas as pd 
import matplotlib.pyplot as plt
import schedule
import time

def generate_report():
    # Step 1: Data Extraction
    data = pd.read_csv('data.csv')
    
    # Step 2: Data Processing
    # Example: Aggregating data by a specific column
    report_data = data.groupby('category').sum()
    
    # Step 3: Report Generation
    plt.figure(figsize=(10, 6))
    report_data.plot(kind='bar')
    plt.title('Category-wise Data Summary')
    plt.xlabel('Category')
    plt.ylabel('Values')
    plt.tight_layout()
    plt.savefig('report.pdf')
    
    print("Report generated successfully.")
    
# Step 4: Automation
# Schedule the report generation to run every day at a specific time
schedule.every().day.at("10:00").do(generate_report)

while True:
    schedule.run_pending()
    time.sleep(1)