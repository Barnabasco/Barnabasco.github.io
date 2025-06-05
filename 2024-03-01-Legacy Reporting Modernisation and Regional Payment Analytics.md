---
layout: post
title: Legacy Reporting Modernisation and Regional Payment Analytics
subtitle: Transforming financial reporting infrastructure
author: Chimaobi Barnabas Opara
categories: business-intelligence data-engineering finance
banner:
  video: https://vjs.zencdn.net/v/oceans.mp4
  loop: true
  volume: 0.8
  start_at: 8.5
  image: https://bit.ly/3xTmdUP
  opacity: 0.618
  background: "#000"
  height: "100vh"
  min_height: "38vh"
  heading_style: "font-size: 4.25em; font-weight: bold; text-decoration: underline"
  subheading_style: "color: gold"
tags: ssrs etl data-migration financial-analytics
top: 1
sidebar: []
---

## Project Overview:
Migrated 50+ legacy Excel reports to a real-time SSRS platform serving 5,000+ users, reducing report generation time from 8 hours to 15 minutes while improving data accuracy by 99.2%. Implemented regional payment analytics that identified $4.8M in recoverable revenue.

## Migration Architecture
### Legacy Reporting Modernization Pipeline

1. 📥 **Extraction: Process Legacy Excel Files**  
   *Tools: Python Pandas, SQL Server Integration Services (SSIS)*  
   *Process: Automated nightly extraction from 50+ Excel workbooks*
   
2. 🔄 **Transformation: ETL Pipeline Processing**  
   *Components: Data validation, Type conversion, Business rule application*  
   *Features: Error logging, Auto-correction of formatting issues*
   
3. 💾 **Storage: SQL Server Data Warehouse**  
   *Architecture: Star schema with fact/dimension tables*  
   *Features: ACID compliance, Columnstore indexing, Daily snapshots*
   
4. 📄 **Reporting: SSRS Report Generation**  
   *Output: 50+ parameterized operational reports*  
   *Features: Role-based security, Subscription delivery, Drill-through capability*
   
5. 🌐 **Delivery: Web Portal Access**  
   *Features: Single sign-on (SSO), Mobile-responsive design, Export to PDF/Excel*  
   *Users: 5,000+ finance and operations staff*
   
6. 📊 **Analytics: Power BI Dashboards**  
   *Components: Regional payment trends, Recovery opportunity heatmaps, AR aging analysis*  
   *Features: Real-time refresh, Natural language Q&A*


### Stage 1: Extraction

``` python
# extraction script
import pandas as pd
from sqlalchemy import create_engine

def extract_excel_to_staging(file_path):
    df = pd.read_excel(file_path, sheet_name='Financials')
    # Handle merged cells and header rows
    df = df.iloc[3:].dropna(how='all')
    df.columns = ['region', 'account', 'due_date', 'amount']
    return df
```

### ETL Pipeline:

```python
def excel_to_sql(excel_file, sql_table):
    df = pd.read_excel(excel_file, sheet_name=None)
    with sql_engine.connect() as conn:
        for sheet, data in df.items():
            data.to_sql(f'{sql_table}_{sheet}', conn, if_exists='replace')
```
### Stage 2: Transformation

```sql
-- Data cleansing 
UPDATE staging_table
SET amount = TRY_CAST(REPLACE(amount, '$', '') AS DECIMAL(18,2))
WHERE ISNUMERIC(REPLACE(amount, '$', '')) = 1;

```

### Stage 3: Storage
Data Warehouse Schema:

```mermaid
┌────────────┐     ┌──────────────┐
│ Dim_Date   │◄────┤ Fact_Payments│
├────────────┤     ├──────────────┤
│ date_key   │     │ payment_key  │
│ full_date  │     │ date_key     │
│ fiscal_yr  │     │ account_key  │
└────────────┘     │ region_key   │
                   │ amount       │
┌────────────┐     └──────────────┘
│ Dim_Region │▲           ▲
├────────────┤└───────────┘
│ region_key │
│ region_name│
└────────────┘
```

### Stage 4: Reporting
SSRS Report Features:
* Dynamic column sorting
* Multi-level grouping
* Conditional formatting
* Cascading parameters

### Stage 5: Delivery

```mermaid
┌────────────────────┐

│ SSRS Report Server │

└─────────┬──────────┘

│

├───────────────────────┐

│                       │

▼                       ▼

┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐

│ Corporate Portal   │   │ Email Subscriptions│   │ Mobile App         │

└─────────┬──────────┘   └─────────┬──────────┘   └────────────────────┘

│                       │

▼                       ▼

┌────────────────────┐   ┌────────────────────┐

│ Department Managers│   │ Finance Team       │

└────────────────────┘   └────────────────────┘
```



### Stage 6: Analytics

```sql
-- Regional revenue recovery analysis
WITH regional_data AS (
    SELECT 
        region,
        SUM(recoverable_amount) AS potential_recovery,
        DATEDIFF(day, MAX(due_date), GETDATE()) AS days_overdue
    FROM payments
    WHERE payment_status = 'delinquent'
    GROUP BY region
)
SELECT 
    region,
    potential_recovery,
    days_overdue,
    potential_recovery * (0.01 * days_overdue) AS recovery_score
FROM regional_data
ORDER BY recovery_score DESC;
```

# Key Findings:

* Region1: $1.2M recoverable (32% of total)

Migration Impact Metrics:

| Pipeline Stage |Before Migration |	After Migration	| Improvement        |
|----------------|-----------------|--------------------|--------------------|
| Data Extraction| 4 hours manual  | 15 min automated   | 94% faster         | 
| Validation	 |Error-prone	   |Automated checks    |99.8% accuracy      |
| Report Access	 |Email attachments|Self-service portal |24/7 availability   |
|Payment Insights|	Monthly static |Real-time dashboards|63% faster decisions|



## SSRS Report Features:

* Parameterized region filters

* Drill-down payment details

* Scheduled email delivery

* Mobile-responsive design

## Project Outcomes:

* Identified $4.8M recoverable revenue

* Reduced reporting errors by 84%

* Enabled 24/7 self-service reporting
