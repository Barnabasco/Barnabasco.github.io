---
layout: post
title: Real-Time User Behaviour Analytics for Platform Optimisation
subtitle: Enhancing digital experience through behavioral insights
author: Chimaobi Barnabas Opara
categories: web-analytics big-data user-experience
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
tags: tableau clickstream heatmaps user-engagement
top: 1
sidebar: []
---

## Project Overview:
Leveraged big data analytics to process 2.5M+ daily user events, identifying usability bottlenecks through clickstream analysis, heatmaps, and session recordings. Implemented real-time dashboards that reduced bounce rate by 31% and increased conversion by 19%.

## Technical Implementation
```python
# Clickstream processing pipeline
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext

spark = SparkSession.builder.appName("UserBehavior").getOrCreate()
stream = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "kafka:9092").load()

# Sessionization
sessions = stream.groupBy(session_window("timestamp", "15 minutes"), "user_id")
                 .agg(collect_list("event").alias("events"))
                 
# Bottleneck detection
bottlenecks = sessions.filter(array_contains(col("events"), "error_click"))
                      .groupBy("page_section").count()
```

```sql
-- Engagement metrics calculation
SELECT
    user_id,
    COUNT(DISTINCT session_id) AS sessions,
    SUM(CASE WHEN event = 'purchase' THEN 1 ELSE 0 END) AS conversions,
    AVG(session_duration) AS avg_session_time
FROM user_events
GROUP BY user_id;

```
# Analytics Dashboard
User Engagement Dashboard
<ahref: Tableau dashboard showing live user behavior metrics><\a>

# Key Findings & Optimization
UX Issue	Impact Before	Solution	Result After
Checkout form complexity	42% dropoff	Simplified 5→2 steps	↓ 18% dropoff
Hidden CTAs	27% lower CTR	Visual prominence	↑ 34% CTR
Slow page loads	3.8s avg load	CDN optimization	↓ 1.2s load

# Behavioral Metrics
$ \text{Engagement Score} = \frac{\log(sessions) \times \sqrt{\text{conversions}}}{\text{bounce_rate}} $

# Technical Components
1. Data Architecture:
   <br>User Devices → Kafka → Spark Streaming → Redshift → Tableau
2. Heatmap Analysis:Click Heatmap
3. Session Recording:

```html
<script>
window.recordSession = true;
window.sessionSampling = 0.15; // 15% of users
</script>
```

## Impact Metrics:

* 31% reduction in bounce rate
* 19% increase in conversion rate
* 28% improvement in NPS scores
