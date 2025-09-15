# Q2.5 Geometric Space Visualization Guide

## Overview
The Q2.5 geometric visualization (`Q25_geometric_space_finqa_test_1630.html`) shows the **same 3D space** as A4 but with **question assignments overlaid**.

## What You'll See in the Visualization

### 🔴 **RED Diamonds: Concept Centroids**
- **20 concept centroids** from A4 (revenue_recognition, income_taxes, net_income, etc.)
- Same positions as in A4 visualization
- Hover to see concept name and coordinates

### 🟢 **GREEN Circles: Document Regions**
- **Simulated chunk positions** around each centroid
- Represent the "area of influence" for each concept
- Lighter green for balls assigned to the question
- Darker green for unassigned balls

### 🟡 **YELLOW Diamond: Question Position**
- **Question embedding** in the same semantic space
- Shows where "What is the percentage change in the revenue from 2018 to 2019?" sits geometrically
- Large, bright marker for easy identification

### 🟠 **ORANGE Lines: Assignment Connections**
- **Direct connections** from question to assigned convex balls
- Line thickness = Assignment confidence (thicker = higher confidence)
- Shows the 3 balls the question was assigned to:
  - `incentive_compensation` (confidence: 0.534)
  - `net_income` (confidence: 0.517)
  - `income_taxes` (confidence: 0.517)

## Key Features

### **Interactive 3D Navigation**
- Rotate, zoom, pan around the space
- Click and drag to explore different angles
- Scroll to zoom in/out

### **Hover Information**
- **Centroids**: Concept name and coordinates
- **Chunks**: Ball assignment and status
- **Question**: Question text and position
- **Lines**: Assignment details and confidence

### **Visual Legend**
- Color coding clearly explained in title
- Assignment details in bottom-left annotation
- PCA variance explained for each axis

## Understanding the Spatial Relationships

1. **Proximity**: Closer objects are more semantically similar
2. **Assignments**: Orange lines show Q2.5's intelligent mapping
3. **Confidence**: Thicker lines indicate higher assignment confidence
4. **Dimensions**: 3D space captures 40.5% of total variance

## Comparison with A4
- **Same coordinate system** as A4 visualizations
- **Same centroids** and spatial relationships
- **Adds question layer** showing where questions map
- **Shows assignment logic** through connecting lines

This visualization answers your question: **"Why was this question assigned to these specific convex balls?"** by showing their spatial proximity and semantic relationships in the geometric space.