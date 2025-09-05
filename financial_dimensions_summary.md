# Financial Dimensions & Elliptical Disks Explanation

## What are Financial Dimensions 1, 2, 3?

The 3D axes in your visualization represent **semantic dimensions** derived from TF-IDF keyword analysis:

### **Financial Dimension 1: Contract ↔ Revenue Axis (54.2% variance)**
- **Positive direction**: Revenue-focused concepts
  - Keywords: `revenue`, `recognition`, `revenue recognition`, `unearned`
- **Negative direction**: Contract-focused concepts  
  - Keywords: `balance`, `contract`, `balances`, `contract balances`
- **Interpretation**: Separates revenue recognition concepts from contractual obligations

### **Financial Dimension 2: Balance Sheet ↔ Operations Axis (45.8% variance)**
- **Positive direction**: Contractual/operational concepts
  - Keywords: `contract`, `balances`, `contract balances`
- **Negative direction**: Balance sheet/accounting concepts
  - Keywords: `consolidated`, `receivable`, `receivable balance`
- **Interpretation**: Distinguishes operational activities from balance sheet items

### **Financial Dimension 3: Recognition Timing Axis (0.0% variance)**
- **Minimal variation** in finqa_test_96 concepts
- **Keywords**: `balance`, timing-related terms
- **Interpretation**: Would capture revenue recognition timing if more varied

## Concept Positions in 3D Space

| Concept | Dimension 1 | Dimension 2 | Dimension 3 | Interpretation |
|---------|-------------|-------------|-------------|----------------|
| **Contract Balances** | -0.416 | +0.663 | 0.000 | Contract-focused, balance-oriented |
| **Revenue Unearned** | +0.833 | 0.000 | 0.000 | Revenue-focused, neutral balance |
| **Receivable Balance** | -0.416 | -0.663 | 0.000 | Contract-focused, operational |

## What are the Elliptical Disks?

The **elliptical disks** around each convex ball represent **semantic uncertainty boundaries**:

### **Mathematical Basis**
- **Source**: Covariance matrix of keywords in semantic space
- **Shape**: Principal axes show directions of maximum variance
- **Size**: Proportional to concept's keyword diversity and uncertainty

### **Visual Interpretation**
```
    Elliptical Disk Anatomy:
    
    ┌─────────────────────────────┐
    │     Uncertainty Region      │
    │  ╭─────────────────────╮    │
    │ ╱                       ╲   │
    │╱         ●               ╲  │  ← Major Axis
    │╲      Centroid           ╱  │    (Max variation)
    │ ╲                       ╱   │
    │  ╰─────────────────────╯    │
    │              ↑              │
    └──────────────┼──────────────┘
                Minor Axis
            (Min variation)
```

### **Financial Meaning**
1. **Sharp, small ellipse** = Precise, well-defined financial concept
2. **Large, stretched ellipse** = Broad financial category with semantic spread  
3. **Ellipse orientation** = Direction of maximum keyword variation
4. **Ellipse overlap** = Concepts sharing semantic space

### **Why Elliptical (not circular)?**
- **Semantic spaces are anisotropic** - concepts stretch more in some directions
- **Reflects natural clustering** - financial terms group along specific axes
- **Shows directional uncertainty** - some keyword variations are more likely than others

## Example for finqa_test_96:

- **Contract Balances**: Large ellipse (8 keywords, high diversity)
- **Revenue Unearned**: Medium ellipse (3 keywords, moderate spread)  
- **Receivable Balance**: Small ellipse (2 keywords, precise concept)

The elliptical disks help you understand:
- How "fuzzy" or precise each concept is
- Which directions have the most semantic variation
- Where concept boundaries might overlap with others

This creates a mathematically grounded, visually intuitive representation of your conceptual space!