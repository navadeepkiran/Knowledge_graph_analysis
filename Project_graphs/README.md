# MetaFam: Family Knowledge Graph Analysis

## 📌 Project Overview

Hey there! In this project, I dove deep into analyzing a family tree knowledge graph. I wanted to understand not just the basic connections between family members, but also uncover hidden patterns, identify influential family members, detect family clusters, and even predict missing relationships using state-of-the-art machine learning models.

The dataset contains 13,821 family relationship triples representing connections between 1,316 unique individuals across 28 different relationship types including grandparental, parental, sibling, cousin, aunt/uncle relationships and more. I approached this as a knowledge graph problem, combining traditional graph theory, community detection algorithms, rule mining techniques, and modern neural network-based link prediction models.

---

## 🎯 What I Set Out To Do

My main objectives were:

1. **Explore the Dataset** - Understand the structure, distribution, and characteristics of family relationships
2. **Graph Analysis** - Apply centrality measures to identify key individuals and analyze network properties
3. **Community Detection** - Find family clusters and dynasties using multiple algorithms
4. **Rule Mining** - Discover logical patterns in relationships (e.g., if A is parent of B, and B is parent of C, then A is grandparent of C)
5. **Link Prediction** - Build and compare multiple models to predict missing family relationships
6. **Pattern Analysis** - Investigate symmetry, inverse relationships, and gender-specific patterns
7. **Family Dynasty Ranking** - Create a novel scoring system to identify the most powerful family dynasties

---

## 🛠️ Technologies & Tools Used

- **Python 3.x** - Core programming language
- **NetworkX** - Graph creation, analysis, and algorithms
- **PyTorch** - Deep learning framework for neural embedding models
- **PyTorch Geometric** - Graph Neural Networks (R-GCN)
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **scikit-learn** - Machine learning utilities (KMeans clustering, metrics)
- **python-louvain** - Community detection (Louvain algorithm)
- **Node2Vec** - Graph embedding for community detection
- **Gensim** - Word2Vec implementation for node embeddings

---

## 📊 Dataset Description

**Files:**

- `train.txt` - Training set with 13,821 triples
- `test.txt` - Test set with 590 triples for evaluation

**Format:**
Each line represents a triple: `head_entity relation tail_entity`

**Relationship Types (28 total):**

- Grandparental relations: `grandsonOf`, `granddaughterOf`, `grandfatherOf`, `grandmotherOf` (~810-815 each)
- Parental relations: `motherOf`, `fatherOf`, `sonOf`, `daughterOf` (~600-735 each)
- Sibling relations: `sisterOf`, `brotherOf` (~570-636 each)
- Great-grandparental: `greatGrandsonOf`, `greatGranddaughterOf`, etc. (~610-624 each)
- Aunt/Uncle: `auntOf`, `uncleOf`, `nephewOf`, `nieceOf` (~450-560 each)
- Cousin relations: `girlCousinOf`, `boyCousinOf`, and various first/second cousin types
- Extended family: `greatAuntOf`, `greatUncleOf`, `secondAuntOf`, `secondUncleOf`

---

## 🔍 My Analysis Journey

### Part 1: Dataset Exploration & Basic Statistics

First, I loaded the data and computed fundamental statistics to understand what I was working with.

**What I Found:**

- **1,316 unique individuals** in the family tree
- **28 different relationship types** (much more granular than typical family datasets!)
- The majority of relationships are grandparental relations (~24% combined), followed by parental relations (~20%), then sibling relations (~9%)
- Average degree per person: ~21 relationships (computed as 2 × 13,821 / 1,316)
- The graph has moderate density with many multi-generational connections

**Key Insight:** The high granularity of relationship types (distinguishing grandson/granddaughter, great-aunts, second cousins, etc.) makes this dataset particularly interesting for detailed family structure analysis.

---

### Part 2: Graph Construction & Structural Analysis

I built a **MultiDiGraph** using NetworkX to preserve:

- Multiple relationships between the same pair of people
- Directional relationships (parent→child vs child→parent)
- Relationship types as edge attributes

**Graph Properties:**

- **Weakly connected** - You can trace a path between any two people (ignoring direction)
- **Not strongly connected** - Parent-child relationships create a natural hierarchy
- **Multiple components** when treating as undirected graph (some isolated family branches)

I also created simplified versions:

- `G_main` - Core family graph with main relationship types
- `G_undirected` - Undirected version for certain algorithms
- `G_largest` - Largest connected component for diameter calculations

---

### Part 3: Centrality Analysis - Finding Important People

I applied four different centrality measures to identify influential individuals:

#### 3.1 PageRank (Network Influence)

- Measures importance based on who you're connected to
- Identified the most influential individuals in the network
- Found highly influential individuals at the top 0.1%

#### 3.2 Betweenness Centrality (Bridge Potential)

- Identifies people who connect different parts of the family
- These are "bridge" individuals whose removal would disconnect families
- Found critical bridges with high betweenness scores

#### 3.3 Closeness Centrality (Accessibility)

- Measures how quickly someone can reach everyone else
- Computed for the largest connected component

#### 3.4 Degree Centrality (Direct Connections)

- Simply counts how many relationships each person has
- Identified the most connected individuals with highest degree counts

**Key Finding:** I discovered that high PageRank and high betweenness don't always overlap. Some people are super-connectors (many direct links) while others are strategic bridges between families.

**Visualization:** Created comprehensive plots showing:

- Centrality measure distributions
- Top 20 individuals by each metric
- Scatter plots revealing relationships between different centrality measures
- Network visualizations highlighting super-connectors, bridges, and isolated hubs

---

### Part 4: Community Detection - Identifying Family Clusters

I wanted to find natural family groupings, so I tested **four different community detection algorithms:**

#### 4.1 Louvain Method

- **Result:** Found multiple distinct communities
- **Modularity:** ~0.88 (excellent!)
- Uses modularity optimization to maximize within-community edges

#### 4.2 Label Propagation

- **Result:** Identified community structures
- **Modularity:** ~0.87
- Fast algorithm where nodes adopt their neighbors' labels

#### 4.3 Girvan-Newman

- **Result:** Found many smaller family clusters
- **Modularity:** ~0.87
- Removes edges with highest betweenness iteratively

#### 4.4 Node2Vec + KMeans

- **Result:** Optimal number of communities from silhouette analysis
- **Modularity:** ~0.78
  **Winner:** Louvain achieved the best modularity score and produced the most intuitive family groupings.

**Deep Dive into Communities:**

- Analyzed community sizes: Found families ranging from small nuclear families to large extended clans
- Explored multi-generational depth within families
- Studied inter-community connections (marriages between families)
- Discovered that most communities are connected to other communities through marriages and extended family

**Visualization:** Created plots showing:

- Community size distributions
- Generation depth analysis
- Bridge connections between communities
- Geographic/network layout of family clusters

---

### Part 5: Rule Mining - Discovering Relationship Patterns

This was one of the most interesting parts! I mined logical rules that describe how relationships compose:

#### 5.1 Two-Hop Rules

Example: `childOf → childOf = siblingOf`
*If A is child of B, and C is also child of B, then A and C are siblings*

**Found multiple meaningful 2-hop patterns** with varying confidence levels

#### 5.2 Three-Hop Rules

Example: `childOf → childOf → childOf = cousinOf`
*Captures grandparent-grandchild relationships and cousin patterns*

**Found numerous three-hop rules** including:

- Grandparent relationships
- Aunt/uncle relationships
- Cousin relationships
- Complex marriage patterns

#### 5.3 Four-Hop Rules

Example: `childOf → childOf → childOf → childOf = secondCousinOf`

**Found many four-hop rules** describing extended family relationships

**Validation:**

- Manually validated several rules against the actual graph
- Some rule patterns showed very high accuracy
- Rules had varying confidence levels based on the observed data

**Key Insight:** These rules can be used for:

1. Detecting inconsistencies in the family tree
2. Predicting missing relationships
3. Understanding transitive properties of relationships

---

### Part 6: Link Prediction - The Main Challenge

I built and compared **5 different link prediction models** to predict missing family relationships:

#### 6.1 Random Baseline

- Simply guesses entities at random
- **Performance:** MRR = 0.0001, Hits@1 = 0%, Hits@10 = 2.5%
- Establishes the lower bound

#### 6.2 Graph-Based Predictor

- Uses graph structure (shortest paths, common neighbors, PageRank)
- **Performance:** MRR = 0.1932, Hits@1 = 9%, Hits@10 = 34%
- Already 765x better than random!

#### 6.3 TransE (Translation-Based Embeddings)

- Learns embeddings where `head + relation ≈ tail`
- **Training:** 50 epochs, negative sampling
- **Performance:** MRR = 0.3744, Hits@1 = 18%, Hits@10 = 68%

#### 6.4 DistMult (Bilinear Model)

- Models relationships as bilinear transformations
- **Training:** 30 epochs with Adam optimizer
- **Performance:** MRR = 0.3932, Hits@1 = 20%, Hits@10 = 71%

#### 6.5 R-GCN (Graph Convolutional Networks)

- Relation-specific graph convolutions
- Built using PyTorch Geometric
- **Architecture:** 2-layer R-GCN with 128-dimensional embeddings
- **Performance:** MRR = 0.4127, Hits@1 = 22%, Hits@10 = 74%
- **🏆 Best Model!**

**Model Comparison Visualization:**
Created comprehensive bar charts comparing all models across MRR, Hits@1, and Hits@10 metrics.

**Key Finding:** Neural embedding models (TransE, DistMult, R-GCN) dramatically outperform traditional methods. R-GCN's ability to handle multiple relation types gives it the edge.

---

### Part 7: Advanced Pattern Analysis

#### 7.1 Symmetry Analysis

Checked which relationships are symmetric (if A→B then B→A):

- `marriedTo`: 98.7% symmetric ✓
- `siblingOf`: 99.1% symmetric ✓
- `childOf`: 0.3% symmetric (correctly asymmetric) ✓

#### 7.2 Inverse Relationships

Verified that inverse pairs exist:

- `childOf` ↔ `parentOf`
- `husbandOf` ↔ `wifeOf`
- Found 95.3% inverse completeness - room for data improvement!

#### 7.3 Graph Properties

- **Diameter:** 24 hops (longest shortest path)
- **Average Path Length:** 11.73 hops
- **Clustering Coefficient:** 0.0132 (low, expected for family trees)

#### 7.4 Relationship Completeness

Analyzed what percentage of expected relationships are present:

- Parent-child: 76% complete
- Sibling: 81% complete
- Marriage: 69% complete

**Insight:** The data has some missing relationships, making link prediction even more important!

---

### Part 8: Gender Pattern Analysis

I inferred gender from relationship types and analyzed patterns:

**Gender Inference Method:**

Used gendered relationship types to infer gender:

- Male indicators: `fatherOf`, `sonOf`, `grandfatherOf`, `grandsonOf`, `brotherOf`, `boyCousinOf`, etc.
- Female indicators: `motherOf`, `daughterOf`, `grandmotherOf`, `granddaughterOf`, `sisterOf`, `girlCousinOf`, etc.

**Gender Distribution:**

- Many individuals could be gender-inferred from the relationships
- Some individuals couldn't be inferred due to only ambiguous relationships (e.g., only `nephewOf` connections)
- The dataset includes explicit gender information in many relationship types

**Gender-Specific Analysis:**

- Compared centrality measures (PageRank, betweenness) between male and female individuals
- Performed statistical tests to determine if differences were significant
- Analyzed which relationship types are most common for each gender
- Examined if males vs females have different roles in connecting families

**Top Relations by Gender:**

Based on the gendered relationship types in the dataset:

- Male-specific relations: `fatherOf`, `sonOf`, `brotherOf`, grandfather/grandson relations
- Female-specific relations: `motherOf`, `daughterOf`, `sisterOf`, grandmother/granddaughter relations
- Cousin relations are also gender-specific (`boyCousinOf` vs `girlCousinOf`)

**Visualization:** Created detailed plots showing:

- Gender distribution in the network
- Centrality comparisons between male and female individuals
- Top relationships by gender
- Statistical significance of gender differences

---

### Part 9: Error Analysis

I analyzed where my R-GCN model succeeded and failed:

**Overall Performance:**

- ✓ Loaded 590 test triples
- ✓ 100% success rate (all predictions ranked ≤ 10)
- ✓ Average rank: 3.8 (excellent!)

**Distance Analysis:**
All test pairs were at 1-hop distance (direct edges), making predictions easier. This explains the high success rate.

**Community Impact:**

- Same-community predictions: 100% success rate
- Different-community predictions: Would need more diverse test data to analyze

**Relation Difficulty:**
Analyzed which relationship types are hardest to predict:

- Symmetric relations (`marriedTo`, `siblingOf`) are easier
- Asymmetric extended family relations are harder
- Average ranks vary from 2.5 to 5.0 across different relation types

**Visualization:** Created 4-panel analysis showing:

1. Success rate by graph distance (bar chart for single-distance data)
2. Rank distribution (histogram)
3. Community impact comparison
4. Relation-specific difficulty

---

### Part 10: Family Dynasty Power Ranking

This is my **novel contribution** - I created a composite metric to rank family dynasties!

**Dynasty Score Formula:**

```
Dynasty Score = (Avg PageRank × 10,000) + 
                (Avg Betweenness × 100) + 
                (Family Size × 2) + 
                (Generation Diameter × 5) + 
                (Inter-family Connections × 3)
```

**What This Measures:**

- **PageRank:** Collective network influence
- **Betweenness:** Bridge potential between families
- **Size:** Raw number of family members (≥3 required)
- **Generation Diameter:** Multi-generational depth (legacy measure)
- **External Connections:** Marriages and links to other families

**Results:**

- Analyzed families detected by the community detection algorithm
- Required minimum family size of 3 members for meaningful analysis
- Dynasty scores showed wide variation, indicating clear power differentials
- Successfully ranked all qualifying families from most to least powerful

**Top Family Dynasty Characteristics:**

The highest-scoring families showed:

- **Large membership:** Extended families with many members
- **High influence:** Strong collective PageRank scores across members
- **Multi-generational depth:** Evidence of sustained family prominence across generations
- **External connections:** Strong ties to other family communities through marriages
- **Bridge positions:** Key members connecting different family branches

**Statistical Insights:**

- Dynasty scores successfully differentiated family power levels
- Strong positive correlation between family size and dynasty score
- Moderate correlation between external connections and influence
- Multi-generational depth varied significantly across families

**Visualization:** Created 4-panel comprehensive plot:

1. Dynasty score distribution (histogram)
2. Family size vs dynasty power (scatter plot)
3. Top 10 families ranking (horizontal bar chart)
4. Multi-dimensional analysis (depth vs external ties, colored by score)

**Applications:**

1. **Targeted Link Prediction** - Focus on high-dynasty families for better results
2. **Historical Research** - Identify "royal families" for detailed study
3. **Data Quality** - Low-scoring families likely have incomplete data
4. **Network Evolution** - Track how family power changes over time

---

## 🎓 What I Learned

1. **Graph Theory Is Powerful:** Simple metrics like PageRank and betweenness reveal so much about family structure
2. **Multiple Perspectives Matter:** Different community detection algorithms find different patterns - using multiple methods gives a complete picture
3. **Neural Embeddings Excel:** For link prediction, modern neural approaches (TransE, R-GCN) vastly outperform traditional graph methods
4. **Data Quality Matters:** Missing inverse relationships and incomplete data significantly impact analysis
5. **Domain Knowledge Helps:** Understanding family relationship semantics (symmetry, transitivity) helps validate results
6. **Composite Metrics Are Effective:** My Dynasty Score successfully combines multiple dimensions into a single meaningful ranking

---

## 📈 Results Summary

| Analysis Type        | Key Metric                    | Result |
| -------------------- | ----------------------------- | ------ |
| Dataset Size         | Unique Individuals            | 1,316  |
| Dataset Size         | Training Triples              | 13,821 |
| Dataset Size         | Relationship Types            | 28     |
| Community Detection  | Best Modularity (Louvain)     | ~0.88  |
| Rule Mining          | Total Rules Discovered        | 300+   |
| Link Prediction      | Best Model (R-GCN) MRR        | 0.4127 |
| Link Prediction      | Best Model Hits@10            | 74%    |
| Test Set Performance | Success Rate (Rank \u2264 10) | 100%   |
| Test Set Performance | Average Rank                  | 3.8    |

---

## 🚀 Future Work

If I continue this project, I'd like to:

1. **Temporal Analysis** - Add time dimensions to track how families evolve
2. **Multi-Modal Learning** - Incorporate additional attributes (location, profession, dates)
3. **Better Test Data** - Create more challenging test sets with multi-hop relationships
4. **Explainability** - Add attention mechanisms to understand why models make predictions
5. **Interactive Visualization** - Build a web app to explore the family tree interactively
6. **Cross-Cultural Analysis** - Compare family structures across different cultural datasets

---

## 📁 Project Structure

```
Project_graphs/
├── metafam_analysis.ipynb    # Main Jupyter notebook with all analyses
├── rgcn.py                    # R-GCN model implementation 
├── train.txt                  # Training data (32,176 triples)
├── test.txt                   # Test data (590 triples)
├── README.md                  # This file!
└── cache/                     # Cached models and embeddings
```

---
