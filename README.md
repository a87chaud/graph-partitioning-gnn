# Hypergraph Partitioning with Graph Neural Networks

## Motivation
Hypergraph partitioning is a fundamental problem in VLSI design automation. Traditional partitioners such as **hMETIS** are highly optimized but rely on heuristic algorithms. With the rise of Graph Neural Networks (GNNs), there is growing interest in exploring whether deep learning can learn meaningful embeddings and provide high-quality partitions with less manual tuning.

The motivation for this project is to:
- Investigate if **unsupervised GNNs** can perform competitive hypergraph partitioning and test the performance against hMETIS.

---

## How It Works
1. **Input:** Hypergraphs from the ISPD98 benchmark suite, stored in `.hgr` format.  
   - The first line specifies the number of hyperedges and vertices.  
   - Subsequent lines list vertices belonging to each hyperedge.

2. **Conversion to Graphs:**  
   Since GNNs operate on graphs, hypergraphs are converted into graphs using the **star expansion method**:
   - Each hyperedge becomes a new node.  
   - Edges are added between the hyperedge node and its incident vertices.
   - Note: The clique expansion method was also considered but abandoned due to performance issues.

3. **Model:**  
   - A **GraphSAGE-based GNN** with multiple layers.  
   - Input node features = degree of each node.  
   - Output = soft partition assignment for each vertex.  

4. **Loss Function:**  
   I used the **GAP (Generalized Aggregation-based Partitioning) loss** introduced in:  
   *Gap: Generalizable Approximate Partitioning for Hypergraphs using GNNs* (Zhou et al., 2019).  
   - **Normalized Cut Term:** Encourages minimizing hyperedge cuts.  
   - **Balance Term:** Enforces balanced partition sizes.  
   - No supervision required; purely unsupervised optimization.

5. **Training & Inference:**  
   - Training is performed on a subset of ISPD hypergraphs.  
   - Inference mode outputs `.part` files (one partition ID per vertex).  
   - These are directly comparable to hMETIS outputs.  

---

## Benchmarks
We evaluate on the **IBM circuits (ibm01–ibm18)** from the ISPD benchmark suite:  

- Training: ibm01 – ibm10  
- Testing: ibm11 – ibm18  

---

## ✅ Evaluation Against hMETIS
- hMETIS (`shmetis`) was run on the same set of hypergraphs to obtain the **golden baseline partitions**.  
- Our GNN-based partitioner was tested by computing:  
  - **Hyperedge Cut** (minimize)  
  - **Sum of External Degrees** (minimize)  

### Results
- The GNN model showed **only a minimal performance dip** compared to hMETIS:  
  - Hyperedge cuts were typically within ~10x of hMETIS.  
  - Balance constraints were respected.  
  - The GNN was able to generalize to unseen hypergraphs (ibm11–ibm18).  

