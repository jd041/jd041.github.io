---
title: "A Gentle Introduction to Graph Neural Networks"
layout: post
date: 2022-11-02 21:48
#image: /assets/images/markdown.jpg
headerImage: false
tag:
- CNN
- GNN
category: blog
author: jasonding
description: My study note of GNN
---

# A Gentle Introduction to Graph Neural Networks

My study note of https://distill.pub/2021/gnn-intro/

## Graph Representations and Problems

A graph contains vertex, edge, and global attributes, each storing different kinds of information. We can represent images, texts, molecules, social networks, and so on as graphs. There are three general types of prediction tasks on graphs: graph-level, node-level, and edge-level. To use machine learning techniques, one thing to note is that representing a graph’s connectivity is complicated. We can use adjacency matrix where each entry indicates whether an edge is formed or not, but there are two drawbacks: 1. The matrix is often sparse and hence space-inefficient; 2. there are many adjacency matrices that can encode the same connectivity, but they are not permutation invariant. To solve the first problem, we can instead use a adjacency list. Each element of the list is a tuple containing the two node indices which form an edge. In this way, the adjacency list is both permutation invariant and memory-efficient.



## Graph Neural Networks (GNNs)

 A GNN is an optimizable transformation on all attributes of the graph (nodes, edges, global-context) that preserves graph symmetries (permutation invariances). 

### The simplest GNN

For each single layer of a simple GNN, a graph is the input, and each component (V,E,U) gets updated by a MLP to produce a new graph. However, the connectivity doesn't change. Only the embeddings change. 

### Prediction

For binary classification on nodes, we can just apply a linear classifier at the final layer of the graph. If we don't have information in the node, then we can use its adjacent edges to predict. One way to do this is through pooling where gathered embeddings are aggregated to replace the missing information. Meanwhile, we can use node features to predict edge or global properties by similar pooling methods. 

### Message passing

To be continued.