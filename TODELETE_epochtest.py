# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 10:54:56 2025

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  7 21:25:07 2018

@author: dedekinds
"""

import numpy as np
import networkx as nx
import os
import time
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

directed = True
p = 5.0  # For p == q in node2vec, it is equivalent to deepwalk
q = 1.0
num_walks = 1000
walk_length = 100
emb_size = 200
iteration = 5

LABEL = {
    'Case_Based': 1,
    'Genetic_Algorithms': 2,
    'Neural_Networks': 3,
    'Probabilistic_Methods': 4,
    'Reinforcement_Learning': 5,
    'Rule_Learning': 6,
    'Theory': 7
}

# Function to load features (IDs and labels)
def load_features(filename):
    ids, labels = [], []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            line_split = line.split()
            ids.append(line_split[0])
            labels.append(line_split[-1])
            line = f.readline()
        return ids, labels


# Function to load graph based on mutual references
def load_graph(filename, id_list):
    g = nx.DiGraph() if directed else nx.Graph()
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            line_split = line.split()
            if line_split[0] in id_list and line_split[1] in id_list and line_split[0] != line_split[1]:
                g.add_edge(line_split[0], line_split[1])
                g[line_split[0]][line_split[1]]['weight'] = 1
            line = f.readline()
    return g


def preprocess_transition_probs(g, directed=False, p=1, q=1):
    alias_nodes, alias_edges = {}, {}
    for node in g.nodes():
        probs = [g[node][nei]['weight'] for nei in sorted(g.neighbors(node))]
        norm_const = sum(probs)
        norm_probs = [float(prob) / norm_const for prob in probs]
        alias_nodes[node] = get_alias_nodes(norm_probs)

    if directed:
        for edge in g.edges():
            alias_edges[edge] = get_alias_edges(g, edge[0], edge[1], p, q)
    else:
        for edge in g.edges():
            alias_edges[edge] = get_alias_edges(g, edge[0], edge[1], p, q)
            alias_edges[(edge[1], edge[0])] = get_alias_edges(g, edge[1], edge[0], p, q)

    return alias_nodes, alias_edges


def get_alias_edges(g, src, dest, p=1, q=1):
    probs = []
    for nei in sorted(g.neighbors(dest)):
        if nei == src:
            probs.append(g[dest][nei]['weight'] / p)
        elif g.has_edge(nei, src):
            probs.append(g[dest][nei]['weight'])
        else:
            probs.append(g[dest][nei]['weight'] / q)
    norm_probs = [float(prob) / sum(probs) for prob in probs]
    return get_alias_nodes(norm_probs)


def get_alias_nodes(probs):
    l = len(probs)
    a, b = np.zeros(l), np.zeros(l, dtype=np.int64)
    small, large = [], []

    for i, prob in enumerate(probs):
        a[i] = l * prob
        if a[i] < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        sma, lar = small.pop(), large.pop()
        b[sma] = lar
        a[lar] += a[sma] - 1.0
        if a[lar] < 1.0:
            small.append(lar)
        else:
            large.append(lar)
    return b, a


def node2vec_walk(g, start, alias_nodes, alias_edges, walk_length=30):
    path = [start]
    while len(path) < walk_length:
        node = path[-1]
        neis = sorted(g.neighbors(node))
        if len(neis) > 0:
            if len(path) == 1:
                l = len(alias_nodes[node][0])
                idx = int(np.floor(np.random.rand() * l))
                if np.random.rand() < alias_nodes[node][1][idx]:
                    path.append(neis[idx])
                else:
                    path.append(neis[alias_nodes[node][0][idx]])
            else:
                prev = path[-2]
                l = len(alias_edges[(prev, node)][0])
                idx = int(np.floor(np.random.rand() * l))
                if np.random.rand() < alias_edges[(prev, node)][1][idx]:
                    path.append(neis[idx])
                else:
                    path.append(neis[alias_edges[(prev, node)][0][idx]])
        else:
            break
    return path


import time   

total_runtime = time.time()
edge_path = 'data/cora/cora.content'
label_path = 'data/cora/cora.cites'
model_path = './output_deepwalk.models'

# Load feature and adjacency matrix from file
id_list, labels = load_features(edge_path)
g = load_graph(label_path, id_list)

# Adding isolated nodes to the graph
for node in id_list:
    if not g.has_node(node):
        g.add_node(node)



if os.path.isfile(model_path):
    model = Word2Vec.load(model_path)
    print('Model loaded successfully.')
else:
    alias_nodes, alias_edges = preprocess_transition_probs(g, directed, p, q)

    walks = []
    for i in range(num_walks):
        r = np.array(range(len(id_list)))
        np.random.shuffle(r)
        for node in [id_list[j] for j in r]:
            walks.append(node2vec_walk(g, node, alias_nodes, alias_edges, walk_length))

    start_model_time = time.time()
    model = Word2Vec(vector_size=emb_size, min_count=0, sg=1)
    model.build_vocab(walks)
    print("Vocabulary built successfully.")

    for epoch in range(iteration):
        start_time = time.time()
        model.train(walks, total_examples=model.corpus_count, epochs=1)
        end_time = time.time()
        print(f"Epoch {epoch + 1}/{iteration} completed in {end_time - start_time:.2f} seconds.")

    model.save('output_deepwalk.model')
    print("Model saved successfully.")

# Preparing data for classification
y = np.array([LABEL[labels[temp]] for temp in range(2708)])

x_train, x_test = np.zeros(emb_size), np.zeros(emb_size)
droppoint = 500

for x in range(droppoint):
    x_train = np.row_stack((x_train, model.wv[id_list[x]]))
x_train = np.delete(x_train, [0], axis=0)
y_train = y[:droppoint]

for x in range(droppoint, 1500):
    x_test = np.row_stack((x_test, model.wv[id_list[x]]))
x_test = np.delete(x_test, [0], axis=0)
y_test = y[droppoint:1500]

# Training and evaluating the classifier
classifier = LogisticRegression()
classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)
print('Node2Vec accuracy:')
print(list(predictions - y_test).count(0) / len(y_test))
print(f"Prepare data runtime: {start_model_time - total_runtime:.2f} sec")
print(f"Model Runtime: {time.time() - start_model_time:.2f} sec")
print(f"Total Runtime: {time.time() - total_runtime:.2f} sec")

#####AUC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, classification_report

# Binarize the labels for multi-class AUC calculation
y_test_binarized = label_binarize(y_test, classes=list(LABEL.values()))

# Predict probabilities
y_pred_probabilities = classifier.predict_proba(x_test)

# Compute the AUC score for each class
auc_scores = roc_auc_score(y_test_binarized, y_pred_probabilities, multi_class="ovr", average=None)

# Compute macro-average AUC score
macro_auc = roc_auc_score(y_test_binarized, y_pred_probabilities, multi_class="ovr", average="macro")

# Print results
print("AUC scores for each class:")
for i, label_name in enumerate(LABEL.keys()):
    print(f"{label_name}: {auc_scores[i]:.4f}")

print(f"Macro-average AUC: {macro_auc:.4f}")

# Print classification report for further details
print("Classification Report:")
print(classification_report(y_test, predictions, target_names=LABEL.keys()))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 3. Visualize Training Results
def visualize_results(y_test, predictions):
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(LABEL.keys()))
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.show()
    
visualize_results(y_test, predictions)
