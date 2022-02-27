import networkx as nx
from gensim.models import Word2Vec
from joblib import Parallel,delayed
import random
import itertools
from classify import read_node_label, Classifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def DeepWalk(nums,walk_length,G,workers = 1):
    def process_deepwalk(start,walk_length,G):
        walk = [start]
        while len(walk)<walk_length:
            cur = walk[-1]
            neiborhoods =  list(G.neighbors(cur))
            if len(neiborhoods) == 0:
                break
            else:
                walk.append(random.choice(neiborhoods))
        return walk
    def process(nums,walk_length,G,nodes):
        walks = []
        for _ in range(nums):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(process_deepwalk(v,walk_length,G))
        return walks
    def get_num(nums,wokers):
        if nums%wokers == 0:
            return [nums//wokers]*wokers
        else:
            return [nums//wokers]*(wokers-1) + [nums%wokers]
    nodes = list(G.nodes())
    results = Parallel(n_jobs=workers,)(delayed(process)(num,walk_length,G,nodes) for num in get_num(nums,workers))
    walks = list(itertools.chain(*results))
    return walks 

def get_model(**kwargs):
    kwargs["min_count"] = kwargs.get("min_count", 0)
    kwargs["sg"] = 1  # skip gram
    kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
    kwargs["workers"] = 3
    kwargs["window"] = 5
    kwargs["epochs"] = 4
    model = Word2Vec(**kwargs)
    return model

def get_embedding(G,model):
    embedding = {}
    for v in list(G.nodes()):
        embedding[v] = model.wv[v]
    return embedding 

def evaluate_embeddings(embeddings):
    X, Y = read_node_label('data/wiki/wiki_labels.txt')
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)

def plot_embeddings(embeddings,):
    X, Y = read_node_label('data/wiki/wiki_labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #nx.DiGraph()意思是创建有向图
    G = nx.read_edgelist('data/wiki/Wiki_edgelist.txt',
    create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    sentences = DeepWalk(20,10,G)
    embed_size = 128
    model = get_model(sentences = sentences,vector_size = embed_size)
    embedding = get_embedding(G,model)
    evaluate_embeddings(embedding)
    plot_embeddings(embedding)
    