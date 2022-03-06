import karateclub
import pandas
import networkx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 

df = pandas.read_csv("/data/edges.csv")
df.head()

# Design graph
graph = networkx.from_pandas_edgelist(df, "n1", "n2", create_using=networkx.Graph())
print(len(graph))

# Training and generate embedding
model = DeepWalk(walk_length = 100, demensions = 64, window_size = 5)
model.fit(G)
embedding = model.get_embedding()
print(embedding.shape)

nodes = list(range(100))

def plot_graph(node_num):
    a = embedding[node_num]
    pca = PCA(n_components=2)
    pca_out = pca.fit_transform(a)

    plt.figure(figsize=(15,10))
    plt.scatter(pca_out[:, 0], pca_out[:, 1])

    for i, node in enumerate(node_num):
        plt.annotate(node, (pca_out[i, 0], pca_out[i, 1]))
    plt.xlabel("Weight")
    plt.ylabel("Height")
    plt.show()
plot_graph(nodes)