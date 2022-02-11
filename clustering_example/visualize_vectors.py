import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import h5py


def visualize(vectors, useplotly=False):
    pca = PCA(n_components=3)
    projected_vectors = pca.fit_transform(vectors)
    print(projected_vectors.shape)
    if useplotly:
        import plotly.express as px
        fig = px.scatter_3d(
            x=projected_vectors[:, 0],
            y=projected_vectors[:, 1],
            z=projected_vectors[:, 2],
        )
        fig.update_traces(marker_size = 2)
        fig.show()
        return
    # default is to use matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.scatter(
        projected_vectors[:, 0],
        projected_vectors[:, 1],
        zs=projected_vectors[:, 2],
        s=5,
    )
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--vectors", required=True, help="HDF5 file containing the vectors")
    parser.add_argument("-p", "--useplotly", required=False, action="store_true",
                        help="Use plotly (in web browser) instead of matplotlib for the 3D scatterplot")
    args = parser.parse_args()
    path = args.vectors

    with h5py.File(path, "r") as f:
        vectors = f["vectors"][:]

    visualize(vectors, args.useplotly)
