import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.sparse import lil_matrix


def process_sequence(filename, chunk_size=5000, scale_factor=500, eps=10, min_samples=5):
    matrix_data = []

    print("Processing file:", filename)

    for chunk in pd.read_csv(
        filename,
        header=None,
        sep=r"\s+",
        names=range(3),
        chunksize=chunk_size,
        engine="python"
    ):

        # --- znajdź wszystkie nagłówki zaczynające się od ">"
        table_names = chunk[0][chunk[0].astype(str).str.startswith(">")].unique().tolist()

        if not table_names:
            continue

        # --- grupowanie danych pod nagłówkami
        groups = chunk[0].astype(str).str.startswith(">").cumsum()
        tables = {g.iloc[0, 0]: g.iloc[1:] for _, g in chunk.groupby(groups)}

        # --- przetwarzanie KAŻDEJ tabeli (forward + reverse)
        for table_name, table in tables.items():

            if table.empty:
                continue

            x_values = table[0].astype(int)
            y_values = table[1].astype(int)
            num_points = table[2].astype(int)

            for x, y, num in zip(x_values, y_values, num_points):
                for i in range(num):
                    matrix_data.append([x + i, y + i])

    print("Total points:", len(matrix_data))

    matrix_data = np.array(matrix_data)

    if matrix_data.size == 0:
        print("No data found in file.")
        return

    # --- coordinates
    x = matrix_data[:, 0] - 1
    y = matrix_data[:, 1] - 1


    #  AUTOMATIC SCALE FACTOR 

    max_coord = max(x.max(), y.max())

    if max_coord < 1000:
        scale_factor = 1
    elif max_coord < 10000:
        scale_factor = 10
    else:
        scale_factor = 100

    print("Auto scale_factor:", scale_factor)

    # --- downsampling
    x_scaled = x // scale_factor
    y_scaled = y // scale_factor

    max_x = int(np.max(x_scaled))
    max_y = int(np.max(y_scaled))

    # --- sparse matrix
    sparse_matrix = lil_matrix((max_x + 1, max_y + 1))

    for i in range(len(x_scaled)):
        sparse_matrix[x_scaled[i], y_scaled[i]] = 1

    dense_matrix = sparse_matrix.toarray()

    print("Matrix size:", dense_matrix.shape)

    # --- DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(np.column_stack((x_scaled, y_scaled)))

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print("Detected clusters:", n_clusters)

    # --- plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].imshow(dense_matrix, origin="lower", cmap="viridis")
    axes[0].set_title("Dot matrix")
    axes[0].set_xlabel("Y")
    axes[0].set_ylabel("X")

    scatter = axes[1].scatter(
        y_scaled, x_scaled, c=labels, cmap="viridis", s=10
    )
    axes[1].set_title("DBSCAN clustering")
    axes[1].set_xlabel("Y")
    axes[1].set_ylabel("X")

    fig.colorbar(scatter, ax=axes[1])

    plt.tight_layout()
    plt.savefig("rand_mums_dbscan.png")
    plt.show()


# --- URUCHOMIENIE ANALIZY
process_sequence("rand.mums")
