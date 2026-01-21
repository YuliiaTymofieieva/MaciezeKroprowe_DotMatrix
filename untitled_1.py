import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.sparse import lil_matrix

# Funkcja do przetwarzania sekwencji
def process_sequence(filename, chunk_size=5000, scale_factor=500, eps=10, min_samples=5):
    matrix_data = []

    print("Rozpoczynanie przetwarzania pliku:", filename)

    for chunk in pd.read_csv(filename, header=None, sep='\s{2,}', names=range(3), chunksize=chunk_size, engine='python'):
        print("Przetwarzanie chunku:")
        print(chunk.head())  # Dodajemy wyświetlanie pierwszych kilku wierszy chunku
        # Znajdowanie nazw tabel
        table_names = ["> gi|12057211|gb|AE003849.1|", "> gi|12057211|gb|AE003849.1| Reverse"]
        groups = chunk[0].isin(table_names).cumsum()
        tables = {g.iloc[0, 0]: g.iloc[1:] for k, g in chunk.groupby(groups)}

        # Przetwarzanie tabeli "> gi|12057211|gb|AE003849.1|"
        if "> gi|12057211|gb|AE003849.1|" in tables:
            table = tables["> gi|12057211|gb|AE003849.1|"]
            x_values = table[0].astype(int)
            y_values = table[1].astype(int)
            num_points = table[2].astype(int)
            for x, y, num in zip(x_values, y_values, num_points):
                for i in range(num):
                    matrix_data.append([x + i, y + i])

        # Przetwarzanie tabeli "> gi|12057211|gb|AE003849.1| Reverse"
        if "> gi|12057211|gb|AE003849.1| Reverse" in tables:
            table_reverse = tables["> gi|12057211|gb|AE003849.1| Reverse"]
            x_values_reverse = table_reverse[0].astype(int)
            y_values_reverse = table_reverse[1].astype(int)
            num_points_reverse = table_reverse[2].astype(int)
            for x, y, num in zip(x_values_reverse, y_values_reverse, num_points_reverse):
                for i in range(num):
                    matrix_data.append([x + i, y + i])

    print("Przetwarzanie zakończone. Liczba punktów danych:", len(matrix_data))

    # Przekształcenie danych do macierzy numpy
    matrix_data = np.array(matrix_data)

    # Podział danych na kolumny
    x = matrix_data[:, 0] - 1
    y = matrix_data[:, 1] - 1

    # Zmniejszenie rozdzielczości danych (skalowanie)
    x_scaled = x // scale_factor
    y_scaled = y // scale_factor

    # Wartość maksymalna dla osi X i Y po skalowaniu
    max_x_scaled = int(np.max(x_scaled))
    max_y_scaled = int(np.max(y_scaled))

    # Utworzenie rzadkiej macierzy
    sparse_matrix = lil_matrix((max_x_scaled + 1, max_y_scaled + 1))

    # Wypełnienie rzadkiej macierzy wartościami
    for i in range(len(x_scaled)):
        sparse_matrix[x_scaled[i], y_scaled[i]] = 1

    # Konwersja rzadkiej macierzy na macierz gęstą
    dense_matrix = sparse_matrix.toarray()

    print("Macierz gęsta wygenerowana. Rozmiar macierzy:", dense_matrix.shape)

    # Klasteryzacja DBSCAN z nowymi parametrami
    try:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        
        # Klasteryzacja punktów
        labels = dbscan.fit_predict(np.column_stack((x_scaled, y_scaled)))
        
        print("Klasteryzacja zakończona. Liczba klastrów:", len(set(labels)) - (1 if -1 in labels else 0))
    except Exception as e:
        print("Wystąpił błąd podczas klasteryzacji:", e)
        return

    # Tworzenie wykresów macierzy i klasteryzacji
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Wykres macierzy
    axes[0].imshow(dense_matrix, cmap='viridis', origin='lower')
    axes[0].set_xticks(np.arange(0, max_y_scaled + 1, max(1, max_y_scaled // 10)))
    axes[0].set_yticks(np.arange(0, max_x_scaled + 1, max(1, max_x_scaled // 10)))
    axes[0].set_xticklabels(np.arange(0, max_y_scaled + 1, max(1, max_y_scaled // 10)))
    axes[0].set_yticklabels(np.arange(0, max_x_scaled + 1, max(1, max_x_scaled // 10)))
    axes[0].set_xlabel('Y')
    axes[0].set_ylabel('X')
    axes[0].set_title('Macierz - Cała sekwencja')
    axes[0].grid(color='black', linewidth=0.5)

    # Wykres klasteryzacji
    scatter = axes[1].scatter(y_scaled, x_scaled, c=labels, cmap='viridis', s=10)
    axes[1].set_xlabel('Y')
    axes[1].set_ylabel('X')
    axes[1].set_title('Klasteryzacja - Cała sekwencja')
    axes[1].grid(color='black', linewidth=0.5)
    fig.colorbar(scatter, ax=axes[1])

    # Zapisanie wykresów do plików
    plt.savefig("macierz_klasteryzacja.png")

    # Wyświetlenie wykresów
    plt.tight_layout()
    plt.show()

# Wykonanie analizy sekwencji dla dużego pliku
process_sequence("C:/Julia/Praca_Licencjacka/9a5c-temecula_0.mums")
