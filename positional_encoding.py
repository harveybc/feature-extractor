import numpy as np
import pandas as pd
import argparse
import os

def generate_positional_encoding(num_positions, num_features):
    """
    Genera una matriz de codificación posicional de tamaño (num_positions, num_features).
    
    Args:
        num_positions (int): Número de posiciones en la serie de tiempo.
        num_features (int): Número de dimensiones de la codificación posicional, debe ser par.

    Returns:
        np.ndarray: Matriz de codificación posicional de tamaño (num_positions, num_features).
    """
    encoding = np.zeros((num_positions, num_features))
    
    for pos in range(num_positions):
        for i in range(0, num_features, 2):
            div_term = 10000 ** (2 * (i // 2) / num_features)
            encoding[pos, i] = np.sin(pos / div_term)
            encoding[pos, i + 1] = np.cos(pos / div_term)
    
    return encoding

def main():
    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Programa para añadir codificación posicional a un archivo CSV.")
    parser.add_argument("input_file", type=str, help="Nombre del archivo CSV de entrada.")
    parser.add_argument("--output", type=str, default="pos_encoded.csv", 
                        help="Nombre del archivo CSV de salida (por defecto: pos_encoded.csv).")
    
    args = parser.parse_args()
    
    # Cargar el archivo CSV de entrada sin encabezados
    if not os.path.exists(args.input_file):
        print(f"Error: El archivo '{args.input_file}' no existe.")
        return
    
    # Cargar datos sin encabezado
    df = pd.read_csv(args.input_file, header=None)
    num_positions = len(df)
    num_features = df.shape[1]

    # Generar codificación posicional
    encoding = generate_positional_encoding(num_positions, num_features)
    
    # Añadir la codificación posicional como columnas adicionales
    for i in range(num_features):
        df[f'pos_enc_{i}'] = encoding[:, i]
    
    # Guardar el nuevo DataFrame en el archivo de salida
    output_file = args.output
    df.to_csv(output_file, index=False, header=False)
    print(f"Archivo con codificación posicional guardado como '{output_file}'.")

if __name__ == "__main__":
    main()
