# prototipo_recoleccion.py (Versión 2.0)
import cv2
import os

# --- 1. CONFIGURACIÓN ---
# Define la carpeta raíz de tus datos
CARPETA_RAIZ = "1_Datos_PC/train"

# Define las categorías. Estas DEBEN coincidir con los nombres de las carpetas
CATEGORIAS = ["01_placa_madre", "02_tarjeta_grafica", "03_modulo_ram", "04_cpu", "05_fuente_poder"]

# --- 2. FUNCIÓN PRINCIPAL ---
def main():

    # --- Selección de Categoría ---
    print("--- Herramienta de Recolección de Datos 'PC-Lens' ---")
    print("Por favor, selecciona la categoría de lo que vas a fotografiar:")
    
    for i, categoria in enumerate(CATEGORIAS):
        print(f"  {i+1}) {categoria}")
    
    try:
        opcion = int(input("Escribe el número de la opción: ")) - 1
        categoria_seleccionada = CATEGORIAS[opcion]
    except:
        print("Opción no válida. Saliendo.")
        return

    # Crear la carpeta de destino si no existe
    carpeta_destino = os.path.join(CARPETA_RAIZ, categoria_seleccionada)
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
        print(f"Carpeta '{carpeta_destino}' creada.")

    # Contar cuántos archivos ya hay para no sobrescribir
    contador = len(os.listdir(carpeta_destino)) + 1
    print(f"Carpeta seleccionada: {categoria_seleccionada} (empezando en la imagen {contador})")

    # --- Tu código de cámara (casi idéntico) ---
    capturar = cv2.VideoCapture(1) # 1 = cámara externa, 0 = interna
    if not capturar.isOpened():
        print("No se puede abrir la cámara")
        return

    print("\n¡Cámara lista! Presiona 's' para tomar la foto, 'q' para salir.")

    while True:
        ret, frame = capturar.read()
        if not ret:
            print("Error al capturar")
            break

        # Mostrar la imagen
        cv2.imshow("Vista de la cámara", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # Guardar la imagen en la carpeta correcta
            nombre_archivo = f"{carpeta_destino}/img_{contador:04d}.jpg" # 04d = 0001, 0002, etc.
            cv2.imwrite(nombre_archivo, frame)
            print(f"¡Foto guardada! -> {nombre_archivo}")
            contador += 1

        elif key == ord('q'):
            break

    # Liberar todo
    capturar.release()
    cv2.destroyAllWindows()
    print("Saliendo. ¡Buen trabajo!")

if __name__ == "__main__":
    main()