# prototipo_placa.py

import cv2;
import os;

def main():
    
    # Abrir la cámara (0 = cámara por defecto)
    #0 → primera cámara que encuentra tu PC (por defecto la integrada).
    #1 → segunda cámara que encuentre (por ejemplo la externa USB).
    
    capturar = cv2.VideoCapture(1)

    if not capturar.isOpened():
        print("No se puede abrir la cámara")
        return

    print("Presiona 's' para tomar una foto, 'q' para salir")


    carpeta = "img"
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

    contador = 1

    while True:
        # Capturar frame
        ret, frame = capturar.read()
        if not ret:
            print("Error al capturar la imagen")
            break

        # Mostrar la imagen en ventana
        cv2.imshow("Vista de la cámara", frame)

        # Esperar tecla
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # Guardar imagen cuando presionas 's'
            nombre_archivo = f"{carpeta}/placa_{contador}.jpg"
            cv2.imwrite(nombre_archivo, frame)
            print("Imagen guardada como {nombre_archivo}")
            contador +=1

        elif key == ord('q'):
            # Salir
            break

    # Liberar cámara y cerrar ventanas
    capturar.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
