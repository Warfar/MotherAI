# --- --- --- --- --- --- --- --- --- --- --- ---
# ARCHIVO: probar_cerebro.py
# (Ejecútalo en tu PC, no en Colab)
# --- --- --- --- --- --- --- --- --- --- --- ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image # Pillow, ya lo instalaste con rembg

# --- 1. Definir la MISMA arquitectura de Red que en Colab ---
# (Tiene que ser idéntica para poder cargar los pesos)

# Tus clases. ¡EL ORDEN IMPORTA!
# Debe ser el mismo orden que Colab te mostró.
CLASSES = ['01_placa_madre', '99_fondo']
num_classes = len(CLASSES)

class Net(nn.Module):
    def __init__(self, num_classes_output):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes_output) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # Aplanar
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- 2. Cargar el Cerebro Entrenado ---
print("Cargando el cerebro 'cerebro_PC_Lens_v1.pth'...")
PATH = './cerebro_PC_Lens_v1.pth'

# Inicializar el modelo
net = Net(num_classes_output=num_classes)
# Cargar los pesos que entrenaste
net.load_state_dict(torch.load(PATH))
# Poner el modelo en modo "evaluación" (importante)
net.eval()
print("¡Cerebro cargado y listo para predecir!")

# --- 3. Cargar y Preparar la Imagen de Prueba ---
# (Debe ser el mismo 'transform' que usaste en el entrenamiento)
transform = transforms.Compose(
    [transforms.Resize((64, 64)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

try:
    img = Image.open("prueba.jpg").convert('RGB') # Abrir la imagen
    # Preparar la imagen para la IA (transformarla y añadir un "batch dimension")
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0) # La IA espera un "lote" de imágenes

    # --- 4. ¡Hacer la Predicción! ---
    with torch.no_grad(): # No necesitamos calcular gradientes
        output = net(batch_t)
    
    # 'output' contiene los puntajes. 
    # Usamos softmax para convertirlos a porcentajes (probabilidades)
    probabilidades = F.softmax(output, dim=1)[0]
    
    # Encontrar la clase con el porcentaje más alto
    _, predicted_index = torch.max(output.data, 1)
    
    # Obtener el nombre de la clase
    prediccion = CLASSES[predicted_index.item()]
    confianza = probabilidades[predicted_index.item()].item() * 100
    
    print("\n--- ¡PREDICCIÓN LISTA! ---")
    print(f"La imagen 'prueba.jpg' es...")
    print(f"===> {prediccion} (con {confianza:.2f}% de confianza)")

    print("\nDetalle de confianza:")
    for i, clase in enumerate(CLASSES):
        print(f" - {clase}: {probabilidades[i].item()*100:.2f}%")

except FileNotFoundError:
    print("--- ERROR ---")
    print("No se encontró el archivo 'prueba.jpg'.")
    print("Asegúrate de tomar una foto de prueba y guardarla en esta carpeta.")
except Exception as e:
    print(f"Ocurrió un error: {e}")