# --- --- --- --- --- --- --- --- --- --- --- ---
# ARCHIVO: api_backend.py
# (El reemplazo de app_web.py - ¡Nuestro "Spring Boot"!)
# --- --- --- --- --- --- --- --- --- --- --- ---
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io # Para leer la imagen subida

# --- --- --- --- --- --- --- --- --- --- --- ---
# MÓDULO 1: LÓGICA DE IA (¡Copiado 95% de probar_cerebro.py!)
# --- --- --- --- --- --- --- --- --- --- --- ---

# Tus clases. ¡EL ORDEN IMPORTA!
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

# Definir las transformaciones de imagen
transform = transforms.Compose(
    [transforms.Resize((64, 64)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

print("Cargando el cerebro 'cerebro_PC_Lens_v1.pth'...")
PATH = './cerebro_PC_Lens_v1.pth'
try:
    net = Net(num_classes_output=num_classes)
    net.load_state_dict(torch.load(PATH))
    net.eval()
    print("¡Cerebro cargado y listo para recibir peticiones!")
except Exception as e:
    print(f"Error fatal al cargar el modelo: {e}"); exit()

# --- --- --- --- --- --- --- --- --- --- --- ---
# MÓGLO 2: La API (¡El "Spring Boot"!)
# --- --- --- --- --- --- --- --- --- --- --- ---
app = FastAPI() # Crear la instancia de la app

# Configurar CORS (¡Importante!)
# Esto le da permiso a tu futuro frontend (ej. localhost:3000)
# para que pueda "hablar" con tu backend (localhost:8000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite todas las conexiones
    allow_credentials=True,
    allow_methods=["*"], # Permite todos los métodos (POST, GET, etc.)
    allow_headers=["*"],
)

# Este es tu "@PostMapping("/api/analizar")"
@app.post("/api/analizar") 
async def analizar_imagen(archivo: UploadFile = File(...)):
    
    # 1. Leer la imagen que el usuario subió
    contenido_imagen = await archivo.read()
    imagen_pil = Image.open(io.BytesIO(contenido_imagen)).convert("RGB")
    
    # 2. Procesar la imagen (¡la misma lógica de siempre!)
    img_t = transform(imagen_pil)
    batch_t = torch.unsqueeze(img_t, 0)

    # 3. Hacer la Predicción (¡el "motor"!)
    with torch.no_grad():
        output = net(batch_t)
    
    probabilidades = F.softmax(output, dim=1)[0]
    _, predicted_index = torch.max(output.data, 1)
    
    prediccion = CLASSES[predicted_index.item()]
    confianza = probabilidades[predicted_index.item()].item() * 100
    
    # 4. Devolver un JSON al frontend (¡No HTML!)
    return {
        "prediccion": prediccion,
        "confianza": round(confianza, 2)
    }

# Esto es para que puedas ejecutarlo con "python api_backend.py"
if __name__ == "__main__":
    print("Iniciando servidor de API en http://localhost:8000")
    uvicorn.run("api_backend:app", host="127.0.0.1", port=8000, reload=True)