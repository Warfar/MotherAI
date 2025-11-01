/* --- --- --- --- --- --- --- ---
   ARCHIVO: script_analizador.js (Lógica FINAL: Arreglo de Subida Local)
   --- --- --- --- --- --- --- --- */

document.addEventListener('DOMContentLoaded', () => {
    
    // --- 1. Definición de Elementos y Variables ---
    const inputFile = document.getElementById('input-file');
    const cameraButton = document.getElementById('camera-button');
    const pasteButton = document.getElementById('paste-button');
    const analizarButton = document.getElementById('analizar-button');
    const userQuestion = document.getElementById('user-question');
    const sendButton = document.getElementById('send-button');
    const chatMessages = document.getElementById('chat-messages');
    
    const imageStatus = document.getElementById('image-status');
    const webcamPreview = document.getElementById('webcam-preview');
    const currentImage = document.getElementById('current-image');
    const previewPlaceholder = document.getElementById('preview-placeholder');
    
    const predictedClassHidden = document.getElementById('predicted-class-hidden');
    const predictedConfidenceHidden = document.getElementById('predicted-confidence-hidden');
    
    // Asumimos que esta URL es correcta (puerto 8000 para FastAPI)
    const API_URL = 'http://127.0.0.1:8000/api/analizar'; 

    let fileToUpload = null; 
    let stream = null; // Para la cámara

    // --- 2. Funciones de UTILIDAD ---

    function updateImagePreview(src) {
        currentImage.src = src;
        currentImage.style.display = 'block';
        webcamPreview.style.display = 'none';
        previewPlaceholder.style.display = 'none';
        imageStatus.textContent = 'Imagen lista para analizar.';
        imageStatus.style.color = 'var(--color-success)';
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
    }

    function createMessage(text, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('chat-message');
        messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');
        
        if (!isUser) { // Si es un mensaje del bot, añade el icono
            const botIcon = document.createElement('div');
            botIcon.classList.add('bot-icon');
            botIcon.textContent = '🤖';
            messageDiv.appendChild(botIcon);
        }

        const p = document.createElement('p');
        p.innerHTML = text; 
        messageDiv.appendChild(p);

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight; 
        return messageDiv;
    }

    function generateBotResponse(prediccion, confianza, pregunta) {
        let text = '';

        if (prediccion && pregunta.toLowerCase().includes('qué es esto')) {
            text += `Análisis completado. Con <span class="ia-confidence-result">${confianza.toFixed(2)}%</span> de confianza, detecto que esto es un: <span class="ia-class-result">${prediccion}</span>.`;
            
            if (prediccion === '01_placa_madre') {
                text += '<br><br>Detecto una placa madre. ¿Quieres que intente identificar los puertos o ranuras principales? (Necesito más entrenamiento para eso 😉)';
            } else if (prediccion === '99_fondo') {
                text += '<br><br>Esto no parece ser un componente. Por favor, intenta con una foto más cercana y bien iluminada.';
            }
        } else if (prediccion && prediccion !== '---') { 
            text += `Actualmente, solo puedo clasificar la imagen. Ya sé que es una <span class="ia-class-result">${prediccion}</span>.`;
            text += `<br>Mi confianza en esta clasificación es del <span class="ia-confidence-result">${confianza.toFixed(2)}%</span>.`;
            text += `<br><br>Para más detalles, primero tengo que ir al "gimnasio" (entrenamiento con más datos). ¡Pero estoy listo para aprender!`;
        } else {
            text += 'Por favor, asegúrate de haber subido y analizado una imagen antes de hacer una pregunta.';
        }

        createMessage(text, false);
    }

    // --- 3. LÓGICA DE INTERACCIÓN (NUEVA ESCUCHA PARA SUBIDA LOCAL) ---
    
    // Lógica para previsualizar el archivo local subido (¡SOLUCIÓN AL BUG!)
    inputFile.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            fileToUpload = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                updateImagePreview(e.target.result); // Actualiza la previsualización
            };
            reader.readAsDataURL(file);
        }
    });

    // --- 4. LÓGICA DE CÁMARA (Mantenida) ---
    cameraButton.addEventListener('click', async () => {
        try {
            if (stream) {
                const canvas = document.createElement('canvas');
                canvas.width = webcamPreview.videoWidth;
                canvas.height = webcamPreview.videoHeight;
                canvas.getContext('2d').drawImage(webcamPreview, 0, 0, canvas.width, canvas.height);
                
                canvas.toBlob((blob) => {
                    fileToUpload = new File([blob], 'webcam_capture.png', { type: 'image/png' });
                    updateImagePreview(canvas.toDataURL('image/png'));
                }, 'image/png');

            } else {
                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                webcamPreview.srcObject = stream;
                webcamPreview.style.display = 'block';
                currentImage.style.display = 'none';
                previewPlaceholder.style.display = 'none';
                imageStatus.textContent = 'Cámara activa. Haz clic en "Cámara" de nuevo para tomar la foto.';
                imageStatus.style.color = 'var(--color-warning)';
            }
        } catch (err) {
            imageStatus.textContent = 'ERROR: No se pudo acceder a la cámara. Revisa permisos.';
            imageStatus.style.color = '#ff4d4d'; 
            console.error('Error al acceder a la cámara:', err);
        }
    });

    // --- 5. LÓGICA DE PORTAPAPELES (Mantenida) ---
    pasteButton.addEventListener('click', async () => {
        try {
            if (navigator.clipboard && navigator.clipboard.read) {
                const items = await navigator.clipboard.read();
                for (const item of items) {
                    if (item.types.includes('image/png')) {
                        const blob = await item.getType('image/png');
                        fileToUpload = new File([blob], 'clipboard_paste.png', { type: 'image/png' });
                        const url = URL.createObjectURL(blob);
                        updateImagePreview(url);
                        return;
                    }
                }
                imageStatus.textContent = 'ERROR: El portapapeles no contiene una imagen PNG.';
                imageStatus.style.color = '#ff4d4d';
            } else {
                imageStatus.textContent = 'ERROR: El navegador no soporta la lectura directa del portapapeles.';
                imageStatus.style.color = '#ff4d4d';
            }
        } catch (err) {
            imageStatus.textContent = 'ERROR: No se pudo leer el portapapeles. (Permisos).';
            imageStatus.style.color = '#ff4d4d';
        }
    });

    // --- 6. LÓGICA DE ANÁLISIS DE IMAGEN (Llamada a FastAPI - Mantenida) ---
    analizarButton.addEventListener('click', async () => {
        if (!fileToUpload) {
            createMessage('Por favor, selecciona o pega una imagen para empezar el análisis.', false);
            return;
        }

        imageStatus.textContent = 'Enviando a la IA... (Puerto 8000)';
        imageStatus.style.color = 'var(--color-understated)'; 
        predictedClassHidden.textContent = 'Analizando...';
        predictedConfidenceHidden.textContent = '...';

        const formData = new FormData();
        formData.append('archivo', fileToUpload); 

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Error HTTP: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();

            predictedClassHidden.textContent = data.prediccion;
            predictedConfidenceHidden.textContent = data.confianza;
            
            const resultadoClasificacion = `¡Análisis de Componente Completado!
            Detecto: <span class="ia-class-result">${data.prediccion}</span>.
            Confianza: <span class="ia-confidence-result">${data.confianza}%</span>.
            
            Ahora puedes hacerme una pregunta sobre el componente (Ej: ¿Qué es esto?, ¿Dónde va la RAM?).`;

            createMessage(resultadoClasificacion, false);
            imageStatus.textContent = 'Análisis completo. Pregunta al bot.';
            imageStatus.style.color = 'var(--color-primary)';

        } catch (error) {
            console.error('Error al conectar con la API de IA:', error);
            imageStatus.textContent = 'ERROR: No se pudo conectar con el servidor de IA (puerto 8000).';
            imageStatus.style.color = '#ff4d4d'; 
            createMessage('ERROR DE CONEXIÓN: El backend de la IA (FastAPI) no está encendido. Presiona CTRL+C y vuelve a ejecutar python api_backend.py.', false);
        }
    });

    // --- 7. LÓGICA DE CHAT (Mantenida) ---
    sendButton.addEventListener('click', () => handleChatSubmit());
    userQuestion.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleChatSubmit();
        }
    });

    function handleChatSubmit() {
        const question = userQuestion.value.trim();
        if (question === '') return;

        createMessage(question, true);
        userQuestion.value = '';

        const prediccion = predictedClassHidden.textContent;
        const confianza = parseFloat(predictedConfidenceHidden.textContent);

        if (prediccion === '---' || prediccion === 'Analizando...') {
            createMessage('Necesito que primero analices una imagen. Haz clic en "Analizar y Predecir".', false);
        } else {
            generateBotResponse(prediccion, confianza, question);
        }
    }
});