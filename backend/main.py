from fastapi import FastAPI, File, UploadFile, Query
import cv2
import numpy as np
import io
from PIL import Image
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import uvicorn

app = FastAPI(
    title="API YOLO com FastAPI",
    description="API para processamento de imagens e detecção de objetos com YOLO, com personalização de borda e texto.",
    version="2.0"
)

@app.get("/")
async def root():
    return {"message": "Bem-vindo à API YOLO com FastAPI!"}

# Mapeamento dos modelos disponíveis para os seus respectivos caminhos de arquivo
modelFiles = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt"
}

# Variáveis globais para o modelo atual
currentModel = "yolov8n"
detector = YOLO(modelFiles[currentModel])

@app.post("/mudar_modelo/")
async def mudar_modelo(model_version: str = Query("yolov8n", enum=list(modelFiles.keys()))):
    global detector, currentModel
    if model_version != currentModel:
        detector = YOLO(modelFiles[model_version])
        currentModel = model_version
        return {"message": f"Modelo alterado para {model_version}"}
    return {"message": f"O modelo atual já é {model_version}"}

@app.post("/processar_imagem/")
async def processar_imagem(
    image_file: UploadFile = File(...),
    selected_model: str = Query("yolov8n", enum=list(modelFiles.keys())),
    min_confidence: float = Query(0.25, description="Confiança mínima para detecção"),
    pad_thickness: int = Query(50, description="Espessura da borda"),
    pad_color: str = Query("50,50,50", description="Cor da borda (R,G,B)"),
    text_scale: float = Query(0.7, description="Escala do texto"),
    text_thickness: int = Query(2, description="Espessura do texto"),
    text_color: str = Query("255,255,255", description="Cor do texto (R,G,B)"),
    bg_color: str = Query("0,0,0", description="Cor de fundo para o texto (R,G,B)"),
    bg_opacity: float = Query(0.5, description="Opacidade do fundo (0 a 1)")
):
    global detector, currentModel

    # Atualiza o modelo se uma versão diferente for solicitada
    if selected_model != currentModel:
        detector = YOLO(modelFiles[selected_model])
        currentModel = selected_model

    # Lê a imagem e converte para o formato OpenCV
    image_data = await image_file.read()
    pilImage = Image.open(io.BytesIO(image_data))
    cvImage = cv2.cvtColor(np.array(pilImage), cv2.COLOR_RGB2BGR)

    # Realiza a detecção de objetos
    detectionResults = detector(cvImage)

    # Converte as strings de cor para tuplas
    borderColor = tuple(map(int, pad_color.split(",")))
    annotationColor = tuple(map(int, text_color.split(",")))
    backgroundColor = tuple(map(int, bg_color.split(",")))

    # Adiciona borda à imagem
    borderedImage = cv2.copyMakeBorder(
        cvImage, pad_thickness, pad_thickness, pad_thickness, pad_thickness,
        cv2.BORDER_CONSTANT, value=borderColor
    )

    # Percorre as detecções e anota a imagem
    for result in detectionResults:
        for box in result.boxes:
            conf = box.conf[0].item()
            if conf >= min_confidence:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{detector.names[int(box.cls[0])]} {conf:.2f}"
                
                # Ajusta as coordenadas para levar em conta a borda
                x1 += pad_thickness
                y1 += pad_thickness
                x2 += pad_thickness
                y2 += pad_thickness

                # Desenha a caixa delimitadora
                cv2.rectangle(borderedImage, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Determina o tamanho e a posição do texto
                font = cv2.FONT_HERSHEY_SIMPLEX
                (textWidth, textHeight), _ = cv2.getTextSize(label, font, text_scale, text_thickness)
                textOrigin = (x1, y1 - 10)

                # Desenha um fundo semitransparente para o texto
                overlay = borderedImage.copy()
                cv2.rectangle(
                    overlay,
                    (textOrigin[0], textOrigin[1] - textHeight - 5),
                    (textOrigin[0] + textWidth + 5, textOrigin[1] + 5),
                    backgroundColor,
                    -1
                )
                cv2.addWeighted(overlay, bg_opacity, borderedImage, 1 - bg_opacity, 0, borderedImage)
                # Escreve o texto de anotação
                cv2.putText(borderedImage, label, textOrigin, font, text_scale, annotationColor, text_thickness, cv2.LINE_AA)

    # Codifica a imagem anotada como JPEG e a retorna
    success, encodedImage = cv2.imencode(".jpg", borderedImage)
    if not success:
        return {"error": "Falha ao codificar a imagem."}
    return StreamingResponse(io.BytesIO(encodedImage.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
