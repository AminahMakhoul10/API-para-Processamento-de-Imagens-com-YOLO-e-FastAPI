# 🌐 API para Processamento de Imagens com YOLO e FastAPI

API para processamento de imagens e detecção de objetos com YOLO, com personalização de borda e texto. 

---

## 🚀 Funcionalidades

- **Detecção de Objetos:**  
  Utiliza o modelo YOLO (via Ultralytics) para identificar objetos em imagens.

- **Personalização das Anotações:**  
  Permite configurar:
  - Modelo YOLO a ser utilizado (ex.: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`)
  - Confiança mínima para considerar uma detecção (valor entre 0 e 1)
  - Espessura e cor da borda
  - Escala, espessura e cor do texto de anotação
  - Cor e opacidade do fundo do texto

- **Endpoints Disponíveis:**
  - **GET `/`**  
    Retorna uma mensagem de boas-vindas.
  - **POST `/mudar_modelo/`**  
    Atualiza o modelo de detecção se uma nova versão for solicitada.
  - **POST `/processar_imagem/`**  
    Processa uma imagem enviada, aplica a detecção e retorna a imagem anotada.

---

- *Bibliotecas*: 
    - **FastAPI**: Framework web para construção de APIs rápidas e eficientes em Python.
    - **Uvicorn**: Servidor ASGI rápido e leve, usado para rodar aplicações FastAPI.
    - **OpenCV**: Biblioteca de visão computacional utilizada para manipulação e processamento de imagens.
    - **Pillow**: Biblioteca para manipulação e edição de imagens em Python.
    - **NumPy**: Biblioteca para operações matemáticas e manipulação de arrays, essencial para processamento de imagens.
    - **Ultralytics (YOLO)**: Implementação do modelo YOLO (You Only Look Once) para detecção de objetos em imagens e vídeos.

---

## Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes do Python)

---

## Instalação e Execução

### 1. Criar e Ativar o Ambiente Virtual

Abra o terminal e execute os seguintes comandos:

```bash
python -m venv venv  # Cria o ambiente virtual

# Para ativar o ambiente:
# No Linux/macOS:
source venv/bin/activate
# No Windows:
venv\Scripts\activate
```

### 2. Atualizar o pip e Instalar as Dependências

Execute os comandos abaixo para atualizar o pip e instalar as bibliotecas necessárias:

```bash
python -m pip install --upgrade pip
pip install fastapi uvicorn opencv-python pillow numpy ultralytics
pip install "fastapi[standard]"
```

### 3. Navegar até a Pasta da API

```bash
cd backend
```

### 4. Executar a API

Para rodar a aplicação via Python utilizando Uvicorn, execute:

```bash
python -m uvicorn main:app --reload --port 8080
```

A API ficará disponível nos seguintes endereços:

- **Documentação Interativa (Swagger UI):** [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs)

---

## Uso da API

### **GET `/`**

**Descrição:**  
Retorna uma mensagem de boas-vindas.

**Exemplo de Resposta:**

```json
{
    "message": "Bem-vindo à API YOLO com FastAPI!"
}
```

### **POST `/mudar_modelo/`**

**Descrição:**  
Atualiza o modelo de detecção se uma versão diferente for solicitada.

**Parâmetro (Query):**

- `model_version` (string): Versão do modelo desejado. Valores válidos: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`.

**Exemplo de Resposta:**

```json
{
    "message": "Modelo alterado para yolov8s"
}
```

### **POST `/processar_imagem/`**

**Descrição:**  
Processa uma imagem enviada, realiza a detecção com YOLO e retorna a imagem com anotações.

**Parâmetros (Query):**

- `selected_model` (string): Modelo a ser utilizado (ex.: `yolov8n`, etc.).
- `min_confidence` (float): Confiança mínima para detecção (valor entre 0 e 1).
- `pad_thickness` (int): Espessura da borda.
- `pad_color` (string): Cor da borda no formato "R,G,B".
- `text_scale` (float): Escala do texto de anotação.
- `text_thickness` (int): Espessura do texto.
- `text_color` (string): Cor do texto no formato "R,G,B".
- `bg_color` (string): Cor de fundo do texto no formato "R,G,B".
- `bg_opacity` (float): Opacidade do fundo (valor entre 0 e 1).

**Corpo (Body):**

- `image_file`: Arquivo da imagem (ex.: JPEG, PNG).

**Exemplo de Uso:**

Envie um formulário multipart com a imagem e os parâmetros desejados. A resposta será a imagem processada em formato JPEG.

---

## 🔗 Exemplo de Clone e Instalação

Clone o repositório para o seu ambiente local:

```bash
git clone https://github.com/AminahMakhoul10/API-para-Processamento-de-Imagens-com-YOLO-e-FastAPI.git
```

Siga as instruções de criação do ambiente virtual, instalação de dependências e execução descritas acima.

## 🌐 Após a execução, acesse este link para testar os endpoints via Swagger UI.
```bash
http://127.0.0.1:8080/docs
 ```
---

Com estas instruções, você estará apto a configurar, executar e utilizar a API para processamento de imagens com YOLO e FastAPI.

# Front-end streamlit

Para acessar o front-end execute:

```bash
streamlit run app.py
```
logo, o link para acesso será este:

```bash
http://localhost:8501/
```
