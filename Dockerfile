# Runs on Hugging Face Spaces (Docker SDK, port 7860) and on Render (sets $PORT).
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# Spaces runs the container as uid 1000: keep HOME (model weights, streamlit config) inside a writable folder
ENV HOME=/app/home PYTHONUNBUFFERED=1 PORT=7860
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# pre-download the three TorchXRayVision encoders so the first prediction is fast
RUN mkdir -p $HOME && python -c "import torchxrayvision as x; [x.models.DenseNet(weights=w) for w in ['densenet121-res224-nih','densenet121-res224-mimic_ch','densenet121-res224-pc']]" \
 && chmod -R a+rwX /app
EXPOSE 7860
CMD streamlit run dashboard/app.py --server.port=${PORT} --server.address=0.0.0.0 --browser.gatherUsageStats=false
