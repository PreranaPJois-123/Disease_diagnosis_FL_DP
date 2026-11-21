# Works on Hugging Face Spaces (Docker), Render, Railway, etc.
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# pre-download the three TorchXRayVision encoders so the first prediction is fast
RUN python -c "import torchxrayvision as x; [x.models.DenseNet(weights=w) for w in ['densenet121-res224-nih','densenet121-res224-mimic_ch','densenet121-res224-pc']]"
ENV PORT=7860
EXPOSE 7860
CMD streamlit run dashboard/app.py --server.port=${PORT} --server.address=0.0.0.0
