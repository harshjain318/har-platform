FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-deploy.txt
COPY --chown=user . .
RUN mkdir -p uploads results && chown -R user:user /app
USER user
EXPOSE 7860
CMD ["streamlit","run","app.py","--server.port=7860","--server.address=0.0.0.0"]
