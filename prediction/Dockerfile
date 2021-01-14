FROM tiangolo/uvicorn-gunicorn:python3.8
RUN pip install --no-cache-dir fastapi==0.63.0 transformers==4.1.1 tensorflow==2.4.0 gunicorn==20.0.4 uvicorn==0.13.3
COPY main.py ./main.py
COPY ./model /model