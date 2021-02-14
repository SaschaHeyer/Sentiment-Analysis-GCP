FROM python:3.7-slim
RUN pip install streamlit
COPY app.py ./app.py
CMD streamlit run app.py --server.port $PORT
