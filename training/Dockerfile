FROM tensorflow/tensorflow:2.2.2-gpu
WORKDIR /root

RUN pip3 install transformers==4.1.1 google-cloud-storage==1.35.0 scikit-learn==0.24.0 pandas==1.1.5

COPY train.py ./train.py

ENTRYPOINT ["python3", "train.py"]