# Unexpected easy Sentiment Analysis using BERT on Google Cloud Platform

This repository contains the demo code for the [DoiT blog article](https://blog.doit-intl.com/performing-surprisingly-easy-sentiment-analysis-on-google-cloud-platform-fc26b2e2b4b).

For a demo head over to [https://sentiment.practical-machine-learning.com/](https://sentiment.practical-machine-learning.com/)
## What is covered
As part of this article, we train and deploy a serverless Sentiment Analysis API to GCP by using BERT, TensorFlow, FastAPI, Python, Google AI Platform Training, Google Storage, Cloud Build, Cloud Container Registry, and Cloud Run.


## Training

The `training` folder contains the logic required to train the sentiment model. 

Adapt `training/cloudbuild.yaml` to your GCP environment.

To build the training image used for AI Platform run

```
gcloud builds submit --config cloudbuild.yaml
```

To start the training run
```
export JOB_NAME=bert_$(date +%Y%m%d_%H%M%S)
export IMAGE_URI=gcr.io/machine-learning-sascha/sentiment-training:latest

export REGION=us-west1

gcloud config set project machine-learning-sascha 

gcloud ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  --scale-tier=BASIC_GPU
```

## Prediction

Adapt `prediction/cloudbuild.yaml` to your GCP environment.

Deploy the application to Cloud Run using Cloud Build
```
gcloud builds submit --config cloudbuild.yaml
```
