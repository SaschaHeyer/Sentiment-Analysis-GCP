export JOB_NAME=bert_$(date +%Y%m%d_%H%M%S)
export IMAGE_URI=gcr.io/machine-learning-sascha/sentiment-training:latest

export REGION=us-west1

gcloud config set project machine-learning-sascha 

gcloud ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  --scale-tier=BASIC_GPU 