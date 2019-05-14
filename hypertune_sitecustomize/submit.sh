#!/bin/bash

# Build and push the docker image to Google Container Registry(GCR).
PROJECT_ID=<your-project-id>
IMAGE_REPO=<your-gcr-image-repo>
TAG=<your-image-tag>
IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO:$TAG
docker build -t $IMAGE_URI ./ && docker push $IMAGE_URI

# Create config.yaml
cat > config.yaml <<EOF
---
trainingInput:
  scaleTier: BASIC
  masterConfig:
    imageUri: $IMAGE_URI
  hyperparameters:
    goal: MINIMIZE
    hyperparameterMetricTag: "loss_1"
    maxTrials: 4
    maxParallelTrials: 2
    enableTrialEarlyStopping: True
    params:
    - parameterName: learning_rate
      type: DOUBLE
      minValue: 0.01
      maxValue: 0.1
    - parameterName: batch_size
      type: DISCRETE
      discreteValues:
      - 32
      - 64
      - 128
EOF

# Submit a job.
JOB_NAME=mnist_custom_container_$(date +%Y%m%d_%H%M%S)
gcloud alpha ml-engine jobs submit training $JOB_NAME \
  --region us-central1 \
  --config=config.yaml \



