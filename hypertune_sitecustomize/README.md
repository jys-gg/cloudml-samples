# sitecustomize for custom container hypertune

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

# Overview
The sitecustomize script does a monkey patching for tf summary writer to automatically report hptuing metrics. 
Users doesn't have to explicitly call the [cloudml-hypetune](https://pypi.org/project/cloudml-hypertune/) 
library to report metrics for tensorflow model training.

# Howto

* Installs the [cloudml-hypertune](https://pypi.org/project/cloudml-hypertune/)  package in the `Dockerfile`.
```
RUN pip install cloudml-hypertune
```

* Installs the [wrapt] package in the `Dockerfile`. It's required to do the monkey patching.
```
RUN pip install wrapt==1.10.10
```

* Setup `sitecustomize.py` in the container.
```
RUN mkdir /var/sitecustomize
COPY sitecustomize.py /var/sitecustomize/
RUN chmod +x /var/sitecustomize/sitecustomize.py
ENV PYTHONPATH /var/sitecustomize
```

# QuickStart
Run the sample in this directory:
```
# Build and push the docker image to Google Container Registry(GCR).
PROJECT_ID=<your-project-id>
IMAGE_REPO=<your-gcr-image-repo>
TAG=<your-image-tag>
IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO:$TAG
docker build -t $IMAGE_URI ./ && docker push $IMAGE_URI

/bin/bash submit.sh
```
