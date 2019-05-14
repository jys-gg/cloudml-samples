FROM tensorflow/tensorflow:1.8

# [REQUIRED]: Installs cloudml-hypertune
RUN pip install cloudml-hypertune
# [REQUIRED]: Installs wrapt which is required by sitecustomize.py
RUN pip install wrapt==1.10.10

# [REQUIRED] Copy sitecustomize.
RUN mkdir /var/sitecustomize
COPY sitecustomize.py /var/sitecustomize/
RUN chmod +x /var/sitecustomize/sitecustomize.py
ENV PYTHONPATH /var/sitecustomize


# Copies trainer file and set up the entrypoint
COPY train.py /root/train.py
ENTRYPOINT ["python", "/root/train.py"]

