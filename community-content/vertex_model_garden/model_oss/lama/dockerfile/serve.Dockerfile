FROM pytorch/torchserve:0.8.2-gpu

# install dependencies
RUN pip3 install transformers
RUN pip3 install stickytape

# create torchserve configuration file
USER root
RUN apt-get -y update
RUN apt-get -y install git vim

RUN git clone https://github.com/advimman/lama.git /home/lama
COPY model_oss/lama/lama.patch /home/lama
WORKDIR /home/lama/
RUN git apply /home/lama/lama.patch
RUN cp /home/lama/models/ade20k/color150.mat /home/model-server/color150.mat
RUN cp /home/lama/models/ade20k/object150_info.csv /home/model-server/object150_info.csv
RUN cp /home/lama/LICENSE /home/model-server/LICENSE

# copy model artifacts, custom handler and other dependencies
COPY model_oss/lama/handler.py /home/lama/bin/serve/
COPY model_oss/lama/requirements.txt /home/model-server/
RUN chmod -R 777 /home/model-server/

# install dependencies
RUN pip install -r /home/model-server/requirements.txt
RUN pip install --upgrade torch==2.0.1
RUN pip install google-cloud-storage==2.7.0

# build single file handler and model
RUN stickytape saicinpainting/training/trainers/default.py --add-python-path . --output-file bin/serve/model.py
RUN stickytape bin/serve/handler.py --add-python-path .  --output-file /home/model-server/handler_bundle.py
RUN cp bin/serve/model.py /home/model-server/model.py
WORKDIR /home/model-server

RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
RUN printf "\ninference_address=http://0.0.0.0:7080" >> /home/model-server/config.properties
RUN printf "\nmanagement_address=http://0.0.0.0:7081" >> /home/model-server/config.properties
USER model-server

# expose health and prediction listener ports from the image
EXPOSE 7080
EXPOSE 7081

# create model archive file packaging model artifacts and dependencies
RUN torch-model-archiver -f \
  --model-name=lama \
  --version=1.0 \
  --model-file=/home/model-server/model.py \
  --handler=/home/model-server/handler_bundle.py \
  --requirements-file=/home/model-server/requirements.txt \
  --extra-files="/home/model-server/requirements.txt,/home/model-server/model.py,/home/model-server/handler_bundle.py,/home/model-server/color150.mat,/home/model-server/object150_info.csv" \
  --export-path=/home/model-server/model-store

ENV TORCH_HOME=$(pwd)
ENV PYTHONPATH=$(pwd)
ENV MODEL_BUNDLE_NAME="lama_model_artifacts"

# run Torchserve HTTP serve to respond to prediction requests
CMD ["torchserve", \
     "--start", \
     "--ts-config=/home/model-server/config.properties", \
     "--models", \
     "lama=lama.mar", \
     "--model-store", \
     "/home/model-server/model-store"]
