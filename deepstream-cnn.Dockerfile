# set base image (host OS)
FROM nvcr.io/nfgnkvuikvjm/mdx-v2-0/mdx-perception:2.1

# set the working directory in the container
WORKDIR /opt/nvidia/deepstream/deepstream-6.4/sources/apps/sample_apps/deepstream-fewshot-learning-app

# copy the dependencies file to the working directory
COPY ./deepstream/configs/cnn-models/* ./

# copy the start script
COPY ./deepstream/init-scripts/ds-start.sh ./

