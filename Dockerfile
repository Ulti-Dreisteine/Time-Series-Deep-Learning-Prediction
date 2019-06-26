FROM harbor.querycap.com/rk-ai/ai-pytorch1.0:0.0.1
RUN pip install requirements.txt

RUN apt-get install git

WORKDIR /workspace
RUN chmod -R a+w /workspace