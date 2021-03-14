FROM mecab-python3:latest
RUN pip3 install flask tensorflow==1.13.1 transformers==3.5.1
ADD . /www/bert-api
WORKDIR /www/bert-api
CMD flask run
