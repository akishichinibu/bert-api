FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install software-properties-common sudo curl make -y
RUN apt-get install git -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update
RUN apt-get install python3.6 -y
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.6 get-pip.py && rm get-pip.py
RUN ln -s /usr/bin/python3.6 /usr/bin/python && ln -s /usr/bin/pip3.6 /usr/bin/pip3
