FROM czentye/opencv-video-minimal 
MAINTAINER Diogo Gomes <diogogomes@gmail.com> 

# -e "TX=Europe/Lisbon"
RUN apk add tzdata

RUN mkdir -p /app
RUN mkdir -p /opt

RUN wget https://pjreddie.com/media/files/yolov3.weights -O /opt/yolov3.weights
RUN wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O /opt/yolov3.cfg
RUN wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -O /opt/coco.names

COPY main.py /app
COPY requirements.txt /app

RUN pip install -r /app/requirements.txt 

WORKDIR /app

CMD ["/usr/bin/python3", "/app/main.py"] 
