FROM python:2.7-onbuild

COPY . /source
WORKDIR /source
RUN pip install -r requirements_inference.txt

EXPOSE 5000
CMD python InferenceClient.py
