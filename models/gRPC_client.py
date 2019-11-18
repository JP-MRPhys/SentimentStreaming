import grpc
import requests
import tensorflow as tf

from serving.tensorflow_serving.apis import predict_pb2 #TODO fux this find the right version for tensorflow 13.1.1
from serving.tensorflow_serving.apis import prediction_service_pb2_grpc


tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def main(_):


  data = tokenized_data # TODO how to get tokenized_data

  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  # Send request
  # See prediction_service.proto for gRPC request/response details.
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'bert'
  request.model_spec.signature_name = 'bert_predictions'
  request.inputs['image_ids'].CopyFrom(
      tf.contrib.util.make_tensor_proto(data.inputs_ids, shape=[1]))
  request.inputs['image_mask'].CopyFrom(
      tf.contrib.util.make_tensor_proto(data.input_mask, shape=[1]))
  request.inputs['segment_ids'].CopyFrom(
      tf.contrib.util.make_tensor_proto(data.segment_ids, shape=[1]))
  result = stub.Predict(request, 10.0)  # 10 secs timeout

  print(result)


if __name__ == '__main__':
  tf.app.run()