import grpc
import requests
import tensorflow as tf
from models.utils import download_and_load_datasets, create_tokenizer_from_hub_module
import bert.run_classifier
import numpy as np
import time


from tensorflow_serving.apis import predict_pb2 #TODO fux this find the right version for tensorflow 13.1.1
from tensorflow_serving.apis import prediction_service_pb2_grpc


tf.app.flags.DEFINE_string('server', '34.68.167.77:8500',
                           'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS
tokenizer = create_tokenizer_from_hub_module()
print("Tokenizer created")


if __name__ == '__main__':

    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


    train_df, test_df = download_and_load_datasets()

    print("Download complete")
    train = train_df.sample(len(train_df) - 24000)
    print(train.head(5))

    DATA_COLUMN = 'sentence'
    LABEL_COLUMN = 'polarity'
      # label_list is the list of labels, i.e. True, False or 0, 1 or 'dog', 'cat'
    label_list = [0, 1]

      # Use the InputExample class from BERT's run_classifier code to create examples from the data
    train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                                   # Globally unique ID for bookkeeping, unused in this example
                                                                                   text_a=x[DATA_COLUMN],
                                                                                   text_b=None,
                                                                                   label=x[LABEL_COLUMN]), axis=1)

    print(train_InputExamples)
      # We'll set sequences to be at most 128 tokens long.
    MAX_SEQ_LENGTH = 128

      # Convert our train and test features to InputFeatures that BERT understands.
    train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
    data_features=train_features[0:50]

    input_ids = [x.input_ids for x in data_features]
    input_mask = [x.input_mask for x in data_features]
    segment_ids = [x.segment_ids for x in data_features]

    batch_size=len(input_ids)

    print("Making request")

    start = time.time()

    # Send request
    # See prediction_service.proto for gRPC request/response details.
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'bert'
    request.model_spec.signature_name = 'bert_predictions'
    request.inputs['input_ids'].CopyFrom(
        tf.contrib.util.make_tensor_proto(input_ids, shape=[batch_size, MAX_SEQ_LENGTH]))
    request.inputs['input_mask'].CopyFrom(
        tf.contrib.util.make_tensor_proto(input_mask, shape=[batch_size, MAX_SEQ_LENGTH]))
    request.inputs['segment_ids'].CopyFrom(
        tf.contrib.util.make_tensor_proto(segment_ids, shape=[batch_size, MAX_SEQ_LENGTH]))
    result = stub.Predict(request, 100.0)  # 10 secs timeout

    end = time.time()

    print("Request out is in time: {}" .format(end-start))
    print(result)

    #ref: https://stackoverflow.com/questions/44785847/how-to-retrieve-float-val-from-a-predictresponse-object
    outputs_tensor_proto = result.outputs["predictated_labels"]
    shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
    #outputs = tf.constant(outputs_tensor_proto.int_val, shape=shape)
    outputs = np.array(outputs_tensor_proto.int_val).reshape(shape)

    print(np.shape(outputs))

    print(shape[0])