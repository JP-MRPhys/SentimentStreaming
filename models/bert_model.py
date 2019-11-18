import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import os
import re
import pandas as pd
import datetime
from models.utils import download_and_load_datasets, create_tokenizer_from_hub_module
import bert.optimization
import bert.run_classifier
import bert.tokenization

#add to git


class BERT():

    def __init__(self):
        # These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 2e-5
        self.NUM_TRAIN_EPOCHS = 100
        # Warmup is a period of time where hte learning rate
        # is small and gradually increases--usually helps training.
        self.WARMUP_PROPORTION = 0.1
        # Model configs
        self.SAVE_CHECKPOINTS_STEPS = 500
        self.SAVE_SUMMARY_STEPS = 100
        # Compute # train and warmup steps from batch size
        self.BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
        self.max_seq_length = 128
        self.num_labels = 2
        self.batch_size = 10
        self.model_dir = "BERT"
        # these are features from the tokenizer output

        self.input_ids = tf.placeholder(tf.int32, [None, self.max_seq_length], name="input_ids")
        self.input_mask = tf.placeholder(tf.int32, [None, self.max_seq_length], name="input_mask")
        self.segment_ids = tf.placeholder(tf.int32, [None, self.max_seq_length], name="segment_ids")
        self.labels = tf.placeholder(tf.int32, [None], name= "labels")

        self.version=0 #default version is zero

        self.create_model()


    def create_model(self):
        """Creates a classification model."""

        bert_module = hub.Module(
            self.BERT_MODEL_HUB,
            trainable=True)
        bert_inputs = dict(
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            segment_ids=self.segment_ids)
        bert_outputs = bert_module(
            inputs=bert_inputs,
            signature="tokens",
            as_dict=True)

        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_outputs" for token-level output.
        output_layer = bert_outputs["pooled_output"]

        hidden_size = output_layer.shape[-1].value

        # Create our own layer to tune for politeness data.
        output_weights = tf.get_variable(
            "output_weights", [self.num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable("output_bias", [self.num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            # Dropout helps prevent overfitting
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            self.log_probs = tf.nn.log_softmax(logits, axis=-1)

            # Convert labels into one-hot encoding
            one_hot_labels = tf.one_hot(self.labels, depth=self.num_labels, dtype=tf.float32)

            self.predicted_labels = tf.squeeze(tf.argmax(self.log_probs, axis=-1, output_type=tf.int32), name="prediction")

            # If we're train/eval, compute loss between predicted and actual label
            per_example_loss = -tf.reduce_sum(one_hot_labels * self.log_probs, axis=-1)
            self.loss = tf.reduce_mean(per_example_loss, name="loss")
            print(self.loss)
            print(self.predicted_labels)

        print(output_bias)


    def train(self, train_features):

        num_train_steps = int(len(train_features) / self.BATCH_SIZE * self.NUM_TRAIN_EPOCHS)
        num_warmup_steps = int(num_train_steps * self.WARMUP_PROPORTION)

        train_op = bert.optimization.create_optimizer(
            self.loss,self.LEARNING_RATE, num_train_steps, num_warmup_steps, use_tpu=False)

        with tf.Session() as sess:

            init=tf.global_variables_initializer()
            sess.run(init)

            for epoch in range(0, int(self.NUM_TRAIN_EPOCHS)):
                print(epoch)
                for idx in range(0, len(train_features), self.batch_size):


                    train_features_batch =train_features[idx:idx+self.batch_size]
                    input_ids =[x.input_ids for x in train_features_batch]
                    input_mask =[x.input_mask for x in train_features_batch]
                    segment_ids =[x.segment_ids for x in train_features_batch]
                    labels = [x.label_id for x in train_features_batch]

                    feed_dict={ self.input_ids: input_ids,
                            self.segment_ids: segment_ids,
                            self.input_mask:input_mask,
                                self.labels: labels
                                }
                    _ = sess.run(train_op, feed_dict)


            print("saving the model")
            self.export_model(sess, version=2)

        print("training completed saving data")

    def inference(self, test_features, version=0):

        dirs=os.path.join(self.model_dir, str(version))

        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], dirs)

            graph=tf.get_default_graph()


            input_ids = [x.input_ids for x in test_features]
            input_mask = [x.input_mask for x in test_features]
            segment_ids = [x.segment_ids for x in test_features]

            self.input_ids = graph.get_tensor_by_name("input_ids:0")
            self.input_mask = graph.get_tensor_by_name("input_mask:0")
            self.segment_ids =graph.get_tensor_by_name("segment_ids:0")
            self.predicted_labels=graph.get_tensor_by_name("loss/predictions:0")


            dictionary = {self.input_ids: input_ids,
                         self.segment_ids: segment_ids,
                         self.input_mask: input_mask }

            predicted_sentiment = sess.run(self.predicted_labels, feed_dict=dictionary)

        return predicted_sentiment


    def export_model(self, session, version=0):
        """Exports the model so that it can used for batch predictions."""

        # session.run(tf.global_variables_initializer())
        # self.saver.restore(session, last_checkpoint)

        a  =tf.saved_model.utils.build_tensor_info(self.input_ids)
        b = tf.saved_model.utils.build_tensor_info(self.input_mask)
        c = tf.saved_model.utils.build_tensor_info(self.segment_ids)
        d = tf.saved_model.utils.build_tensor_info(self.predicted_labels)

        sinputs = {'input_ids': a, 'input_mask': b, 'segment_ids': c}
        soutputs = {'predictated_labels':  d}

        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=sinputs,
            outputs=soutputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        signature_map = {'bert_predictions': signature}

        dirs=os.path.join(self.model_dir, str(version))

        model_builder = tf.saved_model.builder.SavedModelBuilder(dirs)
        model_builder.add_meta_graph_and_variables(session,
                                                   tags=[tf.saved_model.tag_constants.SERVING],
                                                   signature_def_map=signature_map,
                                                   clear_devices=True)
        model_builder.save(as_text=False)  #to save as pb-text change this to true

    def retrain(self, train_features, version=0):

        num_train_steps = int(len(train_features) / self.BATCH_SIZE * self.NUM_TRAIN_EPOCHS)
        num_warmup_steps = int(num_train_steps * self.WARMUP_PROPORTION)


        dirs=os.path.join(self.model_dir, str(version))
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], dirs)

            graph=tf.get_default_graph()

            #get the ops
            self.input_ids = graph.get_tensor_by_name("input_ids:0")
            self.input_mask = graph.get_tensor_by_name("input_mask:0")
            self.segment_ids =graph.get_tensor_by_name("segment_ids:0")
            self.labels =graph.get_tensor_by_name("labels:0")
            self.loss = graph.get_tensor_by_name("loss/loss:0")


            train_op = bert.optimization.create_optimizer(
                self.loss,self.LEARNING_RATE, num_train_steps, num_warmup_steps, use_tpu=False)


            init=tf.global_variables_initializer()
            sess.run(init)

            for epoch in range(0, int(self.NUM_TRAIN_EPOCHS)):
                print(epoch)
                for idx in range(0, len(train_features), self.batch_size):


                    train_features_batch =train_features[idx:idx+self.batch_size]

                    input_ids =[x.input_ids for x in train_features_batch]
                    input_mask =[x.input_mask for x in train_features_batch]
                    segment_ids =[x.segment_ids for x in train_features_batch]
                    labels = [x.label_id for x in train_features_batch]


                    feed_dict={ self.input_ids: input_ids,
                            self.segment_ids: segment_ids,
                            self.input_mask:input_mask,
                                self.labels: labels
                                }
                    _ = sess.run(train_op, feed_dict)


            print("saving the model")
            self.export_model(sess, version=2)

        print("training completed saving data")

    def restore_model(self, version=0):

        dirs = os.path.join(self.model_dir, str(version))
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], dirs)

            graph = tf.get_default_graph()

            for op in graph.get_operations():
                print(op)

            # predicted_sentiment= sess.run(self.predicted_labels, feed_dict=dict)

if __name__ == '__main__':
    model = BERT()


    print("Tensorflow ")

    #input traning data


    train_df, test_df=download_and_load_datasets()

    print("Download complete")
    train = train_df.sample(len(train_df)-24000)

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
    tokenizer = create_tokenizer_from_hub_module()
    print("Tokenizer created")

    # We'll set sequences to be at most 128 tokens long.
    MAX_SEQ_LENGTH = 128

    # Convert our train and test features to InputFeatures that BERT understands.
    train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
    a=model.retrain(train_features)
    print(len(a))
