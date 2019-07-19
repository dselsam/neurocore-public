from concurrent import futures
import time
import tensorflow as tf
import numpy as np
import sys
import json
import scipy
import os
import logging
from collections import namedtuple
import grpc
import numpy as np
import neurosat_pb2
import neurosat_pb2_grpc

NeuroSATOutputs = namedtuple('NeuroSATOutputs', ['pi_core_var_ps'])

python_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "python")
print("Adding %s to path..." % python_dir)
sys.path.append(python_dir)
from neurosat import apply_neurosat, NeuroSATParams, NeuroSATArgs, NeuroSATGuesses

def train_dir(train_id):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "networks", "train_id%d" % train_id)


class NeuroSATServer(neurosat_pb2_grpc.NeuroSATServerServicer):
    def __init__(self, train_id, checkpoint):
        with open(os.path.join(train_dir(train_id), "cfg.json"), "r") as f: cfg = json.load(f)
        self.n_vars     = tf.placeholder(tf.int64, shape=[])
        self.n_clauses  = tf.placeholder(tf.int64, shape=[])
        self.CL_idxs    = tf.placeholder(tf.int32, shape=[None, 2])
        self.v_itau     = tf.placeholder(tf.float32, shape=[])
        self.c_itau     = tf.placeholder(tf.float32, shape=[])

        params          = NeuroSATParams(cfg=cfg)
        nguesses        = apply_neurosat(cfg=cfg, params=params,
                                         args=NeuroSATArgs(self.n_vars, self.n_clauses, self.CL_idxs))

        self.guesses    = NeuroSATOutputs(pi_core_var_ps=tf.nn.softmax(nguesses.pi_core_var_logits * self.v_itau))


        self.sess       = tf.Session()
        saver           = tf.train.Saver()

        saver.restore(self.sess, os.path.join(train_dir(train_id), checkpoint))
        self.dummy_query()

    def dummy_query(self):
        print("Dummy query: ", self.sess.run(self.guesses, feed_dict={
            self.n_vars:3,
            self.n_clauses:2,
            self.CL_idxs:np.array([[0, 0], [0, 1], [0,2], [1, 3], [1, 4], [1, 5]]),
            self.v_itau:4.0,
            self.c_itau:4.0
        }))

    def query_neurosat(self, args, context):
        try:
            assert(args.n_vars > 0)
            assert(args.n_clauses > 0)
            assert(len(args.C_idxs) > 0)
            assert(len(args.L_idxs) > 0)
            assert(len(args.C_idxs) == len(args.L_idxs))

            print("n_vars: ", args.n_vars)
            print("n_clauses: ", args.n_clauses)
            print("n_cells: ", len(args.C_idxs))
            print("v_itau: ", args.v_itau)
            print("c_itau: ", args.c_itau)

            CL_idxs = np.concatenate([np.expand_dims(np.array(args.C_idxs), 1),
                                      np.expand_dims(np.array(args.L_idxs), 1)],
                                     axis=1)

            t_start = time.time()
            guesses = self.sess.run(self.guesses, feed_dict={
                self.n_vars:args.n_vars,
                self.n_clauses:args.n_clauses,
                self.CL_idxs:CL_idxs,
                self.v_itau:args.v_itau,
                self.c_itau:args.c_itau
            })
            n_secs_gpu = time.time() - t_start
            assert(np.size(guesses.pi_core_var_ps) == args.n_vars)
            assert(np.size(guesses.pi_core_clause_ps) == args.n_clauses)
            return neurosat_pb2.NeuroSATGuesses(
                success=True,
                msg="success",
                n_secs_gpu=n_secs_gpu,
                pi_core_var_ps=guesses.pi_core_var_ps
            )
        except tf.errors.OpError as e:
            return neurosat_pb2.NeuroSATGuesses(
                success=False,
                msg="OpError: " + str(e),
                n_secs_gpu=0.0,
                pi_core_var_ps=[]
            )
        except Exception as e:
            return neurosat_pb2.NeuroSATGuesses(
                success=False,
                msg="PyError: " + str(e),
                n_secs_gpu=0.0,
                pi_core_var_ps=[]
            )
        except:
            return neurosat_pb2.NeuroSATGuesses(
                success=False,
                msg="unknown exception type",
                n_secs_gpu=0.0,
                pi_core_var_ps=[]
            )

def serve(opts):
    server_options = [
        ('grpc.max_send_message_length', opts.max_send_message_length),
        ('grpc.max_receive_message_length', opts.max_receive_message_length)
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=opts.max_workers), options=server_options)
    neurosat_pb2_grpc.add_NeuroSATServerServicer_to_server(NeuroSATServer(train_id=opts.train_id, checkpoint=opts.checkpoint), server)
    server.add_insecure_port('[::]:%d' % opts.port)
    server.start()
    try:
        while True:
            time.sleep(1e8)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    logging.basicConfig()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_id', action='store', dest='train_id', type=int, default=21)
    parser.add_argument('--checkpoint', action='store', dest='checkpoint', type=str, default="model.ckpt-661581")
    parser.add_argument('--port', action='store', dest='port', type=int, default=50051)
    parser.add_argument('--max_workers', action='store', dest='max_workers', type=int, default=1)
    parser.add_argument('--max_send_message_length', action='store', dest='max_send_message_length', type=int, default=int(2 ** 30))
    parser.add_argument('--max_receive_message_length', action='store', dest='max_receive_message_length', type=int, default=int(2 ** 30))
    opts = parser.parse_args()

    print(opts)
    serve(opts)
