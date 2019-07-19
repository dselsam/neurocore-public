import logging

import grpc

import neurosat_pb2
import neurosat_pb2_grpc

def run(opts):
    channel_options = [
        ('grpc.max_send_message_length', opts.max_send_message_length),
        ('grpc.max_receive_message_length', opts.max_receive_message_length)
    ]

    with grpc.insecure_channel("%s:%d" % (opts.host, opts.port), options=channel_options) as channel:
        print("Connecting...")
        stub = neurosat_pb2_grpc.NeuroSATServerStub(channel)
        print("Query...")
        response = stub.query_neurosat(neurosat_pb2.NeuroSATArgs(n_vars=3,
                                                                 n_clauses=4,
                                                                 itau=4.0,
                                                                 C_idxs=[0, 1, 2, 3],
                                                                 L_idxs=[1, 0, 2, 2]))

    print("success: ", response.success)
    print("msg: ", response.msg)
    print("n_secs_gpu: ", response.n_secs_gpu)        
    print("pi_core_var_ps: ", response.pi_core_var_ps)

if __name__ == '__main__':
    logging.basicConfig()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('host', action='store', type=str)
    parser.add_argument('--port', action='store', dest='port', type=int, default=50051)
    parser.add_argument('--max_send_message_length', action='store', dest='max_send_message_length', type=int, default=int(1024 ** 3))
    parser.add_argument('--max_receive_message_length', action='store', dest='max_receive_message_length', type=int, default=int(1024 ** 3))
    opts = parser.parse_args()
    run(opts)
