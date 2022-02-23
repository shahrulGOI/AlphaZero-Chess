#define process which will listen to the pipe for observation (game state) and return prediction from policy and value network
from multiprocessing import connection, Pipe
from threading import Thread

import numpy as np

from AlphaBeta.config import Config


class ChessModelAPI: #defines process and listen to which pipe to observe, and return the prediction from policy and value network

    # noinspection PyUnusedLocal
    def __init__(self, agent_model):  # ChessModel #makes prediction
        #agen_model is used to train model to make predictions
        self.agent_model = agent_model
        self.pipes = []

    def start(self): #stars the tread and listen to the pipe and make predictions
        prediction_worker = Thread(target=self._predict_batch_worker, name="prediction_worker")
        prediction_worker.daemon = True
        prediction_worker.start()

    def create_pipe(self):
        #creates a two-way pipe and returns one end of the connection 
        #Return connection is the other end of the pipe
        me, you = Pipe()
        self.pipes.append(me)
        return you

    def _predict_batch_worker(self):
        #waits for the instruction of self.pipe for observation. 
        #outputs the prediction and value networks when observations is received and repeats
        while True:
            ready = connection.wait(self.pipes,timeout=0.001)
            if not ready:
                continue
            data, result_pipes = [], []
            for pipe in ready:
                while pipe.poll():
                    data.append(pipe.recv())
                    result_pipes.append(pipe)

            data = np.asarray(data, dtype=np.float32)
            policy_ary, value_ary = self.agent_model.model.predict_on_batch(data)
            for pipe, p, v in zip(result_pipes, policy_ary, value_ary):
                pipe.send((p, float(v)))
