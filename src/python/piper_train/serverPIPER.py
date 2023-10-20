#!/usr/bin/env python3
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from concurrent import futures
import numpy as np
import ASH_grpc_pb2
import ASH_grpc_pb2_grpc
import grpc
from espeak_phonemizer import Phonemizer

from .vits.lightning import VitsModel
from .vits.utils import audio_float_to_int16
from .vits.wavfile import write as write_wav
from .phonemize import *
_LOGGER = logging.getLogger("piper_train.infer")
import threading

# Define a lock to control access to the processing function
processing_lock = threading.Lock()
CURRENT_MODEL = None
vits_models = {}
model = None

class ServerPIPERService(ASH_grpc_pb2_grpc.ServerTTSService):
    def SynthesizePiper(self, request, context):
        # Get the lock
        with processing_lock:
            print(time.time())
            piper_config = {
                "input_model": request.model,
                "input_text": request.text,
                "noise_scale": request.noise_scale,
                "length_scale": request.length_scale,
                "noise_scale_w": request.noise_scale_w,
                "speaker_id": request.speaker_id,
            }
            print(piper_config)
            # Process the text using piper logic
            start = time.perf_counter()
            processed_audio_bytes = piper_TTS(**piper_config)
            print("time: ", time.perf_counter() - start)
            

            # if doing local rvc
            if request.auto_rvc:
                    # Create and return the response
                channel2 = grpc.insecure_channel("127.0.0.1:50052",options=[
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),  # Set the maximum send message length to 100 MB
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024)  # Set the maximum receive message length to 100 MB
                ])
                rvc_stub = ASH_grpc_pb2_grpc.ServerRVCServiceStub(channel2)

                    # requestr to RVC 
                rvc_request = ASH_grpc_pb2.ServerRVCRequest(
                    sid=0,
                    audio_bytes=processed_audio_bytes,
                    f0_up_key=0.0,
                    f0_file=None,
                    f0_method="crepe-tiny", # prob better to use harvest for tts
                    file_index=None,
                    # file_index="logs/ashera2/added_IVF817_Flat_nprobe_1_ashera2_v2.index", # maybe not
                    file_index2=None,
                    index_rate=0.88,
                    filter_radius=3,
                    resample_sr=0,
                    rms_mix_rate=1,
                    protect=0.33,
                    version="v2",
                    tgt_sr=40000
                )
                rvc_response = rvc_stub.ProcessAudio(rvc_request)
                rvc_audio_bytes = rvc_response.audio_bytes

                response = ASH_grpc_pb2.ServerPIPERResponse(audio_bytes=rvc_audio_bytes)
            else:
                response = ASH_grpc_pb2.ServerPIPERResponse(audio_bytes=processed_audio_bytes)
            return response
            
def piper_TTS(input_model, input_text, noise_scale=0.667, length_scale=1.1, noise_scale_w=0.8 ,speaker_id=None):
    global model

    phonemizer = Phonemizer(default_voice="en-us")

    # Inference
    phoneme_ids = phonemes_to_ids(phonemize(input_text, phonemizer))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    text = [torch.LongTensor(phoneme_ids).unsqueeze(0).to(device)]
    text_lengths = [torch.LongTensor([len(phoneme_ids)]).to(device)]
    scales = [torch.Tensor([noise_scale, length_scale, noise_scale_w]).to(device)]

    with torch.no_grad():  # Avoid gradient computations and memory overhead
        audio = model(text[0], text_lengths[0], scales[0], sid=speaker_id).cpu().detach().numpy()
        audio_data = audio[0][0].tobytes()

    # Explicitly delete intermediate tensors and free memory
    del phonemizer, phoneme_ids, device, text, text_lengths, scales, audio

    return audio_data    

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
    ('grpc.max_receive_message_length', 100 * 1024 * 1024)  # Set the maximum receive message length to 100 MB
])
    ASH_grpc_pb2_grpc.add_ServerPIPERServiceServicer_to_server(ServerPIPERService(), server)
    server.add_insecure_port("[::]:50051")  # Use a different port number for the RVC server
    print("Starting server on port 50051")
    server.start()
    server.wait_for_termination()


def init_model(input_model):
    global model
    model = VitsModel.load_from_checkpoint(input_model, dataset=None)

    # Determine the device (cuda if available, otherwise cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the appropriate device
    model = model.to(device)
    # torch.set_float32_matmul_precision('high')
    # model = torch.compile(model, device=device)
    
    # Inference only
    model.eval()
    

    with torch.no_grad():
        model.model_g.dec.remove_weight_norm()


import argparse
import requests
def get_model(name):
    model_name = name.split("/")[-1]
    # check if url or name
    if name.startswith("http"):
        # download the model
        print("Downloading model from url...")
        resp = requests.get(base+name) # making requests to server

        with open(name, "wb") as f: # opening a file handler to create new file
                f.write(resp.content)

    elif name.startswith("./"):
        model_name = name.split("/")[-1]

    else:
        # download from base url
        base = "https://files.redshiftscience.com/api/public/dl/lMWjjCRp/piper/"

        # download the model
        print("Downloading model from url...")
        print(base+name)
        #urllib.request.urlretrieve(base + name, model_name)
        #wget.download(base + name, model_name)
        resp = requests.get(base+name) # making requests to server

        with open(name, "wb") as f: # opening a file handler to create new file
            f.write(resp.content)
    return model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="./ashera_piper.ckpt",
        help="Model name or url to download model from. ",
    )
    args = parser.parse_args()
    init_model(get_model(args.model))
    serve()
