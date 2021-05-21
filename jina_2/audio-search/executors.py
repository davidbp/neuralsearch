import os
from typing import Dict, Optional, List, Iterable, Union, Tuple

import numpy as np


from jina import Executor, DocumentArray, requests, Document
import vggish
from vggish.vggish_input import waveform_to_examples

class VggishSegmenter(Executor):
    def __init__(self, window_length_secs=0.025, show_exc_info=True, *args, **kwargs):
        """
        :param frame_length: the number of samples in each frame
        :param hop_length: number of samples to advance between frames
        """
        super().__init__(*args, **kwargs)
        self.window_length_secs = window_length_secs
        self.show_exc_info = show_exc_info

    @requests
    def segment(self, docs, *args, **kwargs):

        for doc in docs:
            data, sample_rate = self.read_wav(doc.uri)
            mel_data = self.wav2mel(data, sample_rate)
            for idx, blob in enumerate(mel_data):
                #self.logger.debug(f'blob: {blob.shape}')
                doc = Document(offset=idx, weight=1.0, blob=blob)

    def wav2mel(self, blob, sample_rate):
        #self.logger.debug(f'blob: {blob.shape}, sample_rate: {sample_rate}')
        mel_spec = waveform_to_examples(blob, sample_rate).squeeze()
        #self.logger.debug(f'mel_spec: {mel_spec.shape}')
        return mel_spec

    def read_wav(self, uri):
        import soundfile as sf
        print(f'\n\nuri={uri}\n\n')
        print(f'\n\nos.path.exists(uri)={os.path.exists(uri)}\n\n')
        wav_data, sample_rate = sf.read(uri, dtype='int16')
        #self.logger.debug(f'sample_rate: {sample_rate}')
        if len(wav_data.shape) > 1:
            wav_data = np.mean(wav_data, axis=1)
        data = wav_data / sample_rate #32768.0
        return data, sample_rate



class Encoder(Executor):

    @requests
    def encode(self, docs: DocumentArray, *args, **kwargs):
        pass


class VectorIndexer(Executor):

    @requests
    def index(self, docs: DocumentArray, *args, **kwargs):
        pass



class DocIndexer(Executor):
    @requests
    def index(self, docs: DocumentArray, *args, **kwargs):
        pass
