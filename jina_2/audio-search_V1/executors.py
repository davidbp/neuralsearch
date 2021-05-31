import os
from typing import Dict, Optional, List, Iterable, Union, Tuple
from collections import defaultdict

import numpy as np
import tensorflow as tf
import vggish
from vggish.vggish_input import waveform_to_examples
from vggish.vggish_params import INPUT_TENSOR_NAME, OUTPUT_TENSOR_NAME

from vggish.vggish_slim import load_vggish_slim_checkpoint, define_vggish_slim

from vggish.vggish_postprocess import Postprocessor

from jina import Executor, DocumentArray, requests, Document

cur_dir = os.path.dirname(os.path.abspath(__file__))

## think:
## 1) chunks query time
## 2) top k
## 3) make code reading audio and sending it (instead, fill Docs with mp3)

class VggishSegmenter(Executor):
    def __init__(self, window_length_secs=0.025, show_exc_info=True, *args, **kwargs):
        """
        :param frame_length: the number of samples in each frame
        :param hop_length: number of samples to advance between frames
        """
        super().__init__(*args, **kwargs)
        self.window_length_secs = window_length_secs
        self.show_exc_info = show_exc_info

    @requests()
    def segment(self, docs, *args, **kwargs):
        for doc in docs:
            data, sample_rate = self.read_wav(doc.uri) # keep data blob and sample_rate as tag in tags
            mel_data = self.wav2mel(data, sample_rate)
            for idx, blob in enumerate(mel_data):
                #self.logger.debug(f'blob: {blob.shape}')
                doc.chunks.append(Document(offset=idx, weight=1.0, blob=blob))

    def wav2mel(self, blob, sample_rate):
        #self.logger.debug(f'blob: {blob.shape}, sample_rate: {sample_rate}')
        mel_spec = waveform_to_examples(blob, sample_rate).squeeze()
        #self.logger.debug(f'mel_spec: {mel_spec.shape}')
        return mel_spec

    def read_wav(self, uri):
        import soundfile as sf
        #print(f'\n\nuri={uri}\n\n')
        #print(f'\n\nos.path.exists(uri)={os.path.exists(uri)}\n\n')
        wav_data, sample_rate = sf.read(uri, dtype='int16')
        #self.logger.debug(f'sample_rate: {sample_rate}')
        if len(wav_data.shape) > 1:
            wav_data = np.mean(wav_data, axis=1)
        data = wav_data / sample_rate #32768.0
        #print(f'data.shape={data.shape}\n\n')
        #print(f'type(data)={type(data)}\n\n')
        
        return data, sample_rate


class VggishEncoder(Executor):
    
    def __init__(self, model_path: str=f'{cur_dir}/models/vggish_model.ckpt',
                 pca_path: str=f'{cur_dir}/models/vggish_pca_params.npz', *args, **kwargs):
        # INPUT_TENSOR_NAME is defiened in vggish_params.py
        super().__init__(*args, **kwargs)
        tf.compat.v1.reset_default_graph()
        self.model_path = model_path
        self.pca_path = pca_path
        tf.compat.v1.disable_eager_execution()
        self.sess = tf.compat.v1.Session()
        define_vggish_slim()
        load_vggish_slim_checkpoint(self.sess, self.model_path)
        self.feature_tensor = self.sess.graph.get_tensor_by_name(INPUT_TENSOR_NAME)
        self.embedding_tensor = self.sess.graph.get_tensor_by_name(OUTPUT_TENSOR_NAME)
        self.post_processor = Postprocessor(self.pca_path)

    @requests()
    def encode_and_update(self, docs: DocumentArray, *args, **kwargs):
        self._encode_and_update(docs.traverse_flat(traversal_paths='c'))

    def _encode_and_update(self, docs: DocumentArray, *args, **kwargs):
        embedding_matrix = self._encode(docs)
        for d,e in zip(docs, embedding_matrix):
            d.embedding = e

    def _encode(self, docs: DocumentArray, *args, **kwargs):
        blobs = docs.get_attributes('blob')
        [embedding_batch] = self.sess.run([self.embedding_tensor],
                                          feed_dict={self.feature_tensor: blobs})
        result = self.post_processor.postprocess(embedding_batch)
        embedding_matrix = (np.float32(result) - 128.) / 128.
        return embedding_matrix


class Indexer(Executor):

    def __init__(self, index_folder=f'{cur_dir}/workspace/', *args, **kwargs):
        self.index_folder = index_folder
        self.index_path = os.path.join(self.index_folder,'docs.json')
        self._embedding_matrix = None
        self._darray_chunks = None
        self.docid_to_docpos = None

        if os.path.exists(self.index_path):
            self._docs = DocumentArray.load(self.index_path)
            self._darray_chunks = self._docs.traverse_flat(traversal_paths='c')
            self._embedding_matrix = np.stack(self._darray_chunks.get_attributes('embedding')) 
            self.docid_to_docpos = {doc.id: i for i, doc in enumerate(self._docs)}

        else:
            self._docs = DocumentArray() 

    @requests(on='/index')
    def index(self, docs: DocumentArray, *args, **kwargs):
        self._docs.extend(docs)


    def _add_query_matches_from_chunks(self, query: Document, chunks: DocumentArray, *args, **kwargs):
        parent_ids = chunks.get_attributes('parent_id')
        matches = []
        for parent_id in set(parent_ids):
            chunks_with_common_parent = [chunk for chunk in chunks if chunk.parent_id == parent_id]
            min_score = min([chunk.score.value for chunk in chunks_with_common_parent])
            match = Document(self._docs[self.docid_to_docpos[parent_id]], copy=True, score=min_score)
            matches.append(match)


        query.matches.extend(DocumentArray(sorted(matches, key=lambda x: x.score.value)))

    #@requests(on='/search')
    def search(self, docs: DocumentArray, parameters, **kwargs):
        # top_k = parameters['top_k']
        # pairwise euclidean distance
        #d = self._embedding_matrix  # get all embedding from stored docs

        for query in docs:
            q_emb = np.stack(query.chunks.get_attributes('embedding'))  # get all embedding from query docs
            euclidean_dist = np.linalg.norm(q_emb[:, None, :] - self._embedding_matrix[None, :, :], axis=-1)

            for distances, query_chunk in zip(euclidean_dist, query.chunks):  # add & sort match
                for i, distance in enumerate(distances):
                    matching_chunk = Document(self._darray_chunks[int(i)], copy=True, score=distance)
                    query_chunk.matches.append(matching_chunk)
        # M examples database
        # query-> chunk[i].matches -> [d(x.emb, ),]
        # self._add_query_matches_from_chunks(query_chunk, DocumentArray(matching_chunks))
        self._rank(docs)

    def _rank(self, docs, **kwargs):
        """
        For each query in docs, want to get the top k matches

        Group by parent ID: the score from parent query to parent match equals the minimum distance

        q -> qc1,...qc10
        doc1 -> c1,....c20   -> np.argmin(d(c1,qc1),...d(c20,qc1), ...d(c20,qc10))
        docM -> c1,....c40   -> d(c1,qc1),...d(c40,qc1), ...d(c20,qc10)
        """
        for query in docs:

            parent_ids = defaultdict(list)
            #parent_ids = {}
            for chunk in query.chunks:
                for match in chunk.matches:
                    parent_ids[match.parent_id].append(match.score.value)

            for id in parent_ids.keys():
                match = self._docs[self.docid_to_docpos[id]]
                match.score = np.min(parent_ids[id])
                query.matches.append(match)

            query.matches.sort(key=lambda x: x.score.value)


    def close(self):
        os.makedirs(self.index_folder, exist_ok = True)
        self._docs.save(self.index_path)





class DocIndexer(Executor):
    @requests
    def index(self, docs: DocumentArray, *args, **kwargs):
        pass
