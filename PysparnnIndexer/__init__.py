from jina.executors.indexers.vector import BaseVectorIndexer


class PysparnnIndexer(BaseVectorIndexer):
    """
    :class:`PysparnnIndexer` Approximate Nearest Neighbor Search for Sparse Data in Python using PySparNN.
    """

    def __init__(self, k_clusters=2, num_indexes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.k_clusters = k_clusters
        self.num_indexes = num_indexes

    def post_init(self):
        self.index = {}
        self.mci = None

    def _build_advanced_index(self):
        keys = []
        indexed_vectors = []
        import pysparnn.cluster_index as ci
        for key, vector in self.index.items():
            keys.append(key)
            indexed_vectors.append(vector)
        
        self.mci = ci.MultiClusterIndex(scipy.sparse.vstack(indexed_vectors), keys)

    def query(self, vectors, top_k, *args, **kwargs):
        """Find the top-k vectors with smallest ``metric`` and return their ids in ascending order.

        :return: a tuple of two ndarrays.
            The first array contains indices, the second array contains distances.
            If `n_vectors = vector.shape[0]` both arrays have shape `n_vectors x top_k`

        :param vectors: the vectors with which to search
        :param args: not used
        :param kwargs: not used
        :param top_k: number of results to return
        :return: tuple of arrays of the form `(indices, distances`
        """

        if not self.mci:
            self._build_advanced_index()

        n_elements = search_features_vec.shape[0]
        index_distance_pairs = self.mci.search(vectors,
                                               k=top_k,
                                               k_clusters=self.k_clusters,
                                               num_indexes=self.num_indexes,
                                               return_distance=True)
        distances = []
        indices = [] 
        for record in index_distance_pairs:
            distances_to_record, indices_to_record = zip(*record)
            distances.append(distances_to_record)
            indices.append(indices_to_record)

        return np.array(indices), np.array(distances)
    
    def add(self, keys, vectors, *args, **kwargs):
        """Add keys and vectors to the indexer.

        :param keys: keys associated to the vectors
        :param vectors: vectors with which to search
        :param args: not used
        :param kwargs: not used

        """
        if self.mci is not None:
            raise Exception(' Not possible query while indexing')
        for key, vector in zip(keys, vectors):
            self.index[key] = vector

    def update(
            self, keys, vectors, *args, **kwargs
    ) -> None:
        """Update the embeddings on the index via document ids (keys).

        :param keys: keys associated to the vectors
        :param vectors: vectors with which to search
        :param args: not used
        :param kwargs: not used
        """

        if self.mci is not None:
            raise Exception(' Not possible query while indexing')
        for key, vector in zip(keys, vectors):
            self.index[key] = vector

    def delete(self, keys, *args, **kwargs) -> None:
        """Delete the embeddings from the index via document ids (keys).

        :param keys: a list of ids
        :param args: not used
        :param kwargs: not used
        """
        if self.mci is not None:
            raise Exception(' Not possible query while indexing')
        for key in keys:
            del self.index[key]