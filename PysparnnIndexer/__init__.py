from jina.executors.indexers.vector import BaseVectorIndexer




class PysparnnIndexer(BaseIndexer):
    """
    :class:`PysparnnIndexer` Approximate Nearest Neighbor Search for Sparse Data in Python using PySparNN.


    """

    def __init__(self, k_clusters=2, num_indexes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.k_clusters = k_clusters 
        self.num_indexes = num_indexes

    def post_init(self):
        self.mci = None

    def query(self, vectors, top_k ,*args, **kwargs):

        # CHECK WHAT HAPPENS if mci is None
        indices, dist = self.mci.search(vectors,
                                        k=top_k,
                                        k_clusters=self.k_clusters, 
                                        num_indexes=self.num_indexes, 
                                        return_distance=True)
        return indices, dist

    def add(self, keys, vectors, *args, **kwargs):
        
        import pysparnn.cluster_index as ci        
        assert len(vectors) == len(data), "features_vec and data lengths should match"
        
        if self.mci:
            for vector, key in zip(vectors, keys):
                self.mci.insert(vector, key)
        else:
            self.mci = ci.MultiClusterIndex(vectors, keys)

    
