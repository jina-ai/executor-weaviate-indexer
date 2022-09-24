from jina import Executor, requests
from typing import Optional, Dict, List, Tuple, Any, Union
from docarray import DocumentArray
from jina.logging.logger import JinaLogger


class WeaviateIndexer(Executor):
    """WeaviateIndexer indexes Documents into a Weaviate server using DocumentArray  with `storage='weaviate'`"""

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 8080,
        protocol: str = 'http',
        name: str = 'Persisted',
        n_dim: int = 128,
        match_args: Optional[Dict] = None,
        ef: Optional[int] = None,
        ef_construction: Optional[int] = None,
        max_connections: Optional[int] = None,
        columns: Optional[Union[List[Tuple[str, str]], Dict[str, str]]] = None,
        **kwargs,
    ):
        """
        :param host: Hostname of the Weaviate server
        :param port: port of the Weaviate server
        :param protocol: protocol to be used. Can be 'http' or 'https'
        :param name: Weaviate class name used for the storage
        :param n_dim: number of dimensions
        :param match_args: the arguments to `DocumentArray`'s match function
        :param ef: The size of the dynamic list for the nearest neighbors (used during the search). The higher ef is
            chosen, the more accurate, but also slower a search becomes. Defaults to the default `ef` in the weaviate
            server.
        :param ef_construction: The size of the dynamic list for the nearest neighbors (used during the construction).
            Controls index search speed/build speed tradeoff. Defaults to the default `ef_construction` in the weaviate
            server.
        :param max_connections: The maximum number of connections per element in all layers. Defaults to the default
            `max_connections` in the weaviate server.
        :param columns: precise columns for the Indexer (used for filtering).
        """
        super().__init__(**kwargs)
        self._match_args = match_args or {}

        self._index = DocumentArray(
            storage='weaviate',
            config={
                'host': host,
                'port': port,
                'protocol': protocol,
                'name': name,
                'n_dim': n_dim,
                'ef': ef,
                'ef_construction': ef_construction,
                'max_connections': max_connections,
                'columns': columns,
            },
        )

        self.logger = JinaLogger(self.metas.name)

    @requests(on='/index')
    def index(
        self,
        docs: 'DocumentArray',
        **kwargs,
    ):
        """Index new documents
        :param docs: the Documents to index
        """
        self._index.extend(docs)

    @requests(on='/search')
    def search(
        self,
        docs: 'DocumentArray',
        parameters: Dict = {},
        **kwargs,
    ):
        """Perform a vector similarity search and retrieve the full Document match

        :param docs: the Documents to search with
        :param parameters: Dictionary to define the `filter` that you want to use.
        :param kwargs: additional kwargs for the endpoint

        """
        match_args = (
            {**self._match_args, **parameters}
            if parameters is not None
            else self._match_args
        )
        docs.match(self._index, **match_args)


    @requests(on='/delete')
    def delete(self, parameters: Dict, **kwargs):
        """Delete entries from the index by id

        :param parameters: parameters of the request

        Keys accepted:
            - 'ids': List of Document IDs to be deleted
        """
        deleted_ids = parameters.get('ids', [])
        if len(deleted_ids) == 0:
            return
        del self._index[deleted_ids]

    @requests(on='/update')
    def update(self, docs: DocumentArray, **kwargs):
        """Update existing documents
        :param docs: the Documents to update
        """

        for doc in docs:
            try:
                self._index[doc.id] = doc
            except IndexError:
                self.logger.warning(
                    f'cannot update doc {doc.id} as it does not exist in storage'
                )

    @requests(on='/filter')
    def filter(self, parameters: Dict, **kwargs):
        """
        Query documents from the indexer by the filter `query` object in parameters. The `query` object must follow the
        specifications in the `find` method of `DocumentArray` using Weaviate: https://docarray.jina.ai/advanced/document-store/weaviate/#example-of-find-with-a-filter-only
        :param parameters: parameters of the request, containing the `filter` object.
        """
        return self._index.find(filter=parameters.get('filter', None))

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: DocumentArray, **kwargs):
        """Fill embedding of Documents by id

        :param docs: DocumentArray to be filled with Embeddings from the index
        """
        for doc in docs:
            doc.embedding = self._index[doc.id].embedding

    @requests(on='/clear')
    def clear(self, **kwargs):
        """Clear the index"""
        self._index.clear()

    def close(self) -> None:
        super().close()
        del self._index
