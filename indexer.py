import pickle
from typing import Optional, Dict
import base64
import weaviate
import numpy as np
from jina import Executor, requests
from docarray import Document, DocumentArray


def to_base64(doc):
    return base64.b64encode(pickle.dumps(doc)).decode('utf-8')


def from_base64(s):
    return pickle.loads(base64.b64decode(s))


class WeaviateIndexer(Executor):
    def __init__(
        self,
        credentials: Optional[Dict] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ):
        super(WeaviateIndexer, self).__init__(**kwargs)
        self.client = weaviate.Client(**credentials)
        self.batch_size = batch_size
        self.class_obj = {
            'class': 'Document',
            'vectorizer': 'none',
            'properties': [
                {
                    'name': 'serialized',
                    'dataType': ['blob'],
                },
            ],
        }

        self.client.schema.delete_all()
        if not self.client.schema.contains(self.class_obj):
            try:
                self.client.schema.create_class(self.class_obj)
            except weaviate.exceptions.UnexpectedStatusCodeException as e:
                self.client.schema.delete_all()
                self.client.schema.create_class(self.class_obj)

    @requests(on='/index')
    def index(self, docs: DocumentArray, **kwargs):
        with self.client.batch(batch_size=self.batch_size) as batch:
            for doc in docs:
                # TODO use doc.to_base64()
                batch.add_data_object(
                    {'serialized': to_base64(doc)}, 'Document', vector=doc.embedding
                )

    @requests(on='/clear')
    def clear(self, **kwargs):
        self.client.schema.delete_all()
        self.client.schema.create_class(self.class_obj)

    @requests(on='/search')
    def search(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            nearVector = {'vector': doc.embedding.tolist()}
            # TODO use and investigate performance of batch query using 'where'
            response = (
                self.client.query.get(
                    'Document', ['serialized', '_additional {certainty}']
                )
                .with_near_vector(nearVector)
                .do()['data']['Get']['Document']
            )
            # TODO use doc.from_base64()
            matches = DocumentArray(
                [from_base64(doc['serialized']) for doc in response]
            )
            for match, resp in zip(matches, response):
                match.scores['certainty'] = resp['_additional']['certainty']
            doc.matches.extend(matches)
