import os
import time

import numpy as np
import pytest
from docarray import Document, DocumentArray
from docarray.array.weaviate import DocumentArrayWeaviate

from executor import WeaviateIndexer

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(cur_dir, '../docker-compose.yml'))


def assert_document_arrays_equal(arr1, arr2):
    assert len(arr1) == len(arr2)
    for d1, d2 in zip(arr1, arr2):
        assert d1.id == d2.id
        assert d1.content == d2.content
        assert d1.chunks == d2.chunks
        assert d1.matches == d2.matches


@pytest.fixture
def docs():
    return DocumentArray(
        [
            Document(id='doc1', embedding=np.random.rand(128)),
            Document(id='doc2', embedding=np.random.rand(128)),
            Document(id='doc3', embedding=np.random.rand(128)),
            Document(id='doc4', embedding=np.random.rand(128)),
            Document(id='doc5', embedding=np.random.rand(128)),
            Document(id='doc6', embedding=np.random.rand(128)),
        ]
    )


@pytest.fixture
def update_docs():
    return DocumentArray(
        [
            Document(id='doc1', text='modified', embedding=np.random.rand(128)),
        ]
    )


@pytest.fixture(scope='function')
def docker_compose():
    os.system(
        f"docker-compose -f {compose_yml} --project-directory . up  --build -d --remove-orphans"
    )
    time.sleep(5)
    yield
    os.system(
        f"docker-compose -f {compose_yml} --project-directory . down --remove-orphans"
    )


def test_init(docker_compose):
    indexer = WeaviateIndexer(name='Test')

    assert isinstance(indexer._index, DocumentArrayWeaviate)
    assert indexer._index.name == 'Test'
    assert indexer._index._config.port == 8080


def test_index(docs, docker_compose):
    indexer = WeaviateIndexer(name='Test')
    indexer.index(docs)

    assert len(indexer._index) == len(docs)


def test_delete(docs, docker_compose):
    indexer = WeaviateIndexer(name='Test')
    indexer.index(docs)

    ids = ['doc1', 'doc2', 'doc3']
    indexer.delete({'ids': ids})
    assert len(indexer._index) == len(docs) - 3
    for doc_id in ids:
        assert doc_id not in indexer._index


def test_update(docs, update_docs, docker_compose):
    # index docs first
    indexer = WeaviateIndexer(name='Test')
    indexer.index(docs)
    assert_document_arrays_equal(indexer._index, docs)

    # update first doc
    indexer.update(update_docs)
    assert indexer._index[0].id == 'doc1'
    assert indexer._index['doc1'].text == 'modified'


def test_fill_embeddings(docker_compose):
    indexer = WeaviateIndexer(name='Test', distance='euclidean', n_dim=1)

    indexer.index(DocumentArray([Document(id='a', embedding=np.array([1.0]))]))
    search_docs = DocumentArray([Document(id='a')])
    indexer.fill_embedding(search_docs)
    assert search_docs['a'].embedding is not None
    assert (search_docs['a'].embedding == np.array([1.0])).all()

    with pytest.raises(KeyError, match='b'):
        indexer.fill_embedding(DocumentArray([Document(id='b')]))


def test_filter(docker_compose):
    docs = DocumentArray.empty(5)
    docs[0].text = 'hello'
    docs[1].text = 'world'
    docs[2].tags['x'] = 0.3
    docs[2].tags['y'] = 0.6
    docs[3].tags['x'] = 0.8

    indexer = WeaviateIndexer(name='Test')
    indexer.index(docs)

    result = indexer.filter(parameters={'query': {'text': {'$eq': 'hello'}}})
    assert len(result) == 1
    assert result[0].text == 'hello'

    result = docs.find({'tags__x': {'$gte': 0.5}})
    assert len(result) == 1
    assert result[0].tags['x'] == 0.8


def test_persistence(docs, docker_compose):
    indexer1 = WeaviateIndexer(name='Test', distance='euclidean')
    indexer1.index(docs)
    indexer2 = WeaviateIndexer(name='Test', distance='euclidean')
    assert_document_arrays_equal(indexer2._index, docs)


@pytest.mark.parametrize('metric, metric_name', [('euclidean', 'euclid_similarity'), ('cosine', 'cosine_similarity')])
def test_search(metric, metric_name, docs, docker_compose):
    # test general/normal case
    indexer = WeaviateIndexer(name='Test', distance=metric)
    indexer.index(docs)
    query = DocumentArray([Document(embedding=np.random.rand(128)) for _ in range(10)])
    indexer.search(query)

    for doc in query:
        similarities = [
            t[metric_name].value for t in doc.matches[:, 'scores']
        ]
        assert sorted(similarities, reverse=True) == similarities
