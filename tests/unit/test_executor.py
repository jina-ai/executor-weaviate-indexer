import os
import time
import operator

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

    docs = DocumentArray([Document(id=f'r{i}', tags={'price': i}) for i in range(10)])
    n_dim = 3
    indexer = WeaviateIndexer(name='Test', n_dim=n_dim, columns=[('price', 'float')])

    indexer.index(docs)

    max_price = 3
    filter_ = {'path': 'price', 'operator': 'Equal', 'valueNumber': max_price}

    result = indexer.filter(parameters={'filter': filter_})

    assert len(result) == 1
    assert result[0].tags['price'] == max_price

@pytest.mark.parametrize('limit', [1, 2, 3])
def test_search_with_match_args(docs, limit, docker_compose):
    indexer = WeaviateIndexer(name='test', match_args={'limit': limit})
    indexer.index(docs)
    assert 'limit' in indexer._match_args.keys()
    assert indexer._match_args['limit'] == limit

    query = DocumentArray([Document(embedding=np.random.rand(128))])
    indexer.search(query)

    assert len(query[0].matches) == limit

    docs[0].tags['text'] = 'hello'
    docs[1].tags['text'] = 'world'
    docs[2].tags['text'] = 'hello'

    indexer = WeaviateIndexer(
        name='test',
        columns=[('text', 'str')],
        match_args={'filter': {'path': 'text', 'operator': 'Equal', 'valueNumber': 'hello'}, 'limit': 1},
    )
    indexer.index(docs)

    result = indexer.search(query)
    assert len(result) == 1
    assert result[0].tags['text'] == 'hello'


def test_persistence(docs, docker_compose):
    indexer1 = WeaviateIndexer(name='Test', distance='euclidean')
    indexer1.index(docs)
    indexer2 = WeaviateIndexer(name='Test', distance='euclidean')
    assert_document_arrays_equal(indexer2._index, docs)


@pytest.mark.parametrize(
    'metric, metric_name',
    [('euclidean', 'euclid_similarity'), ('cosine', 'cosine_similarity')],
)
def test_search(metric, metric_name, docs, docker_compose):
    # test general/normal case
    indexer = WeaviateIndexer(name='Test', distance=metric)
    indexer.index(docs)
    query = DocumentArray([Document(embedding=np.random.rand(128)) for _ in range(10)])
    indexer.search(query)

    for doc in query:
        similarities = [t[metric_name].value for t in doc.matches[:, 'scores']]
        assert sorted(similarities, reverse=True) == similarities


def test_clear(docs, docker_compose):
    indexer = WeaviateIndexer(name='Test')
    indexer.index(docs)
    assert len(indexer._index) == 6
    indexer.clear()
    assert len(indexer._index) == 0


@pytest.mark.parametrize('type_', ['int', 'float'])
def test_columns(docker_compose, type_):
    n_dim = 3
    indexer = WeaviateIndexer(
        name=f'Test{type_}', n_dim=n_dim, columns=[('price', type_)]
    )

    docs = DocumentArray(
        [
            Document(id=f'r{i}', embedding=i * np.ones(n_dim), tags={'price': i})
            for i in range(10)
        ]
    )
    indexer.index(docs)
    assert len(indexer._index) == 10


numeric_operators_weaviate = {
    'GreaterThanEqual': operator.ge,
    'GreaterThan': operator.gt,
    'LessThanEqual': operator.le,
    'LessThan': operator.lt,
    'Equal': operator.eq,
    'NotEqual': operator.ne,
}


@pytest.mark.parametrize('operator', list(numeric_operators_weaviate.keys()))
def test_filtering(docker_compose, operator: str):
    n_dim = 3
    indexer = WeaviateIndexer(name='Test', n_dim=n_dim, columns=[('price', 'float')])

    docs = DocumentArray(
        [
            Document(id=f'r{i}', embedding=np.random.rand(n_dim), tags={'price': i})
            for i in range(50)
        ]
    )
    indexer.index(docs)

    for threshold in [10, 20, 30]:

        filter_ = {'path': ['price'], 'operator': operator, 'valueNumber': threshold}

        doc_query = DocumentArray([Document(embedding=np.random.rand(n_dim))])
        indexer.search(doc_query, parameters={'filter': filter_})

        assert len(doc_query[0].matches)

        assert all(
            [
                numeric_operators_weaviate[operator](r.tags['price'], threshold)
                for r in doc_query[0].matches
            ]
        )
