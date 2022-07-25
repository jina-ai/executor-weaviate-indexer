from jina import Flow
from docarray import Document
import numpy as np

from executor import WeaviateIndexer


def test_replicas(docker_compose):
    n_dim = 10

    f = Flow().add(
        uses=WeaviateIndexer,
        uses_with={'name': 'Testone', 'n_dim': n_dim, 'ef': 10000},
    )

    docs_index = [Document(embedding=np.random.random(n_dim)) for _ in range(1000)]

    docs_query = docs_index[:100]

    with f:
        f.post(on='/index', inputs=docs_index, request_size=1)

        docs_without_replicas = sorted(
            f.post(on='/search', inputs=docs_query, request_size=1),
            key=lambda doc: doc.id,
        )

    f_with_replicas = Flow().add(
        uses=WeaviateIndexer,
        uses_with={'name': 'Testone', 'n_dim': n_dim},
        replicas=4,
    )

    with f_with_replicas:
        docs_with_replicas = sorted(
            f_with_replicas.post(on='/search', inputs=docs_query, request_size=1),
            key=lambda doc: doc.id,
        )

    assert docs_without_replicas == docs_with_replicas


def test_replicas_reindex(docker_compose):
    n_dim = 10

    f = Flow().add(
        uses=WeaviateIndexer,
        uses_with={'name': 'Testtwo', 'n_dim': n_dim, 'ef': 10000},
    )

    docs_index = [Document(id=f'd{i}', embedding=np.random.random(n_dim)) for i in range(1000)]

    docs_query = docs_index[:100]

    with f:
        f.post(on='/index', inputs=docs_index, request_size=1)

        docs_without_replicas = sorted(
            f.post(on='/search', inputs=docs_query, request_size=1),
            key=lambda doc: doc.id,
        )

    f_with_replicas = Flow().add(
        uses=WeaviateIndexer,
        uses_with={'name': 'Testtwo', 'n_dim': n_dim},
        replicas=4,
    )

    with f_with_replicas:
        f_with_replicas.post(on='/index', inputs=docs_index, request_size=1)

        docs_with_replicas = sorted(
            f_with_replicas.post(on='/search', inputs=docs_query, request_size=1),
            key=lambda doc: doc.id,
        )

    assert docs_without_replicas == docs_with_replicas
