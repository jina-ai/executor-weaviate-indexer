# WeaviateIndexer

`WeaviateIndexer` indexes Documents into a `DocumentArray`  using `storage='weaviate'`. Underneath, the `DocumentArray`  uses 
 [weaviate](https://github.com/semi-technologies/weaviate) to store and search Documents efficiently. 
The indexer relies on `DocumentArray` as a client for Weaviate, you can read more about the integration here: 
https://docarray.jina.ai/advanced/document-store/weaviate/

## Setup
`WeaviateIndexer` requires a running Weaviate server. Make sure a server is up and running and your indexer is configured 
to use it before starting to index documents. For quick testing, you can run a containerized version locally using 
docker-compose :

```shell
docker-compose -f tests/docker-compose.yml up -d
```

## Usage

#### via Docker image (recommended)

```python
from jina import Flow
from docarray import Document
import numpy as np

f = Flow().add(
    uses='jinahub+docker://WeaviateIndexer',
    uses_with={
        'name': 'MyIndexer',
        'n_dim': 256,
    }
)

with f:
    f.post('/index', inputs=[Document(embedding=np.random.rand(256)) for _ in range(3)])
```

#### via source code

```python
from jina import Flow
from docarray import Document
import numpy as np

f = Flow().add(uses='jinahub://WeaviateIndexer',
    uses_with={
        'name': 'MyIndexer',
        'n_dim': 256,
    }
)

with f:
    f.post('/index', inputs=[Document(embedding=np.random.rand(256)) for _ in range(3)])
```



## CRUD Operations

You can perform CRUD operations (create, read, update and delete) using the respective endpoints:

- `/index`: Add new data to `Weaviate`. 
- `/search`: Query the `Weaviate` index (created in `/index`) with your Documents.
- `/update`: Update Documents in `Weaviate`.
- `/delete`: Delete Documents in `Weaviate`.


## Vector Search

The following example shows how to perform vector search using`f.post(on='/search', inputs=[Document(embedding=np.array([1,1]))])`.


```python
from jina import Flow
from docarray import Document
import numpy as np

f = Flow().add(
         uses='jinahub://WeaviateIndexer',
         uses_with={'n_dim': 2},
     )

with f:
    f.post(
        on='/index',
        inputs=[
            Document(id='a', embedding=np.array([1, 3])),
            Document(id='b', embedding=np.array([1, 1])),
        ],
    )

    docs = f.post(
        on='/search',
        inputs=[Document(embedding=np.array([1, 1]))],
    )

# will print "The ID of the best match of [1,1] is: b"
print('The ID of the best match of [1,1] is: ', docs[0].matches[0].id)
```