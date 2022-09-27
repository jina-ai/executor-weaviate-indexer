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

Note that if you run a `Weaviate` service locally and try to run the `WeaviateIndexer` via `docker`, you 
have to specify `'host': 'host.docker.internal'` instead of `localhost`, otherwise the client will not be 
able to reach the service from within the container.

## Usage

#### via Docker image (recommended)

```python
from jina import Flow
from docarray import Document
import numpy as np

f = Flow().add(
    uses='jinahub+docker://WeaviateIndexer',
    uses_with={
        'name': 'Indexer',
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
        'name': 'Indexer',
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

### Using filtering

To do filtering with the WeaviateIndexer you should first define columns and precise the dimension of your embedding space.
For instance :

```python
from jina import Flow

f = Flow().add(
    uses='jinahub+docker://WeaviateIndexer',
    uses_with={
        'name': 'Test',
        'n_dim': 3,
        'columns': [('price', 'float')],
    },
)

```

Then you can pass a filter as a parameters when searching for document:

```python
from docarray import Document, DocumentArray
import numpy as np

docs = DocumentArray(
    [
        Document(id=f'r{i}', embedding=np.random.rand(3), tags={'price': i})
        for i in range(50)
    ]
)


filter_ = {'path': ['price'], 'operator': 'LessThanEqual', 'valueNumber': 30}

with f:
    f.index(docs)
    doc_query = DocumentArray([Document(embedding=np.random.rand(3))])
    f.search(doc_query, parameters={'filter': filter_})
```

### Search using filter only

You can also search Documents using just a filter query (no vector search involved) using the `/filter` endpoint.
Given the previous setup, you can perform a filter query like so:

```python
filter_ = {'path': ['price'], 'operator': 'LessThanEqual', 'valueInt': 30}
with f:
    f.post('/find', parameters={'filter': filter_})
```

### Limit results

In some cases, you will want to limit the total number of retrieved results. `WeaviateIndexer` uses the `limit` argument 
from the `match` function to set this limit. Note that when using `shards=N`, the `limit=K` is the number of retrieved results for **each shard** and total number of retrieved results is `N*K`. By default, `limits` is set to `20`. For more information about shards, please read [Jina Documentation](https://docs.jina.ai/fundamentals/flow/topology/#partition-data-by-using-shards)

```python
f =  Flow().add(
    uses='jinahub+docker://WeaviateIndexer',
    uses_with={'match_args': {'limit': 10}})
```

### Configure other search behaviors

You can use `match_args` argument to pass arguments to the `match` function as below. The match function will be called
during `/search` endpoint.

```python
f =  Flow().add(
     uses='jinahub+docker://WeaviateIndexer',
     uses_with={
         'match_args': {
             'limit': 5, 
}})
```

- For more details about overriding configurations, please refer to [this page](https://docs.jina.ai/fundamentals/executor/executor-in-flow/#special-executor-attributes).
- You can find more about the `match` function at [here](https://docarray.jina.ai/api/docarray.array.mixins.match/#docarray.array.mixins.match.MatchMixin.match)

### Configure the Search Behaviors on-the-fly

**At search time**, you can also pass arguments to config the `match` function. This can be useful when users want to query with different arguments for different data requests. For instance, the following codes query with a custom `limit` in `parameters` and only retrieve the top 100 nearest neighbors. This will overwrite existing `match_args` if defined when initialized.

```python
with f:
    f.search(
        inputs=Document(text='hello'), 
        parameters={'limit': 100})

For more information please refer to the docarray [documentation](https://docarray.jina.ai/advanced/document-store/weaviate/#vector-search-with-filter)
