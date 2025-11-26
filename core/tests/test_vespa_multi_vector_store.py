"""Test Vespa Multi-Vector Store implementation."""

import asyncio
import numpy as np
import pytest
from core.models.chunk import DocumentChunk
from core.vector_store.vespa_multi_vector_store import VespaMultiVectorStore
from core.vector_store.multi_vector_store import MultiVectorStore


@pytest.fixture
async def vespa_store():
    """Create a Vespa store instance for testing."""
    store = VespaMultiVectorStore(
        vespa_url="http://localhost:8080",
        application="morphik_test",
        schema="multi_vector_test",
        enable_external_storage=True
    )
    await store.initialize()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_store_and_query(vespa_store):
    """Test storing and querying embeddings."""
    # Create test chunks with multi-vector embeddings (5 patches, 128 dims each)
    chunks = [
        DocumentChunk(
            document_id="test_doc_1",
            chunk_number=0,
            content="This is the first test chunk",
            embedding=np.random.randn(5, 128).tolist(),  # Multi-vector: [5 patches, 128 dims]
            metadata={"is_image": False, "test": True}
        ),
        DocumentChunk(
            document_id="test_doc_1",
            chunk_number=1,
            content="This is the second test chunk",
            embedding=np.random.randn(5, 128).tolist(),  # Multi-vector: [5 patches, 128 dims]
            metadata={"is_image": False, "test": True}
        ),
    ]

    # Store embeddings
    success, stored_ids = await vespa_store.store_embeddings(chunks)
    assert success is True
    assert len(stored_ids) == 2

    # Query similar chunks with multi-vector query (3 patches, 128 dims each)
    query_embedding = np.random.randn(3, 128)
    results = await vespa_store.query_similar(query_embedding, k=2)

    assert len(results) <= 2
    for chunk in results:
        assert chunk.document_id == "test_doc_1"
        assert chunk.chunk_number in [0, 1]
        assert chunk.score >= 0


@pytest.mark.asyncio
async def test_get_chunks_by_id(vespa_store):
    """Test retrieving chunks by ID."""
    # Create and store test chunk with multi-vector embeddings
    chunk = DocumentChunk(
        document_id="test_doc_2",
        chunk_number=0,
        content="Test chunk for ID retrieval",
        embedding=np.random.randn(5, 128).tolist(),  # Multi-vector: [5 patches, 128 dims]
        metadata={"is_image": False}
    )

    await vespa_store.store_embeddings([chunk])

    # Retrieve by ID
    results = await vespa_store.get_chunks_by_id([("test_doc_2", 0)])

    assert len(results) == 1
    assert results[0].document_id == "test_doc_2"
    assert results[0].chunk_number == 0
    assert results[0].content == "Test chunk for ID retrieval"


@pytest.mark.asyncio
async def test_delete_chunks(vespa_store):
    """Test deleting chunks by document ID."""
    # Create and store test chunks with multi-vector embeddings
    chunks = [
        DocumentChunk(
            document_id="test_doc_3",
            chunk_number=i,
            content=f"Test chunk {i}",
            embedding=np.random.randn(5, 128).tolist(),  # Multi-vector: [5 patches, 128 dims]
            metadata={"is_image": False}
        )
        for i in range(3)
    ]

    await vespa_store.store_embeddings(chunks)

    # Delete chunks
    success = await vespa_store.delete_chunks_by_document_id("test_doc_3")
    assert success is True

    # Verify deletion
    results = await vespa_store.get_chunks_by_id([("test_doc_3", 0)])
    assert len(results) == 0


@pytest.mark.skip(reason="Binary quantization removed from VespaMultiVectorStore")
@pytest.mark.asyncio
async def test_binary_quantization():
    """Test binary quantization function."""
    # This test is skipped as _binary_quantize method no longer exists
    pass


if __name__ == "__main__":
    # Run a simple manual test
    async def manual_test():
        print("Running manual Vespa test...")
        embedding_list = [np.random.randn(3, 128) for _ in range(3)]

        store = VespaMultiVectorStore(
            vespa_url="http://localhost:8080",
            enable_external_storage=True
        )

        storepg = MultiVectorStore(
            uri="postgresql+asyncpg://morphik:morphik@localhost:5433/morphik",
            enable_external_storage=True
        )
        results = await storepg.query_similar(embedding_list[0], k=2,)

        try:
            # Initialize the store
            await store.initialize()

            # Create multi-vector embeddings (3 patches, 128 dims each)
            results = await store.query_similar(embedding_list[0], k=2, doc_ids=['296af5e9-07be-4d9a-883f-2cb67abf4e47'])

            # Create test data with multi-vector embeddings
            chunks = [
                DocumentChunk(
                    document_id="manual_test",
                    chunk_number=i,
                    content=f"Manual test chunk {i}",
                    embedding=embedding_list[i].tolist(),  # Multi-vector: [3 patches, 128 dims]
                    metadata={"test": True}
                )
                for i in range(3)
            ]

            # Store
            print("Storing chunks...")
            success, ids = await store.store_embeddings(chunks)
            print(f"Stored: {success}, IDs: {ids}")

            # Query with multi-vector query (3 patches, 128 dims)
            print("\nQuerying similar chunks...")
            query = embedding_list[0]  # Use first embedding as query
            results = await store.query_similar(query, k=2)
            print(f"Found {len(results)} results")
            for r in results:
                print(f"  - {r.document_id}-{r.chunk_number}: score={r.score:.4f}")

            # Retrieve
            print("\nRetrieving by ID...")
            specific = await store.get_chunks_by_id([("manual_test", 0)])
            print(f"Retrieved {len(specific)} chunks")

            # Delete
            print("\nDeleting chunks...")
            deleted = await store.delete_chunks_by_document_id("manual_test")
            print(f"Deleted: {deleted}")

            print("\n✓ Manual test completed successfully!")

        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await store.close()

    asyncio.run(manual_test())

