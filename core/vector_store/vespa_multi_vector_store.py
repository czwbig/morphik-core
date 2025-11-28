import asyncio
import base64
import json
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from vespa.application import Vespa
from vespa.io import VespaResponse, VespaQueryResponse

from core.config import get_settings
from core.models.chunk import DocumentChunk
from core.storage.base_storage import BaseStorage
from core.storage.local_storage import LocalStorage
from core.storage.s3_storage import S3Storage
from core.storage.utils_file_extensions import detect_file_type

from .base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

MULTIVECTOR_CHUNKS_BUCKET = "multivector-chunks"
DEFAULT_APP_ID = "default"


class VespaMultiVectorStore(BaseVectorStore):
    """Vespa implementation for storing and querying multi-vector embeddings."""

    def __init__(
            self,
            vespa_url: str = "http://localhost:8080",
            application: str = "morphik",
            schema: str = "multi_vector",
            max_retries: int = 3,
            retry_delay: float = 1.0,
            enable_external_storage: bool = True,
    ):
        """Initialize Vespa connection for multi-vector storage.

        Args:
            vespa_url: Vespa endpoint URL
            application: Vespa application name
            schema: Document schema name
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
            enable_external_storage: Use external storage for chunks

        Note:
            Call await initialize() after creating an instance to set up the schema.
        """
        self.vespa_url = vespa_url.rstrip("/")
        self.application = application
        self.schema = schema
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_external_storage = enable_external_storage
        self.storage: Optional[BaseStorage] = None
        self._document_app_id_cache: Dict[str, str] = {}
        self.vespa_client: Optional[Vespa] = None
        self._initialized = False

        if enable_external_storage:
            self.storage = self._init_storage()
        self.vespa_client = Vespa(url=self.vespa_url)



    def _init_storage(self) -> BaseStorage:
        """Initialize storage backend based on settings."""
        try:
            settings = get_settings()
            if settings.STORAGE_PROVIDER == "aws-s3":
                logger.info("Initializing S3 storage for multi-vector chunks")
                return S3Storage(
                    aws_access_key=settings.AWS_ACCESS_KEY,
                    aws_secret_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_REGION,
                    default_bucket=MULTIVECTOR_CHUNKS_BUCKET,
                )
            else:
                logger.info("Initializing local storage for multi-vector chunks")
                storage_path = getattr(settings, "LOCAL_STORAGE_PATH", "./storage")
                return LocalStorage(storage_path=storage_path)
        except Exception as e:
            logger.error(f"Failed to initialize external storage: {e}")
            return None

    async def initialize(self):
        """Initialize Vespa connection and verify it's ready.

        This method should be called after creating an instance.
        It's safe to call multiple times - will skip if already initialized.

        Note: The schema should be deployed separately using `vespa deploy vespa-app`.
        """
        if self._initialized:
            logger.debug("VespaMultiVectorStore already initialized, skipping")
            return True

        try:
            # Initialize Vespa client using pyvespa

            # Test connection with a simple query (run in thread pool since Vespa client is sync)
            response = await asyncio.to_thread(
                self.vespa_client.query,
                yql="select * from sources * where true limit 0"
            )

            if response.is_successful():
                logger.info("Vespa connection verified successfully")
                self._initialized = True
                return True
            else:
                logger.warning(f"Vespa health check failed: {response.status_code}")
                # Still mark as initialized to allow usage
                self._initialized = True
                return True
        except Exception as e:
            logger.error(f"Error connecting to Vespa: {e}")
            # Still mark as initialized to allow usage
            self._initialized = True
            return False


    def _prepare_embeddings(self, embeddings: Union[np.ndarray, torch.Tensor, List]) -> List[List[float]]:
        """Convert embeddings to multi-vector format for Vespa (2D list)."""
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        # Convert to numpy array if needed
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)

        # Ensure 2D array
        if isinstance(embeddings, np.ndarray):
            if embeddings.ndim == 1:
                # Single vector - wrap it in a list to make it 2D
                embeddings = embeddings.reshape(1, -1)
            # Return as list of lists (multi-vector format)
            return embeddings.tolist()

        return embeddings


    async def _get_document_app_id(self, document_id: str) -> str:
        """Get app_id for a document with caching."""
        if document_id in self._document_app_id_cache:
            return self._document_app_id_cache[document_id]
        return DEFAULT_APP_ID

    def _determine_file_extension(self, content: str, chunk_metadata: Optional[str]) -> str:
        """Determine file extension based on content and metadata."""
        try:
            if chunk_metadata:
                metadata = json.loads(chunk_metadata)
                is_image = metadata.get("is_image", False)
                if is_image:
                    return detect_file_type(content)
                else:
                    return ".txt"
            else:
                return detect_file_type(content)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Error parsing chunk metadata: {e}")
            return detect_file_type(content)

    def _generate_storage_key(self, app_id: str, document_id: str, chunk_number: int, extension: str) -> str:
        """Generate storage key path."""
        return f"{app_id}/{document_id}/{chunk_number}{extension}"

    async def _store_content_externally(
            self,
            content: str,
            document_id: str,
            chunk_number: int,
            chunk_metadata: Optional[str],
            app_id: Optional[str] = None,
    ) -> Optional[str]:
        """Store chunk content in external storage."""
        if not self.storage:
            return None

        try:
            if app_id is None:
                app_id = await self._get_document_app_id(document_id)

            extension = self._determine_file_extension(content, chunk_metadata)
            storage_key = self._generate_storage_key(app_id, document_id, chunk_number, extension)

            if extension == ".txt":
                content_bytes = content.encode("utf-8")
                content_b64 = base64.b64encode(content_bytes).decode("utf-8")
                await self.storage.upload_from_base64(
                    content=content_b64, key=storage_key, content_type="text/plain", bucket=MULTIVECTOR_CHUNKS_BUCKET
                )
            else:
                await self.storage.upload_from_base64(
                    content=content, key=storage_key, bucket=MULTIVECTOR_CHUNKS_BUCKET
                )

            logger.debug(f"Stored chunk content externally: {storage_key}")
            return storage_key
        except Exception as e:
            logger.error(f"Failed to store content externally: {e}")
            return None

    def _is_storage_key(self, content: str) -> bool:
        """Check if content is a storage key."""
        return (
                len(content) < 500 and "/" in content and not content.startswith("data:") and not content.startswith("http")
        )

    @staticmethod
    def _normalize_storage_key(key: str) -> str:
        """Strip bucket prefix from key."""
        if key.startswith(f"{MULTIVECTOR_CHUNKS_BUCKET}/"):
            return key[len(MULTIVECTOR_CHUNKS_BUCKET) + 1 :]
        return key

    async def _retrieve_content_from_storage(self, storage_key: str, chunk_metadata: Optional[str]) -> str:
        """Retrieve content from external storage."""
        if not self.storage:
            return storage_key

        try:
            if isinstance(self.storage, S3Storage):
                storage_key = f"{MULTIVECTOR_CHUNKS_BUCKET}/{storage_key}"
            try:
                content_bytes = await self.storage.download_file(bucket=MULTIVECTOR_CHUNKS_BUCKET, key=storage_key)
            except Exception:
                storage_key = f"{MULTIVECTOR_CHUNKS_BUCKET}/{storage_key}.txt"
                content_bytes = await self.storage.download_file(bucket=MULTIVECTOR_CHUNKS_BUCKET, key=storage_key)

            if not content_bytes:
                return storage_key

            if chunk_metadata:
                metadata = json.loads(chunk_metadata)
                is_image = metadata.get("is_image", False)
                return content_bytes.decode("utf-8")
                # if is_image:
                #     return base64.b64encode(content_bytes).decode("utf-8")
                # else:
                #     return content_bytes.decode("utf-8")
            else:
                try:
                    return content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    return base64.b64encode(content_bytes).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to retrieve content from storage: {e}")
            return storage_key

    async def store_embeddings(
            self, chunks: List[DocumentChunk], app_id: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """Store document chunks with multi-vector embeddings."""
        stored_ids = []

        for chunk in chunks:
            if not hasattr(chunk, "embedding") or chunk.embedding is None:
                logger.error(f"Missing embeddings for chunk {chunk.document_id}-{chunk.chunk_number}")
                continue

            # Convert to float embeddings (keep multi-vector structure)
            float_embeddings = self._prepare_embeddings(chunk.embedding)

            # Convert to Vespa tensor format using blocks format
            # For tensor<float>(patch{}, x[128]), we use the "blocks" format
            blocks = []
            for patch_idx, patch_vector in enumerate(float_embeddings):
                blocks.append({
                    "address": {"patch": str(patch_idx)},
                    "values": [float(v) for v in patch_vector]
                })

            embeddings_tensor = {"blocks": blocks}

            content_to_store = chunk.content

            # Convert metadata to proper JSON string early
            metadata_json = json.dumps(chunk.metadata) if chunk.metadata else "{}"

            if self.enable_external_storage and self.storage:
                storage_key = await self._store_content_externally(
                    chunk.content, chunk.document_id, chunk.chunk_number, metadata_json, app_id
                )
                if storage_key:
                    content_to_store = storage_key

            doc_id = f"{chunk.document_id}_{chunk.chunk_number}"

            document = {
                "fields": {
                    "document_id": chunk.document_id,
                    "chunk_number": chunk.chunk_number,
                    "content": content_to_store,
                    "chunk_metadata": metadata_json,
                    "embeddings": embeddings_tensor,  # Already formatted as {"blocks": [...]}
                }
            }

            # Debug: log document structure for first chunk only
            if len(stored_ids) == 0:
                logger.debug(f"Storing document with ID: {doc_id}")
                logger.debug(f"Document structure sample: {str(document)[:200]}")

            for attempt in range(self.max_retries):
                try:
                    # Use pyvespa's feed_data_point method (run in thread pool since Vespa client is sync)
                    response: VespaResponse = await asyncio.to_thread(
                        self.vespa_client.feed_data_point,
                        schema=self.schema,
                        data_id=doc_id,
                        fields=document["fields"]
                    )

                    if response.is_successful():
                        stored_ids.append(doc_id)
                        break
                    else:
                        error_msg = f"Failed to store chunk {doc_id}: {response.status_code}"
                        if hasattr(response, 'json'):
                            error_msg += f" - {response.json}"
                        logger.warning(error_msg)
                        if attempt == self.max_retries - 1:
                            logger.error(f"Final attempt failed for {doc_id}")
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                    else:
                        logger.error(f"Error storing chunk {doc_id}: {e}")

        logger.info(f"Stored {len(stored_ids)} embeddings in Vespa")
        return True, stored_ids

    async def query_similar(
            self,
            query_embedding: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
            k: int = 5,
            doc_ids: Optional[List[str]] = None,
            app_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Find similar chunks using Vespa max_sim ranking."""
        # Vespa has a default limit of 400 hits per query
        MAX_HITS = 400
        if k > MAX_HITS:
            logger.warning(f"Requested {k} hits, but Vespa limit is {MAX_HITS}. Limiting to {MAX_HITS}.")
            k = MAX_HITS

        # Convert query embeddings to multi-vector format
        float_query = self._prepare_embeddings(query_embedding)

        # Convert to Vespa tensor format using blocks format
        # For tensor<float>(qpatch{}, x[128])
        blocks = []
        for qpatch_idx, qpatch_vector in enumerate(float_query):
            blocks.append({
                "address": {"qpatch": str(qpatch_idx)},
                "values": [float(v) for v in qpatch_vector]
            })

        query_tensor = {"blocks": blocks}

        yql_query = f"select * from {self.schema} where true"
        if doc_ids:
            # Use matches or contains for string field matching
            # For exact match with string fields, use: document_id matches "value"
            doc_filter = " document_id in ( " + " , ".join([f'"{doc_id}"' for doc_id in doc_ids]) + " ) "
            yql_query = f"select * from {self.schema} where {doc_filter}"

        try:
            # Use pyvespa's query method (run in thread pool since Vespa client is sync)
            logger.debug(f"Querying Vespa... {yql_query[:500]}")
            response: VespaQueryResponse = await asyncio.to_thread(
                self.vespa_client.query,
                body={
                    "input.query(q)": query_tensor,
                    "yql": yql_query,
                    "ranking": "max_sim",
                    "hits": k,
                }
            )
            logger.debug(f"Querying Vespa... done")

            if not response.is_successful():
                logger.error(f"Query failed: {response.status_code}")
                return []

            hits = response.hits

            content_tasks = []
            for hit in hits:
                hit_fields = hit.get("fields", {})
                hit["document_id"] = hit_fields.get("document_id", "")
                hit["chunk_number"] = hit_fields.get("chunk_number", "")
                hit["content"] = hit_fields.get("content", "")
                hit["chunk_metadata"] = hit_fields.get("chunk_metadata", "{}")
                content = hit.get("content", "")
                metadata = hit.get("chunk_metadata", "{}")

                if self.enable_external_storage and self._is_storage_key(content):
                    content_tasks.append(self._retrieve_content_from_storage(content, metadata))
                else:
                    content_tasks.append(asyncio.sleep(0, result=content))

            resolved_contents = await asyncio.gather(*content_tasks, return_exceptions=True)

            chunks = []
            hit_relevance_list = [hit.get("relevance", 0.0) for hit in hits]
            logger.debug("Vespa query hits relevance scores: " + ", ".join([f"{score:.4f}" for score in hit_relevance_list]))
            for hit, resolved in zip(hits, resolved_contents):
                try:
                    metadata = json.loads(hit.get("chunk_metadata", "{}"))
                except Exception:
                    metadata = {}

                content = hit.get("content", "") if isinstance(resolved, Exception) else resolved

                chunk = DocumentChunk(
                    document_id=hit.get("document_id", ""),
                    chunk_number=hit.get("chunk_number", 0),
                    content=content,
                    embedding=[],
                    metadata=metadata,
                    score=hit.get("relevance", 0.0),
                )
                if hit.get("relevance", 0.0) < 8:
                    continue
                chunks.append(chunk)

            return chunks
        except Exception as e:
            logger.error(f"Error querying similar chunks: {e}")
            return []

    async def get_chunks_by_id(
            self,
            chunk_identifiers: List[Tuple[str, int]],
            app_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Retrieve specific chunks by document ID and chunk number."""
        if not chunk_identifiers:
            return []

        chunks = []
        for doc_id, chunk_num in chunk_identifiers:
            vespa_id = f"{doc_id}_{chunk_num}"

            try:
                # Use pyvespa's get_data method (run in thread pool since Vespa client is sync)
                response: VespaResponse = await asyncio.to_thread(
                    self.vespa_client.get_data,
                    schema=self.schema,
                    data_id=vespa_id
                )

                if response.is_successful():
                    fields = response.json.get("fields", {})
                    content = fields.get("content", "")
                    metadata_str = fields.get("chunk_metadata", "{}")

                    if self.enable_external_storage and self._is_storage_key(content):
                        content = await self._retrieve_content_from_storage(content, metadata_str)

                    try:
                        metadata = json.loads(metadata_str)
                    except Exception:
                        metadata = {}

                    chunk = DocumentChunk(
                        document_id=fields.get("document_id", ""),
                        chunk_number=fields.get("chunk_number", 0),
                        content=content,
                        embedding=[],
                        metadata=metadata,
                        score=0.0,
                    )
                    chunks.append(chunk)
            except Exception as e:
                logger.error(f"Error retrieving chunk {vespa_id}: {e}")

        return chunks

    async def delete_chunks_by_document_id(self, document_id: str, app_id: Optional[str] = None) -> bool:
        """Delete all chunks associated with a document."""
        storage_keys: Set[str] = set()

        try:
            # Use matches for string field matching in YQL
            # For string fields with attribute: fast-search, use matches
            yql_query = f'select * from {self.schema} where document_id matches "{document_id}"'

            # Vespa has a default limit of 400 hits, so we need to paginate
            max_hits_per_query = 400
            offset = 0
            total_deleted = 0

            while True:
                # Use pyvespa's query method to find chunks (run in thread pool since Vespa client is sync)
                response: VespaQueryResponse = await asyncio.to_thread(
                    self.vespa_client.query,
                    yql=yql_query,
                    hits=max_hits_per_query,
                    offset=offset
                )

                if not response.is_successful():
                    logger.error(f"Query failed while deleting chunks: {response.status_code}")
                    break

                hits = response.hits

                if not hits:
                    # No more results
                    break

                for hit in hits:
                    content = hit.get("content", "")

                    if self.enable_external_storage and self._is_storage_key(content):
                        storage_keys.add(self._normalize_storage_key(content))

                    # Extract document ID from hit
                    hit_id = hit.get("id", "")
                    if "::" in hit_id:
                        doc_id = hit_id.split("::")[-1]
                    else:
                        doc_id = f"{hit.get('document_id', '')}_{hit.get('chunk_number', 0)}"

                    # Use pyvespa's delete_data method (run in thread pool since Vespa client is sync)
                    try:
                        await asyncio.to_thread(
                            self.vespa_client.delete_data,
                            schema=self.schema,
                            data_id=doc_id
                        )
                        total_deleted += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete document {doc_id}: {e}")

                # If we got fewer results than requested, we've reached the end
                if len(hits) < max_hits_per_query:
                    break

                offset += max_hits_per_query

            if storage_keys and self.storage:
                delete_storage_tasks = [
                    self.storage.delete_file(MULTIVECTOR_CHUNKS_BUCKET, key) for key in storage_keys
                ]
                await asyncio.gather(*delete_storage_tasks, return_exceptions=True)

            logger.info(f"Deleted {total_deleted} chunks for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting chunks for document {document_id}: {e}")
            return False

    async def close(self):
        """Close the Vespa client connection."""
        # pyvespa doesn't require explicit connection closing
        if self.vespa_client:
            logger.info("Vespa client closed")
            self.vespa_client = None

