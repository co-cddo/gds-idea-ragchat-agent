import logging

import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth

logger = logging.getLogger(__name__)


class OpenSearchHandler:
    def __init__(self, config):
        logger.info("Initializing OpenSearchHandler")
        self.url = config.COLLECTION_URL
        self.index_name = config.INDEX_NAME
        self.embedding_model = config.EMBEDDING_MODEL
        self.region = config.DATA_REGION
        logger.debug(f"OpenSearch URL: {self.url}")
        logger.debug(f"Index Name: {self.index_name}")
        logger.debug(f"Embedding Model: {self.embedding_model}")
        logger.debug(f"Data Region: {self.region}")

        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            self.awsauth = AWS4Auth(
                credentials.access_key,
                credentials.secret_key,
                self.region,
                "aoss",
                session_token=credentials.token,
            )
            logger.info("Successfully created AWS authentication")
        except Exception as e:
            logger.error(f"Error creating AWS authentication: {str(e)}")
            raise

    def get_retriever(self, k=10):
        logger.info(f"Creating retriever with k={k}")

        try:
            embeddings = BedrockEmbeddings(
                model_id=self.embedding_model, region_name=self.region
            )
            logger.debug("Created Bedrock embeddings")
            docsearch = OpenSearchVectorSearch(
                opensearch_url=self.url,
                index_name=self.index_name,
                embedding_function=embeddings,
                http_auth=self.awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
            )
            logger.debug("Created OpenSearchVectorSearch instance")
            retriever = docsearch.as_retriever(search_kwargs={"k": k})
            logger.info("Successfully created retriever")
            return retriever
        except Exception as e:
            logger.error(f"Error creating retriever: {str(e)}")
            raise
