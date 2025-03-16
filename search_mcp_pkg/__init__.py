# search_mcp_pkg/__init__.py
from .core import (
    search,
    search_products_by_category,
    search_products_by_brand,
    index_product,
    create_ecommerce_test_index,
    create_test_index,
    DEFAULT_INDEX,
    es,
    mcp,
)

# Package metadata
__version__ = "0.1.0"
__author__ = "Your Name"
__description__ = "E-commerce search with Elasticsearch and LLM query planning"
