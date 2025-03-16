#!/usr/bin/env python3
"""
Search MCP - A Machine Conversation Protocol server for keyword search
with Elasticsearch and LLM query planning.
"""

import os
import json
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("search")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Elasticsearch client
es_host = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
es_user = os.getenv("ELASTICSEARCH_USER", "")
es_pass = os.getenv("ELASTICSEARCH_PASSWORD", "")

es = Elasticsearch(
    es_host,
    basic_auth=(es_user, es_pass) if es_user and es_pass else None,
    verify_certs=False,
)

# Default index name
DEFAULT_INDEX = os.getenv("ELASTICSEARCH_INDEX", "ecommerce")


def get_index_schema(index: str) -> Dict[str, Any]:
    """
    Get the schema (mappings) for the specified Elasticsearch index.

    Args:
        index: The name of the index

    Returns:
        A dictionary containing the index schema/mappings or an empty dict if not found
    """
    try:
        if es.indices.exists(index=index):
            index_info = es.indices.get(index=index)
            if index in index_info and "mappings" in index_info[index]:
                return index_info[index]["mappings"]

        # If we can't get the schema, return a default schema based on the ecommerce index
        return {
            "properties": {
                "product_name": {"type": "text"},
                "description": {"type": "text"},
                "price": {"type": "float"},
                "brand": {"type": "keyword"},
                "category": {"type": "keyword"},
                "rating": {"type": "float"},
                "in_stock": {"type": "boolean"},
                "tags": {"type": "keyword"},
            }
        }
    except Exception as e:
        print(f"Error getting index schema: {e}")
        return {}


def generate_query_plan(query: str, index: str = DEFAULT_INDEX) -> Dict[str, Any]:
    """
    Use LLM to generate a query plan for the given search query.

    Args:
        query: The user's search query
        index: The Elasticsearch index to search

    Returns:
        A dictionary containing the query plan
    """
    # Get the index schema
    schema = get_index_schema(index)

    # Format the schema information for the prompt
    schema_info = json.dumps(schema, indent=2)

    # Get available categories and brands if the index exists
    available_categories = []
    available_brands = []
    common_tags = []

    try:
        if es.indices.exists(index=index):
            # Get distinct categories
            category_query = {
                "size": 0,
                "aggs": {"categories": {"terms": {"field": "category", "size": 50}}},
            }
            category_response = es.search(index=index, body=category_query)
            available_categories = [
                bucket["key"]
                for bucket in category_response["aggregations"]["categories"]["buckets"]
            ]

            # Get distinct brands
            brand_query = {
                "size": 0,
                "aggs": {"brands": {"terms": {"field": "brand", "size": 50}}},
            }
            brand_response = es.search(index=index, body=brand_query)
            available_brands = [
                bucket["key"]
                for bucket in brand_response["aggregations"]["brands"]["buckets"]
            ]

            # Get common tags
            tags_query = {
                "size": 0,
                "aggs": {"tags": {"terms": {"field": "tags", "size": 50}}},
            }
            tags_response = es.search(index=index, body=tags_query)
            common_tags = [
                bucket["key"]
                for bucket in tags_response["aggregations"]["tags"]["buckets"]
            ]
    except Exception as e:
        print(f"Error getting available values: {e}")

    # Add information about available values to the prompt
    available_values = {
        "categories": available_categories,
        "brands": available_brands,
        "common_tags": common_tags,
    }
    available_values_info = json.dumps(available_values, indent=2)

    prompt = f"""
You are a search query planner for an e-commerce platform. Given a user's search query, determine the best search strategy.

Here is the schema of the index you're searching:
{schema_info}

Here are the available values in the data:
{available_values_info}

IMPORTANT: When filtering by categories, brands, or tags, ONLY use values from the lists provided above.

Analyze the query and provide a JSON response with the following fields:
- should_expand: boolean indicating if query expansion would be beneficial
- expanded_query: if should_expand is true, provide an expanded version of the query
- ranking_algorithm: recommend one of ["bm25", "vector_similarity", "hybrid"]
- filters: any filters that should be applied based on the query, including:
  - price_range: optional object with min and max price if mentioned
  - categories: optional array of product categories (MUST be from the available categories list)
  - brands: optional array of brand names (MUST be from the available brands list)
  - ratings: optional minimum rating (1-5)
  - in_stock: optional boolean for availability
  - tags: optional array of tags to filter by (MUST be from the common_tags list)
- search_fields: array of fields to prioritize in search (e.g., ["product_name", "description", "brand"])
- sort_by: optional field to sort results by (e.g., "price.asc", "rating.desc", "relevance")
- explanation: brief explanation of your recommendations

Use the schema information to ensure that:
1. You only reference fields that actually exist in the index
2. You use the correct field types (text, keyword, numeric) for filtering and sorting
3. You optimize the search strategy based on the available fields and their types
4. You ONLY use category, brand, and tag values from the provided lists

User query: {query}

Respond with a valid JSON object only.
"""

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    plan_text = response.choices[0].message.content.strip()

    # Extract JSON from the response
    try:
        # Try to parse the entire response as JSON
        plan = json.loads(plan_text)
    except json.JSONDecodeError:
        # If that fails, try to extract JSON from the text
        import re

        json_match = re.search(r"```json\n(.*?)\n```", plan_text, re.DOTALL)
        if json_match:
            try:
                plan = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                plan = {
                    "should_expand": False,
                    "expanded_query": query,
                    "ranking_algorithm": "bm25",
                    "filters": {},
                    "search_fields": [
                        "product_name",
                        "description",
                        "brand",
                        "category",
                    ],
                    "sort_by": "relevance",
                    "explanation": "Failed to parse LLM response, using default settings.",
                }
        else:
            plan = {
                "should_expand": False,
                "expanded_query": query,
                "ranking_algorithm": "bm25",
                "filters": {},
                "search_fields": ["product_name", "description", "brand", "category"],
                "sort_by": "relevance",
                "explanation": "Failed to parse LLM response, using default settings.",
            }

    return plan


def execute_search(
    query: str, index: str, plan: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Execute a search query based on the query plan.

    Args:
        query: The original search query
        index: The Elasticsearch index to search
        plan: The query plan generated by the LLM

    Returns:
        A list of search results
    """
    search_query = plan["expanded_query"] if plan["should_expand"] else query
    search_fields = plan.get(
        "search_fields", ["product_name", "description", "brand", "category"]
    )

    # Build the Elasticsearch query
    if plan["ranking_algorithm"] == "bm25":
        es_query = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {field: search_query}} for field in search_fields
                    ]
                }
            }
        }
    elif plan["ranking_algorithm"] == "vector_similarity":
        # This would require vector embeddings in Elasticsearch
        # For simplicity, we'll fall back to BM25 with field boosting
        es_query = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "product_name": {"query": search_query, "boost": 3}
                            }
                        },
                        {"match": {"description": {"query": search_query, "boost": 1}}},
                        {"match": {"brand": {"query": search_query, "boost": 2}}},
                        {"match": {"category": {"query": search_query, "boost": 2}}},
                    ]
                }
            }
        }
    else:  # hybrid
        es_query = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {field: search_query}} for field in search_fields
                    ]
                }
            }
        }

    # Add filters if specified
    if plan.get("filters"):
        # Process price range filter
        if "price_range" in plan["filters"]:
            price_range = plan["filters"]["price_range"]
            range_filter = {}

            if "min" in price_range:
                range_filter["gte"] = price_range["min"]
            if "max" in price_range:
                range_filter["lte"] = price_range["max"]

            if range_filter:
                es_query["query"]["bool"].setdefault("filter", []).append(
                    {"range": {"price": range_filter}}
                )

        # Process categories filter
        if "categories" in plan["filters"]:
            categories = plan["filters"]["categories"]
            if categories:
                es_query["query"]["bool"].setdefault("filter", []).append(
                    {"terms": {"category": categories}}
                )

        # Process brands filter
        if "brands" in plan["filters"]:
            brands = plan["filters"]["brands"]
            if brands:
                es_query["query"]["bool"].setdefault("filter", []).append(
                    {"terms": {"brand": brands}}
                )

        # Process ratings filter
        if "ratings" in plan["filters"]:
            min_rating = plan["filters"]["ratings"]
            es_query["query"]["bool"].setdefault("filter", []).append(
                {"range": {"rating": {"gte": min_rating}}}
            )

        # Process in_stock filter
        if "in_stock" in plan["filters"]:
            in_stock = plan["filters"]["in_stock"]
            if in_stock:
                es_query["query"]["bool"].setdefault("filter", []).append(
                    {"term": {"in_stock": True}}
                )

        # Process any other filters
        for field, value in plan["filters"].items():
            if field not in [
                "price_range",
                "categories",
                "brands",
                "ratings",
                "in_stock",
            ]:
                if isinstance(value, list):
                    es_query["query"]["bool"].setdefault("filter", []).append(
                        {"terms": {field: value}}
                    )
                else:
                    es_query["query"]["bool"].setdefault("filter", []).append(
                        {"term": {field: value}}
                    )

    # Add sorting if specified
    if plan.get("sort_by") and plan["sort_by"] != "relevance":
        sort_field, sort_order = plan["sort_by"].split(".")
        es_query["sort"] = [{sort_field: {"order": sort_order}}]

    # Execute the search
    try:
        response = es.search(index=index, body=es_query, size=10)
        results = [
            {**hit["_source"], "score": hit["_score"]}
            for hit in response["hits"]["hits"]
        ]
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []


@mcp.tool()
def index_product(
    product_name: str,
    description: str,
    price: float,
    brand: str = "",
    category: str = "",
    rating: float = 0.0,
    in_stock: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    index: str = DEFAULT_INDEX,
) -> str:
    """
    Index a product in Elasticsearch.

    Args:
        product_name: The name of the product
        description: The product description
        price: The product price
        brand: The product brand
        category: The product category
        rating: The product rating (0-5)
        in_stock: Whether the product is in stock
        metadata: Optional additional metadata for the product
        index: The Elasticsearch index to use (defaults to environment variable)

    Returns:
        A message indicating success or failure
    """
    if metadata is None:
        metadata = {}

    document = {
        "product_name": product_name,
        "description": description,
        "price": price,
        "brand": brand,
        "category": category,
        "rating": rating,
        "in_stock": in_stock,
        **metadata,
    }

    # Create the index if it doesn't exist
    if not es.indices.exists(index=index):
        es.indices.create(index=index)

    try:
        response = es.index(index=index, document=document)
        return f"Product indexed successfully with ID: {response['_id']}"
    except Exception as e:
        return f"Failed to index product: {str(e)}"


@mcp.tool()
def search(query: str, index: str = DEFAULT_INDEX) -> str:
    """
    Search for products matching a query with LLM-powered query planning.

    Args:
        query: The search query (supports natural language queries like "red shoes under $50")
        index: The Elasticsearch index to search (defaults to environment variable)

    Returns:
        Formatted search results with query plan explanation
    """
    # Special command to delete the index (used by demo scripts)
    if query == "DELETE_INDEX":
        try:
            if es.indices.exists(index=index):
                es.indices.delete(index=index)
                return f"Successfully deleted index '{index}'."
            else:
                return f"Index '{index}' does not exist."
        except Exception as e:
            return f"Error deleting index '{index}': {str(e)}"

    # Generate query plan using LLM
    plan = generate_query_plan(query, index)

    # Execute the search based on the plan
    results = execute_search(query, index, plan)

    # Format the results
    if not results:
        return f"No products found for query: {query}\n\nQuery plan: {json.dumps(plan, indent=2)}"

    formatted_results = "\n\n".join(
        [
            f"Product {i+1}:\n"
            f"Name: {result.get('product_name', 'Unnamed product')}\n"
            f"Brand: {result.get('brand', 'N/A')}\n"
            f"Price: ${result.get('price', 'N/A')}\n"
            f"Rating: {result.get('rating', 'N/A')}/5\n"
            f"In Stock: {'Yes' if result.get('in_stock', False) else 'No'}\n"
            f"Category: {result.get('category', 'N/A')}\n"
            f"Description: {result.get('description', 'No description')[:150]}..."
            for i, result in enumerate(results)
        ]
    )

    return f"""Search results for: {query}

Query plan:
{json.dumps(plan, indent=2)}

Results:
{formatted_results}
"""


@mcp.tool()
def create_test_index(num_documents: int = 10, index: str = "test_documents") -> str:
    """
    Create a test index with sample documents for demonstration purposes.

    Args:
        num_documents: Number of test documents to create
        index: The index name to use

    Returns:
        A message indicating success or failure
    """
    # Sample documents for testing
    sample_docs = [
        {
            "content": "Elasticsearch is a distributed, RESTful search and analytics engine.",
            "category": "technology",
            "tags": ["search", "database", "analytics"],
        },
        {
            "content": "Python is a programming language that lets you work quickly and integrate systems effectively.",
            "category": "technology",
            "tags": ["programming", "language", "development"],
        },
        {
            "content": "Machine learning is a method of data analysis that automates analytical model building.",
            "category": "technology",
            "tags": ["ai", "data science", "algorithms"],
        },
        {
            "content": "Climate change is a long-term change in the average weather patterns.",
            "category": "environment",
            "tags": ["climate", "global warming", "science"],
        },
        {
            "content": "Renewable energy is energy that is collected from renewable resources.",
            "category": "environment",
            "tags": ["energy", "sustainability", "solar"],
        },
        {
            "content": "The Great Barrier Reef is the world's largest coral reef system.",
            "category": "environment",
            "tags": ["ocean", "coral", "australia"],
        },
        {
            "content": "The COVID-19 pandemic is a global pandemic of coronavirus disease 2019.",
            "category": "health",
            "tags": ["virus", "pandemic", "medicine"],
        },
        {
            "content": "Exercise is any bodily activity that enhances or maintains physical fitness.",
            "category": "health",
            "tags": ["fitness", "wellness", "activity"],
        },
        {
            "content": "Nutrition is the science that interprets the nutrients and other substances in food.",
            "category": "health",
            "tags": ["food", "diet", "wellness"],
        },
        {
            "content": "Artificial intelligence is intelligence demonstrated by machines.",
            "category": "technology",
            "tags": ["ai", "machine learning", "computer science"],
        },
    ]

    # Create more documents if needed
    while len(sample_docs) < num_documents:
        sample_docs.append(
            {
                "content": f"This is a test document number {len(sample_docs) + 1}.",
                "category": "test",
                "tags": ["test", "sample"],
            }
        )

    # Delete the index if it exists
    if es.indices.exists(index=index):
        es.indices.delete(index=index)

    # Create the index
    es.indices.create(index=index)

    # Index the documents
    for doc in sample_docs:
        es.index(index=index, document=doc)

    # Refresh the index to make the documents searchable immediately
    es.indices.refresh(index=index)

    return f"Created test index '{index}' with {len(sample_docs)} documents"


@mcp.tool()
def create_ecommerce_test_index(
    num_products: int = 20, index: str = "ecommerce"
) -> str:
    """
    Create a test e-commerce index with sample products for demonstration purposes.

    Args:
        num_products: Number of test products to create
        index: The index name to use

    Returns:
        A message indicating success or failure
    """
    # Sample products for testing
    sample_products = [
        {
            "product_name": "Premium Wireless Headphones",
            "description": "High-quality wireless headphones with noise cancellation and 20-hour battery life.",
            "price": 199.99,
            "brand": "SoundMaster",
            "category": "Electronics",
            "rating": 4.7,
            "in_stock": True,
            "tags": ["wireless", "headphones", "audio", "bluetooth"],
        },
        {
            "product_name": "Commuter Wireless Headphones",
            "description": "Lightweight wireless headphones with noise cancellation perfect for daily commute. Foldable design with 15-hour battery life.",
            "price": 149.99,
            "brand": "SoundMaster",
            "category": "Electronics",
            "rating": 4.6,
            "in_stock": True,
            "tags": [
                "wireless",
                "headphones",
                "audio",
                "commute",
                "noise cancellation",
                "travel",
            ],
        },
        {
            "product_name": "Budget Noise Cancelling Earbuds",
            "description": "Affordable wireless earbuds with basic noise cancellation, perfect for commuting and workouts. 8-hour battery life.",
            "price": 89.99,
            "brand": "AudioBasics",
            "category": "Electronics",
            "rating": 4.3,
            "in_stock": True,
            "tags": ["wireless", "earbuds", "audio", "commute", "noise cancellation"],
        },
        {
            "product_name": "TravelQuiet Pro Headphones",
            "description": "Premium noise cancellation headphones designed for commuters and travelers. Blocks out subway and traffic noise with industry-leading technology.",
            "price": 179.99,
            "brand": "AudioPro",
            "category": "Electronics",
            "rating": 4.8,
            "in_stock": True,
            "tags": [
                "wireless",
                "headphones",
                "audio",
                "commute",
                "noise cancellation",
                "premium",
                "travel",
            ],
        },
        {
            "product_name": "CommuterFit Wireless Earbuds",
            "description": "Ergonomic wireless earbuds with active noise cancellation technology. Perfect for daily commutes with secure fit and sweat resistance.",
            "price": 129.99,
            "brand": "FitAudio",
            "category": "Electronics",
            "rating": 4.5,
            "in_stock": True,
            "tags": [
                "wireless",
                "earbuds",
                "audio",
                "commute",
                "noise cancellation",
                "fitness",
            ],
        },
        {
            "product_name": "CityCommuter Noise Cancelling Headphones",
            "description": "On-ear wireless headphones with noise cancellation optimized for urban commuting. Compact foldable design with 25-hour battery life.",
            "price": 159.99,
            "brand": "UrbanSound",
            "category": "Electronics",
            "rating": 4.4,
            "in_stock": True,
            "tags": [
                "wireless",
                "headphones",
                "audio",
                "commute",
                "noise cancellation",
                "urban",
                "compact",
            ],
        },
        {
            "product_name": "Premium Wireless Headphones",
            "description": "High-quality wireless headphones with noise cancellation and 30-hour battery life.",
            "price": 219.99,
            "brand": "SoundMaster",
            "category": "Electronics",
            "rating": 4.8,
            "in_stock": True,
            "tags": ["wireless", "headphones", "audio", "bluetooth"],
        },
        {
            "product_name": "Ergonomic Office Chair",
            "description": "Comfortable office chair with lumbar support and adjustable height.",
            "price": 249.99,
            "brand": "ComfortPlus",
            "category": "Furniture",
            "rating": 4.5,
            "in_stock": True,
            "tags": ["chair", "office", "ergonomic", "furniture"],
        },
        {
            "product_name": "Ergonomic Office Chair",
            "description": "Comfortable office chair with lumbar support, adjustable height, and headrest.",
            "price": 269.99,
            "brand": "ComfortPlus",
            "category": "Furniture",
            "rating": 4.6,
            "in_stock": True,
            "tags": ["chair", "office", "ergonomic", "furniture"],
        },
        {
            "product_name": "Smartphone XS Max",
            "description": "Latest smartphone with 6.5-inch display, 128GB storage, and triple camera system.",
            "price": 899.99,
            "brand": "TechGiant",
            "category": "Electronics",
            "rating": 4.8,
            "in_stock": True,
            "tags": ["smartphone", "mobile", "camera", "tech"],
        },
        {
            "product_name": "Smartphone XS Max",
            "description": "Latest smartphone with 6.5-inch display, 256GB storage, and triple camera system.",
            "price": 999.99,
            "brand": "TechGiant",
            "category": "Electronics",
            "rating": 4.9,
            "in_stock": True,
            "tags": ["smartphone", "mobile", "camera", "tech"],
        },
        {
            "product_name": "Cotton T-Shirt",
            "description": "Soft, breathable cotton t-shirt available in multiple colors.",
            "price": 19.99,
            "brand": "FashionBasics",
            "category": "Clothing",
            "rating": 4.2,
            "in_stock": True,
            "tags": ["t-shirt", "cotton", "clothing", "casual"],
        },
        {
            "product_name": "Cotton T-Shirt",
            "description": "Soft, breathable cotton t-shirt available in various sizes.",
            "price": 21.99,
            "brand": "FashionBasics",
            "category": "Clothing",
            "rating": 4.3,
            "in_stock": True,
            "tags": ["t-shirt", "cotton", "clothing", "casual"],
        },
        {
            "product_name": "Stainless Steel Water Bottle",
            "description": "Insulated water bottle that keeps drinks cold for 24 hours or hot for 12 hours.",
            "price": 29.99,
            "brand": "EcoHydrate",
            "category": "Kitchen",
            "rating": 4.6,
            "in_stock": True,
            "tags": ["water bottle", "stainless steel", "insulated", "eco-friendly"],
        },
        {
            "product_name": "Stainless Steel Water Bottle",
            "description": "Insulated water bottle with a sleek design, keeps drinks cold for 24 hours.",
            "price": 32.99,
            "brand": "EcoHydrate",
            "category": "Kitchen",
            "rating": 4.7,
            "in_stock": True,
            "tags": ["water bottle", "stainless steel", "insulated", "eco-friendly"],
        },
        {
            "product_name": "Yoga Mat",
            "description": "Non-slip yoga mat with alignment lines for proper positioning.",
            "price": 39.99,
            "brand": "ZenFitness",
            "category": "Sports",
            "rating": 4.4,
            "in_stock": True,
            "tags": ["yoga", "fitness", "exercise", "mat"],
        },
        {
            "product_name": "Yoga Mat",
            "description": "Eco-friendly yoga mat with extra cushioning for comfort.",
            "price": 44.99,
            "brand": "ZenFitness",
            "category": "Sports",
            "rating": 4.5,
            "in_stock": True,
            "tags": ["yoga", "fitness", "exercise", "mat"],
        },
        {
            "product_name": "Smart Watch Series 5",
            "description": "Fitness tracker and smartwatch with heart rate monitoring and GPS.",
            "price": 299.99,
            "brand": "TechGiant",
            "category": "Electronics",
            "rating": 4.5,
            "in_stock": True,
            "tags": ["smartwatch", "fitness", "wearable", "tech"],
        },
        {
            "product_name": "Organic Coffee Beans",
            "description": "Fair trade, organic coffee beans with rich, bold flavor.",
            "price": 14.99,
            "brand": "MountainBrew",
            "category": "Grocery",
            "rating": 4.7,
            "in_stock": True,
            "tags": ["coffee", "organic", "fair trade", "beans"],
        },
        {
            "product_name": "Leather Wallet",
            "description": "Genuine leather wallet with RFID protection and multiple card slots.",
            "price": 49.99,
            "brand": "LuxeLeather",
            "category": "Accessories",
            "rating": 4.3,
            "in_stock": True,
            "tags": ["wallet", "leather", "accessories", "RFID"],
        },
        {
            "product_name": "Wireless Charging Pad",
            "description": "Fast wireless charging pad compatible with all Qi-enabled devices.",
            "price": 29.99,
            "brand": "PowerUp",
            "category": "Electronics",
            "rating": 4.2,
            "in_stock": True,
            "tags": ["charger", "wireless", "electronics", "Qi"],
        },
        {
            "product_name": "Cast Iron Skillet",
            "description": "Pre-seasoned cast iron skillet for versatile cooking on any heat source.",
            "price": 34.99,
            "brand": "KitchenPro",
            "category": "Kitchen",
            "rating": 4.8,
            "in_stock": True,
            "tags": ["skillet", "cast iron", "cooking", "kitchen"],
        },
        {
            "product_name": "Premium Chef's Knife",
            "description": "High-carbon stainless steel chef's knife with ergonomic handle for precision cutting.",
            "price": 49.99,
            "brand": "KitchenPro",
            "category": "Kitchen",
            "rating": 4.9,
            "in_stock": True,
            "tags": ["knife", "chef", "cooking", "kitchen", "cutting"],
        },
        {
            "product_name": "Silicone Cooking Utensil Set",
            "description": "Set of 5 heat-resistant silicone cooking utensils with wooden handles.",
            "price": 29.99,
            "brand": "KitchenPro",
            "category": "Kitchen",
            "rating": 4.7,
            "in_stock": True,
            "tags": ["utensils", "cooking", "kitchen", "silicone"],
        },
        {
            "product_name": "Digital Kitchen Scale",
            "description": "Precise digital kitchen scale with tare function and multiple measurement units.",
            "price": 19.99,
            "brand": "KitchenPro",
            "category": "Kitchen",
            "rating": 4.6,
            "in_stock": True,
            "tags": ["scale", "kitchen", "measuring", "baking"],
        },
        {
            "product_name": "Fitness Resistance Bands Set",
            "description": "Set of 5 resistance bands of varying strengths for home workouts and physical therapy.",
            "price": 24.99,
            "brand": "FitActive",
            "category": "Sports",
            "rating": 4.7,
            "in_stock": True,
            "tags": ["fitness", "exercise", "resistance bands", "workout", "home gym"],
        },
        {
            "product_name": "Insulated Hiking Water Bottle",
            "description": "Double-walled stainless steel bottle that keeps water cold for 24 hours. Perfect for hiking and outdoor activities.",
            "price": 32.99,
            "brand": "AdventureGear",
            "category": "Sports",
            "rating": 4.8,
            "in_stock": True,
            "tags": ["outdoor", "hiking", "water bottle", "insulated", "camping"],
        },
        {
            "product_name": "Ultralight Packable Daypack",
            "description": "Lightweight, foldable 20L backpack for hiking and travel. Water-resistant and durable.",
            "price": 29.99,
            "brand": "AdventureGear",
            "category": "Sports",
            "rating": 4.6,
            "in_stock": True,
            "tags": ["outdoor", "hiking", "backpack", "travel", "lightweight"],
        },
        {
            "product_name": "Fitness Tracker Band",
            "description": "Waterproof fitness tracker with heart rate monitor, step counter, and sleep tracking.",
            "price": 49.99,
            "brand": "FitActive",
            "category": "Electronics",
            "rating": 4.5,
            "in_stock": True,
            "tags": ["fitness", "wearable", "tracker", "exercise", "health"],
        },
    ]

    # Create more products if needed
    while len(sample_products) < num_products:
        sample_products.append(
            {
                "product_name": f"Test Product {len(sample_products) + 1}",
                "description": f"This is a test product number {len(sample_products) + 1}.",
                "price": 9.99,
                "brand": "TestBrand",
                "category": "Test",
                "rating": 3.0,
                "in_stock": True,
                "tags": ["test", "sample"],
            }
        )

    # Delete the index if it exists
    if es.indices.exists(index=index):
        es.indices.delete(index=index)

    # Create the index with appropriate mappings for e-commerce
    mappings = {
        "mappings": {
            "properties": {
                "product_name": {"type": "text"},
                "description": {"type": "text"},
                "price": {"type": "float"},
                "brand": {"type": "keyword"},
                "category": {"type": "keyword"},
                "rating": {"type": "float"},
                "in_stock": {"type": "boolean"},
                "tags": {"type": "keyword"},
            }
        }
    }

    es.indices.create(index=index, body=mappings)

    # Index the products
    for product in sample_products:
        es.index(index=index, document=product)

    # Refresh the index to make the products searchable immediately
    es.indices.refresh(index=index)

    return (
        f"Created e-commerce test index '{index}' with {len(sample_products)} products"
    )


@mcp.tool()
def search_products_by_category(
    category: str,
    min_price: float = 0,
    max_price: float = 1000,
    min_rating: float = 0,
    in_stock_only: bool = False,
    index: str = DEFAULT_INDEX,
) -> str:
    """
    Search for products in a specific category with optional price and rating filters.

    Args:
        category: The product category to search for
        min_price: Minimum price filter
        max_price: Maximum price filter
        min_rating: Minimum rating filter (0-5)
        in_stock_only: Whether to show only in-stock products
        index: The Elasticsearch index to search

    Returns:
        Formatted search results
    """
    # Build the Elasticsearch query
    es_query = {
        "query": {
            "bool": {
                "must": [{"term": {"category": category}}],
                "filter": [
                    {"range": {"price": {"gte": min_price, "lte": max_price}}},
                    {"range": {"rating": {"gte": min_rating}}},
                ],
            }
        },
        "sort": [{"rating": {"order": "desc"}}, {"price": {"order": "asc"}}],
    }

    # Add in_stock filter if requested
    if in_stock_only:
        es_query["query"]["bool"]["filter"].append({"term": {"in_stock": True}})

    # Execute the search
    try:
        response = es.search(index=index, body=es_query, size=10)
        results = [
            {**hit["_source"], "score": hit["_score"]}
            for hit in response["hits"]["hits"]
        ]
    except Exception as e:
        print(f"Search error: {e}")
        return f"Error searching for products in category '{category}': {str(e)}"

    # Format the results
    if not results:
        return f"No products found in category '{category}' matching your criteria."

    formatted_results = "\n\n".join(
        [
            f"Product {i+1}:\n"
            f"Name: {result.get('product_name', 'Unnamed product')}\n"
            f"Brand: {result.get('brand', 'N/A')}\n"
            f"Price: ${result.get('price', 'N/A')}\n"
            f"Rating: {result.get('rating', 'N/A')}/5\n"
            f"In Stock: {'Yes' if result.get('in_stock', False) else 'No'}\n"
            f"Description: {result.get('description', 'No description')[:150]}..."
            for i, result in enumerate(results)
        ]
    )

    return f"""Products in category '{category}':
Price range: ${min_price} - ${max_price}
Minimum rating: {min_rating}/5
In stock only: {'Yes' if in_stock_only else 'No'}

Results:
{formatted_results}
"""


@mcp.tool()
def search_products_by_brand(brand: str, index: str = DEFAULT_INDEX) -> str:
    """
    Search for products from a specific brand.

    Args:
        brand: The brand name to search for
        index: The Elasticsearch index to search

    Returns:
        Formatted search results
    """
    # Build the Elasticsearch query
    es_query = {
        "query": {"term": {"brand": brand}},
        "sort": [{"rating": {"order": "desc"}}],
    }

    # Execute the search
    try:
        response = es.search(index=index, body=es_query, size=10)
        results = [
            {**hit["_source"], "score": hit["_score"]}
            for hit in response["hits"]["hits"]
        ]
    except Exception as e:
        print(f"Search error: {e}")
        return f"Error searching for products from brand '{brand}': {str(e)}"

    # Format the results
    if not results:
        return f"No products found from brand '{brand}'."

    formatted_results = "\n\n".join(
        [
            f"Product {i+1}:\n"
            f"Name: {result.get('product_name', 'Unnamed product')}\n"
            f"Price: ${result.get('price', 'N/A')}\n"
            f"Category: {result.get('category', 'N/A')}\n"
            f"Rating: {result.get('rating', 'N/A')}/5\n"
            f"In Stock: {'Yes' if result.get('in_stock', False) else 'No'}\n"
            f"Description: {result.get('description', 'No description')[:150]}..."
            for i, result in enumerate(results)
        ]
    )

    return f"""Products from brand '{brand}':

Results:
{formatted_results}
"""
