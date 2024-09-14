import os

from altair import JsonDataFormat
from dotenv import load_dotenv

load_dotenv()

from httpx import request
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema.document import Document

from consts import INDEX_NAME
import requests
from typing import List
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("****Loading to vectorstore done ***")


def ingest_docs2() -> None:
    """_summary_
    """
    from langchain_community.document_loaders import FireCrawlLoader

    langchain_documents_base_urls = [
        "https://python.langchain.com/v0.2/docs/integrations/chat/",
        "https://python.langchain.com/v0.2/docs/integrations/llms/" ,
        "https://python.langchain.com/v0.2/docs/integrations/text_embedding/",
        "https://python.langchain.com/v0.2/docs/integrations/document_loaders/",
        "https://python.langchain.com/v0.2/docs/integrations/document_transformers/",
        "https://python.langchain.com/v0.2/docs/integrations/vectorstores/",
        "https://python.langchain.com/v0.2/docs/integrations/retrievers/",
        "https://python.langchain.com/v0.2/docs/integrations/tools/",
        "https://python.langchain.com/v0.2/docs/integrations/stores/",
        "https://python.langchain.com/v0.2/docs/integrations/llm_caching/",
        "https://python.langchain.com/v0.2/docs/integrations/graphs/",
        "https://python.langchain.com/v0.2/docs/integrations/memory/",
        "https://python.langchain.com/v0.2/docs/integrations/callbacks/",
        "https://python.langchain.com/v0.2/docs/integrations/chat_loaders/",
        "https://python.langchain.com/v0.2/docs/concepts/",
    ]
    langchain_documents_base_urls2 = [langchain_documents_base_urls[0]]
    for url in langchain_documents_base_urls2:
        print(f"FireCrawling {url=}")
        loader = FireCrawlLoader(
            url=url,
            mode="crawl",
            params={
                "limit": 5,
            },
        )
        docs = loader.load()

        print(f"Going to add {len(docs)} documents to Pinecone")
        PineconeVectorStore.from_documents(
            docs, embeddings, index_name="firecrawl-index"
        )
        print(f"****Loading {url}* to vectorstore done ***")

def retrieve_recipes_freemealAPI()-> List[JsonDataFormat]:
    """calls the FreemealAPI to retrieve recipes given a serch argument

    Returns:
        List[JsonDataFormat]: list of recipes in json format
    """
    
    recipes = []
    for recipe in ["Arrabiata", "Pizza", "Tacos"]:

        print(f"Retrieving {recipe} recipe...\n")

        endpoint = f"https://www.themealdb.com/api/json/v1/1/search.php?s={recipe}"
        response = requests.get(endpoint)
        res = response.json()

        print(f"The recipe for {recipe}  was retrieved\n")
        recipes.append(res)
    
    return recipes

def populate_vectorDB(recipes: List[str]):
    """Takes a list of recipes and populates a pinecone vector database

    Args:
        recipes (List[str]): list of recipes in string format 
    """

    INDEX_NAME = "nmpfit-recipes"
   
    documents = [Document(page_content=recipe) for recipe in recipes]
    # Upsert vectors into Pinecone
    PineconeVectorStore.from_documents(documents, embeddings, index_name=INDEX_NAME)
    # Extracting fields

def format_freemeal_recipe(recipe: JsonDataFormat)-> str:
    """Takes a recipe in Json format retrieved from the FreeMealAPI and formats it in natural text

    Args:
        recipe (JsonDataFormat): FreeMealAPI recipe in json format 
    """

    meal_name = recipe["meals"][0]["strMeal"]
    instructions = recipe["meals"][0]["strInstructions"]

    ingredients = []
    i = 1
    while f"strIngredient{i}" in recipe["meals"][0] and recipe["meals"][0][f"strIngredient{i}"] != '':

        ingredient = recipe["meals"][0][f"strIngredient{i}"]

        measure = recipe["meals"][0][f"strMeasure{i}"]
       
        ingredients.append(f"- {ingredient}: {measure}")
        i += 1

        # Formatting the recipe with all ingredients
    formatted_recipe = f"""
        Recipe: {meal_name}

        Instructions:
        {instructions}

        Ingredients:
        {chr(10).join(ingredients)}
    """
    return formatted_recipe
        
    
                
if __name__ == "__main__":

    print("Hello nmpFit\n")
    recipes = retrieve_recipes_freemealAPI()
    formattes_recipes = [format_freemeal_recipe(recipe) for recipe in recipes]
    populate_vectorDB(formattes_recipes)
    
    print("Ingested all Recipes...\n")
    print("Ending")