# here we write script for embedding the documents
# we are going to host eh vector database locally using ChromaDB
# and this will quickly look up relevant information and model will use this
# to reply with response

# we will send the csv push to the vector database
# and then as soon as we ask the question
# it will look for relevant documents in that databse
# we will pass that to llm as list of reviews and answer the question


from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
# LangChain uses this 'Document' class as a standard way to represent a piece of text
import os
import pandas as pd

# load csv : Data for vector store
df = pd.read_csv("realistic_restaurant_reviews.csv")

# load the embeddings model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# defining  the location where locally host the vector db
db_location = "./chroma_langchain_db"

# check and see if the db already exists (optional)
# if it already exists it means the process 
# of converting csv to vectors and storing it in db is already performed
add_documents = not os.path.exists(db_location)

#'if' statement checks the 'add_documents' variable. The code inside
# this block will only run if 'add_documents' is 'True' (i.e., the database directory
# doesn't exist yet, and we need to process the CSV and add documents to the new store).

if add_documents: # if we actually need to add them or doesnt exists then
    
    documents = []
    ids = []
    
    # iterate through csv row by row
    for i, row in df.iterrows(): #loops through each row in the DataFrame
        # i : index  , row : record
        document = Document(
            # what we will actually vectorize (title and review)
            page_content=row["Title"] + " " + row["Review"],
            # metadata is additional information we want to store
            # but we wont be querying based on metadata
            metadata={"rating": row["Rating"], "date": row["Date"]},
            # This creates a dictionary of metadata associated with the document.
        # It stores the "Rating" and "Date" from the CSV. This data isn't directly embedded
        #  but is stored alongside the vector and can be retrieved.

            id=str(i) # id is just the index of the row
        )
        ids.append(str(i)) # we are appendigs ids and documents
        documents.append(document)
         # - The newly created 'document' object (containing the page_content and metadata
        #   for the current review) is appended to the 'documents' list.

# create the vector store
vector_store = Chroma(
    collection_name="restaurant_reviews", # name of the collection
    persist_directory=db_location, # store in db
    embedding_function=embeddings # embedding function is the model we are using
)


if add_documents:
    vector_store.add_documents(documents=documents, ids=ids) # add docs to the vector store
 
# making vector store usable 
# retriever is the object that will actually look up the relevant documents   
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5} # tells the retriever to find and return the top 5 most relevant documents
    # it will look up for 5 relevant reviews and then pass them to the LLM
    # If you wanted the top 10, you'd use `"k": 10`.
)