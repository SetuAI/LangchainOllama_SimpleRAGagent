from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# specify the model name
model = OllamaLLM(model="llama3.2:1b")

# define the template


template = """
You are an exeprt in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

# define the prompt
prompt = ChatPromptTemplate.from_template(template)

# define the chain
chain = prompt | model

while True:
    # this will help us ask questions until we quit : q
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    # use retriever to get the relevant reviews
    # retriever is going to embed the question
    # then go to the vector store and look for relevant reviews
    # using similarity search algorithm ,then extract top 5 reviews
    # and then pass it through the chain
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews,
                           "question": question})
    print(result)