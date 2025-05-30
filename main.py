from langchain_ollama.llms import OllamaLLM
# helps langchain interact with Ollama models

from langchain_core.prompts import ChatPromptTemplate
# 'ChatPromptTemplate' is a tool for creating structured prompts, especially for chat-based models.
# It helps you define a template for how you want to ask questions or give instructions to the LLM.

from vector import retriever
# This imports an object named 'retriever' from your 'vector.py' file.
# We'll look at 'vector.py' later, but for now, know that this 'retriever' is responsible
# for fetching the relevant restaurant reviews based on the user's question.


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
# LangChain uses this structured prompt object to manage how input variables (like 'reviews' and 'question')
# are formatted into the final instruction for the LLM.


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
    
'''
questions : 
how are the vegan options


'''