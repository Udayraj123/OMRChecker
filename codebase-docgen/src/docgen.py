from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

def generate_comments(code, api_key):
    # Initialize Gemini with the user-provided API key
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",  # Use a supported model name
        temperature=0.0,
        google_api_key=api_key
    )

    # Define the prompt template using ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                """
Please read and understand the code file(s) below and do the following steps:
1. Identify the purpose of the code file(s) and describe it in a few sentences.
2. Add only the comments to the code file(s) to make it more readable and understandable.
3. Do not make any code changes in the file(s).
4. Generate a documentation wiki that explains all the features and functionalities of the code file(s) in detail.
5. Suggest any improvements or changes that can be done to the code file(s) to make it more efficient and readable.

Here is the code:
{code}
                """,
            ),
        ]
    )

    # Create a chain with the prompt and the model
    chain = prompt | llm

    # Invoke the chain with the code as input
    response = chain.invoke({"code": code})
    return response.content