PROMPT_TEMPLATE = """
Please read and understand the code file(s) below and do the following steps:
1. Identify the purpose of the code file(s) and describe it in a few sentences for each file.
2. Add only the comments to the code file(s) to make it more readable and understandable.
3. Do not make any code change(s) in the file(s).
4. Generate a documentation wiki that explains all the features and functionalities of the code file(s) in detail.
5. Suggest any improvements or changes that can be done to the code file(s) to make it more efficient and readable.
{code}
"""