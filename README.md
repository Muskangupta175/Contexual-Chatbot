Here is a chatbot designed to assist with querying a PDF document. It accepts questions related to the content of the document and provides the top three answers. These answers are ranked based on their cosine similarity to the question, offering the most likely solutions.
Steps:
1. The Langchain module serves as the central tool for executing all tasks.
2. The document is taken and segmented into a number of chunks. The chunk size can be specified.
3. Each chunk is transformed into word embeddings using the efficient pre-trained transformer model "all-MiniLM-L6-v2".
4. These document embeddings are then sent to the Pinecone vector database by creating an index.
5. Every question asked to the PDF is converted into word embeddings and compared to the chunks using cosine similarity. The top 3 most similar texts, along with their cosine similarity scores are provided.
6. If the scores are below 0.3, the bot indicates that the answer was not found.
7. This LLM app is hosted using Streamlit platform.
8. By running streamlit python app.py you can run the app in your local system. The .env file requires the path of the PDF in your system which is used for question answering.
