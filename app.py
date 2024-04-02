import streamlit as st
from dotenv import load_dotenv
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
import os
from sentence_transformers import SentenceTransformer
import spacy

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from tqdm.autonotebook import tqdm
from pinecone import Pinecone as pc

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 PDF Question Asnwering Chatbot')
    st.markdown('''
    ## About
    This app is an LLM-powered PDF chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [BERT] 
 
    ''')
    add_vertical_space(5)
    # st.write('Made with ❤️ by [Prompt Engineer](https://youtube.com/@engineerprompt)')



def main():
    load_dotenv()
    filepath=os.getenv('filepath')
    model_name=os.getenv('model_name')
    key=os.getenv('key')
    # model_name='all-MiniLM-L6-v2'
    # filepath="Survey_of_MLAlgorithmsfor6g.pdf"
    # key="3399662c-37e6-43fa-8f62-706d2ba6050f"
    st.header("Query your PDF to ger answers 💬")
    embeddings = {"id":[],"embed":[]}
    
 
    # upload a PDF file
    file_loader = PyPDFLoader(os.path.join(os.getcwd(),filepath))
    
    
    # st.write(pdf)
    if file_loader is not None:
        documents = file_loader.load()
        # st.write(documents)
 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=30)
        chunks = text_splitter.split_documents(documents)

        ###Specify model
        model = SentenceTransformer(model_name, device='cpu') 

        ##Create Embeddings
        for i in range(len(chunks)):
            chunk_embedding=model.encode(chunks[i].page_content).tolist()
            if len(chunk_embedding)!=0:
                embeddings['id'].append(i)
                embeddings['embed'].append(list(chunk_embedding))
        
        ##Define pinecone vector and create index
        
            
        index_name="chatbot"
        p = pc( api_key=key)
        if index_name not in p.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=384,
                metric='cosine',
                    )

        ###Upsert the data
        
        index=p.Index(index_name)
        # st.write(index.describe_index(index_name))
        if index.describe_index_stats()['total_vector_count']==0:
            
            for i in range(len(chunks)):
                index.upsert(vectors=[{
                    "id": str(embeddings["id"][i]),
                    "values": embeddings["embed"][i]
                    }],
                    namespace="data_ml")
        
         # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
        st.write("Here are the top three answers......")
        if query:
            query_embedding = model.encode(query).tolist()
            # query_embedding=list(map(float, query_embedding))
            xc = index.query(vector=query_embedding, top_k=3,namespace='data_ml')
                    
            for i in xc['matches']:
                if i['score']<0.3:
                    
                    st.write("I couldn't find the answer")
                    break
                else:
                    doc_id=i['id']
                    ans=chunks[int(doc_id)].page_content
                    st.write(ans)


 
       
 
if __name__ == '__main__':
    main()
