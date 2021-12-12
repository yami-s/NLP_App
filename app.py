import streamlit as st 

#nlp packages
import spacy
from textblob import TextBlob


#tokenization and lemmatization
def token_lemmatization(raw_text):
    nlp=spacy.load('en_core_web_sm')
    doc=nlp(raw_text)
    result = [('Token: {},\n Lemma: {}'.format(token.text,token.lemma_))for token in doc]
    return result

#POS TAGGING
def pos_text(raw_text):
    nlp=spacy.load('en_core_web_sm')
    doc=nlp(raw_text)
    result=[('Token: {},\n POS Tag: {}'.format(token.text,token.pos_))for token in doc]
    return result
    
#Named entity
def named_entity_extraction(raw_text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(raw_text)
    entities = [(entity.text,entity.label_)for entity in doc.ents]
    result = ['Entities: {}'.format(entities)]
    return result

def main():
    st.title("NLP With Streamlit")
    st.markdown("""
            This is a Natural Language Processing(NLP) based App for basic NLP task:
            Tokenization,Lemmatization,Parts of Speech(POS) Tagging, Named Entity Extraction(NER) 
            and Sentiment Analysis
            """)
    given_text = st.text_area("Enter Text","Type Here ..")
    process = st.selectbox("Select precess: ",[' ','Tokens and Lemma','POS Tag','Named Entities','Sentiment Analysis'])
    if st.button("Analyse"):
        #Tokenization and lemmatization
        if process=='Tokens and Lemma':
            st.subheader("Tokenized text")
            nlp_results=token_lemmatization(given_text)
            st.json(nlp_results)
        #POS Tag
        elif process=='POS Tag':
            st.subheader("POS Tags")
            pos_result = pos_text(given_text)
            st.json(pos_result)
        #named entities
        elif process=='Named Entities':
            st.subheader("Extracted entities")
            entity_result = named_entity_extraction(given_text)
            st.json(entity_result)
        #sentiment analysis
        elif process=='Sentiment Analysis':
            st.subheader("Analysed text")
            blob = TextBlob(given_text)
            result_sentiment = blob.sentiment
            st.success(result_sentiment)
        #in case no process is selected
        else:
            st.warning("Select process")
  
if __name__ == '__main__':
    main()