import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

@st.cache(allow_output_mutation=True)
def cached_model():
  model = SentenceTransformer('jhgan/ko-sroberta-multitask')
  return model

@st.cache(allow_output_mutation=True)
def get_dataset():
  df = pd.read_csv('wellness_dataset.csv')
  df['embedding'] = df['embedding'].apply(json.loads)
  return df

model = cached_model()
df = get_dataset()

st.header('고려고등학교 공대갈동아리')
st.markdown("당신의 마음을 편하게 말해주세요.")

if 'generated' not in st.session_state:
  st.session_state['generated'] = []

if 'past' not in st.session_state:
  st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
  user_input = st.text_input('당신: ', key='input')
  submitted = st.form_submit_button('전송')

if submitted and user_input:
  embedding = model.encode(user_input)

  df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
  answer = df.loc[df['distance'].idxmax()]

  st.session_state.past.append(user_input)
  st.session_state.generated.append(answer['챗봇'])

if st.session_state['generated']:
  for i in range(len(st.session_state['generated'])-1, -1, -1):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    message(st.session_state["generated"][i], key=str(i))