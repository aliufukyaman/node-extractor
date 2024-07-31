from langchain_community.llms import HuggingFaceTextGenInference
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.chat_models import Llama2Chat

from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from fastapi import APIRouter

router = APIRouter()

template_messages = [
    SystemMessage(content=
                  """
                  You are a language analyst and I want you to to analyze the sentence I give and extract its sequence of actions in right order. Right order should be the time of actions.
                  These are the only possible return values: OnVariableChange,OnKeyRelease,OnKeyPress,OnClick,OnWindowResize,OnMouseEnter,OnMouseLeave,OnTimer,Console,Alert,Log,Assign,SendRequest,Navigate,Save,Delete,PlaySound,PauseSound,StopSound,Branch,Map,Filter,Reduce,Sort,GroupBy,Merge,Split,Show,Hide,Update,DisplayModal,CloseModal,Highlight,Tooltip,RenderChart,FetchData,StoreData,UpdateData,DeleteData,CacheData
                  Consider the list below when analyzing and only return listed items above. Don't use anything else but only these action items above. Also pay attention to the definitions just at right of the actions when analyzing the input sentence.
                  The input only consists of the text to analyze, just analyze the sentence you get.


                  These are some examples with input and its results:
                    input: 'Play a sound when a key is pressed and stop it when the key is released.', result: 'OnKeyPress, PlaySound, OnKeyRelease, StopSound'
                    input: 'Fetch data when a button is clicked, cache the data, and then display it on the screen.', result: 'OnClick, FetchData, CacheData, Show'
                    input: 'Reduce a list of scores to find the highest score and log the result.', result: 'Reduce, Log'
                    input: 'Highlight an element and show a tooltip with additional information and save data to database when data is deleted.', result: 'DeleteData, Highlight, Tooltip, Save'
                    input: 'Log a message when the mouse leaves an element, then play a sound and show an alert, then pause the sound.', result: 'OnMouseLeave, Log, PlaySound, Alert, PauseSound'
                    input: 'When a variable changes, fetch data, transform it, update the information, and display it on the screen.', result: 'OnVariableChange, FetchData, Map, Update, Show'
                    input: 'Log a message when the window is resized.', result: 'OnWindowResize, Log'
                    input: 'Highlight an element when the mouse enters and hide it when the mouse leaves.', result: 'OnMouseEnter, Highlight, OnMouseLeave, Hide'
                    input: 'Fetch new data at specified time intervals and render it as a chart on the screen.', result: 'OnTimer, FetchData, RenderChart'
                    input: 'At specified time intervals, fetch data, split it into parts, group the data by an attribute, and print message to the console.', result: 'OnTimer, FetchData, Split, GroupBy, Console'
                    input: 'Assign a value to a variable and store the data and delete records.', result: 'Assign, StoreData, Delete'
                    input: 'Send a request to fetch user data, filter out inactive users, and then show the active users on the screen.', result: 'SendRequest, Filter, Show'
                    input: 'When an element is clicked, evaluate a condition and navigate to different pages based on the result.', result: 'OnClick, Branch, Navigate'
                    input: 'Display a modal when a key is pressed, then close it when the key is released.', result: 'OnKeyPress','DisplayModal','OnKeyRelease', CloseModal'
                    input: 'Sort the data afret updating them, then merge with other data.', result: 'Sort, UpdateData, Merge'

                  Only write whe actions and don't write anything else like explanation or other dialogues.
                  """),
    HumanMessagePromptTemplate.from_template("{text}"),
]

prompt_template = ChatPromptTemplate.from_messages(template_messages)

llm = HuggingFaceTextGenInference(
    inference_server_url="http://host.docker.internal:8080/",
    max_new_tokens=512,
    top_k=50,
    temperature=0.01,
    repetition_penalty=1.0,
)

model = Llama2Chat(llm=llm)
chain = LLMChain(llm=model, prompt=prompt_template)


@router.get("/actions")
async def get_actions(sentence: str):
    result = chain.run(text=sentence)
    return result