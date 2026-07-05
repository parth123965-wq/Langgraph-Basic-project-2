from langgraph.graph import START , END , StateGraph
from typing import Annotated , TypedDict
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# Create a Reducers
def merge_dict(existing:dict,new:dict)->dict:
    if existing is None:
        return new
    return {
        **existing , **new
    }

# Create Agent
llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.1)
# Create State
class MainState(TypedDict):
    user_input:str
    score : Annotated[dict[str,int],merge_dict]
    
# Create a Node
def toxic(state:MainState)->dict:
    prompt = (
        "Analyze the following text. Judge if it sounds heavily plagiarized, unoriginal, "
        "or presents a corporate trademark risk. Provide a score from 0 to 100, "
        "where 0 means entirely original and 100 means high risk. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['user_input']}"
    )
    response = llm.invoke(prompt)
    try:
        score = response.content.strip()
    except ValueError:
        score = 0
    return {
        'score':{'toxic':score}
    }
    
def culture(state:MainState)->dict:
    prompt = (
        "Analyze the following text for regional sensitivities, political landmines, "
        "or cultural insensitivity that might offend a global audience. Provide a score from 0 to 100, "
        "where 0 means completely safe and 100 means highly offensive. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['user_input']}"
    )
    response = llm.invoke(prompt)
    try:
        score = response.content.strip()
    except ValueError:
        score = 0
    return {
        'score':{'culture':score}
    }
    
def copywright(state:MainState)->dict:
    prompt = (
        "Analyze the following text. Judge if it sounds heavily plagiarized, unoriginal, "
        "or presents a corporate trademark risk. Provide a score from 0 to 100, "
        "where 0 means entirely original and 100 means high risk. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['user_input']}"
    )
    response = llm.invoke(prompt)
    try:
        score = response.content.strip()
    except ValueError:
        score = 0
    return {
        'score':{'copywright':score}
    }
    

# Create Graph
graph = StateGraph(MainState)

# Create Node
graph.add_node('toxic',toxic)
graph.add_node('culture',culture)
graph.add_node('copywright',copywright)

# Create Edge
graph.add_edge(START,'toxic')
graph.add_edge(START,'culture')
graph.add_edge(START,'copywright')

graph.add_edge('toxic',END)
graph.add_edge('culture',END)
graph.add_edge('copywright',END)

# Compile
app = graph.compile()

inputs = """
    Yo guys! Welcome back to the stream. Today I am going to show you how to hack into 
    your friend's system using a script I copied directly from an online forum. 
    Honestly, traditional security protocols are absolute garbage and anyone still using 
    them is an absolute idiot. Let's dive into the code!
    """
    
response = app.invoke({
    'user_input':inputs,
    'score':{}
})

print(f"Result : {response['score']}")