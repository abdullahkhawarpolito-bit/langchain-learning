# OpenAI Chat Model
from langchain_openai import ChatOpenAI

# Anthropic Model (not used currently)
from langchain_anthropic import ChatAnthropic

# Loads API keys from .env file
from dotenv import load_dotenv

# Used to create dynamic prompts
from langchain_core.prompts import PromptTemplate

# Converts AIMessage output into plain string
from langchain_core.output_parsers import StrOutputParser

# Runnable utilities
from langchain_core.runnables import RunnableBranch, RunnableLambda

# Parses model output into structured Pydantic object
from langchain_core.output_parsers import PydanticOutputParser

# Pydantic model creation
from pydantic import BaseModel, Field

# Restricts sentiment values to only positive or negative
from typing import Literal

# Load environment variables
load_dotenv()

# Create OpenAI model
model = ChatOpenAI(model="gpt-4o-mini")

# Parser to convert AI output into plain text
parser = StrOutputParser()


# ---------------------------------------------------
# Pydantic Schema
# ---------------------------------------------------
# This defines the structure we expect from the model
# Model MUST return:
# {
#   "sentiment": "positive"
# }
# OR
# {
#   "sentiment": "negative"
# }
# ---------------------------------------------------

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(
        description='Give the sentiment of the feedback'
    )


# Parser that converts model output into Feedback object
parser2 = PydanticOutputParser(pydantic_object=Feedback)


# ---------------------------------------------------
# SENTIMENT CLASSIFICATION PROMPT
# ---------------------------------------------------
# This prompt asks model to classify feedback sentiment
# ---------------------------------------------------

prompt1 = PromptTemplate(

    template='''
Classify the sentiment of the following feedback text
into positive or negative.

Feedback:
{feedback}

{format_instruction}
''',

    # User input variable
    input_variables=['feedback'],

    # Automatically inject formatting instructions
    partial_variables={
        'format_instruction': parser2.get_format_instructions()
    }
)


# ---------------------------------------------------
# CLASSIFIER CHAIN
# ---------------------------------------------------
# Flow:
# Prompt -> Model -> Pydantic Parser
#
# Output:
# Feedback(sentiment='positive')
# ---------------------------------------------------

classifier_chain = prompt1 | model | parser2


# ---------------------------------------------------
# POSITIVE RESPONSE PROMPT
# ---------------------------------------------------

prompt2 = PromptTemplate(

    template='''
Write an appropriate response to this positive feedback:

{feedback}
''',

    input_variables=['feedback']
)


# ---------------------------------------------------
# NEGATIVE RESPONSE PROMPT
# ---------------------------------------------------

prompt3 = PromptTemplate(

    template='''
Write an appropriate response to this negative feedback:

{feedback}
''',

    input_variables=['feedback']
)


# ---------------------------------------------------
# BRANCH CHAIN
# ---------------------------------------------------
# RunnableBranch works like IF-ELSE logic
#
# IF sentiment == positive:
#       use positive response chain
#
# IF sentiment == negative:
#       use negative response chain
#
# ELSE:
#       return fallback message
# ---------------------------------------------------

branch_chain = RunnableBranch(

    # Condition 1
    (
        lambda x: x.sentiment == 'positive',

        # Positive response chain
        prompt2 | model | parser
    ),

    # Condition 2
    (
        lambda x: x.sentiment == 'negative',

        # Negative response chain
        prompt3 | model | parser
    ),

    # Default fallback
    RunnableLambda(
        lambda x: "Could not determine sentiment"
    )
)


# ---------------------------------------------------
# FINAL CHAIN
# ---------------------------------------------------
# First classify sentiment
# Then send output into branch chain
# ---------------------------------------------------

chain = classifier_chain | branch_chain


# ---------------------------------------------------
# RUN CHAIN
# ---------------------------------------------------

result = chain.invoke({

    'feedback': 'This is a beautiful phone'

})


# Print final response
print("\nFINAL RESPONSE:\n")
print(result)


# ---------------------------------------------------
# PRINT CHAIN GRAPH
# ---------------------------------------------------
# Visualizes the complete runnable pipeline
# ---------------------------------------------------

print("\nCHAIN GRAPH:\n")

print(chain.get_graph().draw_ascii())