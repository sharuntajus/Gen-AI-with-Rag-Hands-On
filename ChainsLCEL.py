
#OLd Way
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

template = """Your job is to come up with a classic dish from the area that the users suggests.
{location}
 YOUR RESPONSE:
"""

# Create a PromptTemplate object by providing:
# - The template string defined above
# - A list of input variables that will be used to format the template
prompt = PromptTemplate.from_template(template)

Ollm = OllamaLLM(
    model="mistral",
    temperature=0.7,
    max_tokens=200,
    top_p=0.9
)

chain = prompt | Ollm
res = chain.invoke({"location": "China"})

print(res)

#Modern Way
from langchain_core.output_parsers import StrOutputParser
template = """Your job is to come up with a classic dish from the area that the user suggests.
{location}
YOUR RESPONSE:
"""

location_chain_lcel = prompt | Ollm | StrOutputParser()
result = location_chain_lcel.invoke({"location": "China"})
print(result)

#Modern LCEL
from langchain_core.runnables import RunnablePassthrough

location_template = """Your job is to come up with a classic dish from the area that the users suggests.
{location}
YOUR RESPONSE:
"""

dish_template = """Given a meal {meal}, give a short and simple recipe on how to make that dish at home.
YOUR RESPONSE:
"""

time_template = """Given the recipe {recipe}, estimate how much time I need to cook it.
YOUR RESPONSE:
"""

# Create the location chain using LCEL (LangChain Expression Language)
# This chain takes a location and returns a classic dish from that region
location_chain_lcel = (
    PromptTemplate.from_template(location_template)  # Format the prompt with location
    | Ollm                                    # Send to the LLM
    | StrOutputParser()                              # Extract the string response
)
dish_chain_lcel = (
    PromptTemplate.from_template(dish_template)      # Format the prompt with meal
    | Ollm                                    # Send to the LLM
    | StrOutputParser()                              # Extract the string response
)
time_chain_lcel = (
    PromptTemplate.from_template(time_template)      # Format the prompt with recipe
    | Ollm                                    # Send to the LLM
    | StrOutputParser()                              # Extract the string response
)
overall_chain_lcel = (
    # Step 1: Generate a meal based on location and add it to the input dictionary
    RunnablePassthrough.assign(meal=lambda x: location_chain_lcel.invoke({"location": x["location"]}))
    # Step 2: Generate a recipe based on the meal and add it to the input dictionary
    | RunnablePassthrough.assign(recipe=lambda x: dish_chain_lcel.invoke({"meal": x["meal"]}))
    # Step 3: Estimate cooking time based on the recipe and add it to the input dictionary
    | RunnablePassthrough.assign(time=lambda x: time_chain_lcel.invoke({"recipe": x["recipe"]}))
)
result = overall_chain_lcel.invoke({"location": "China"})
print(result)


#Multi Step Processing LCEL
from langchain.chains import LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

positive_review = """I absolutely love this coffee maker! It brews quickly and the coffee tastes amazing. 
The built-in grinder saves me so much time in the morning, and the programmable timer means 
I wake up to fresh coffee every day. Worth every penny and highly recommended to any coffee enthusiast."""

negative_review = """Disappointed with this laptop. It's constantly overheating after just 30 minutes of use, 
and the battery life is nowhere near the 8 hours advertised - I barely get 3 hours. 
The keyboard has already started sticking on several keys after just two weeks. Would not recommend to anyone."""

sentiment_template = """Analyze the sentiment of the following product review as positive, negative, or neutral.
Provide your analysis in the format: "SENTIMENT: [positive/negative/neutral]"

Review: {review}

Your analysis:
"""

summary_template = """Summarize the following product review into 3-5 key bullet points.
Each bullet point should be concise and capture an important aspect mentioned in the review.

Review: {review}
Sentiment: {sentiment}

Key points:
"""

response_template = """Write a helpful response to a customer based on their product review.
If the sentiment is positive, thank them for their feedback. If negative, express understanding 
and suggest a solution or next steps. Personalize based on the specific points they mentioned.

Review: {review}
Sentiment: {sentiment}
Key points: {summary}

Response to customer:
"""

sentiment_prompt = PromptTemplate.from_template(sentiment_template)
summary_prompt = PromptTemplate.from_template(summary_template)
response_prompt = PromptTemplate.from_template(response_template)

sentiment_chain = LLMChain(
    llm=Ollm,
    prompt=sentiment_prompt,
    output_key="sentiment"
)

summary_chain = LLMChain(
    llm=Ollm,
    prompt=summary_prompt,
    output_key="summary"
)

response_chain = LLMChain(
    llm=Ollm,
    prompt=response_prompt,
    output_key="response"
)

traditional_chain = SequentialChain(
    chains=[sentiment_chain, summary_chain, response_chain],
    input_variables=["review"],
    output_variables=["sentiment", "summary", "response"],
    verbose=True
)

sentiment_chain_lcel = sentiment_prompt | Ollm | StrOutputParser()
summary_chain_lcel = summary_prompt | Ollm | StrOutputParser()
response_chain_lcel = response_prompt | Ollm | StrOutputParser()

lcel_chain = (
    RunnablePassthrough.assign(
        sentiment=lambda x: sentiment_chain_lcel.invoke({"review": x["review"]})
    )
    | RunnablePassthrough.assign(
        summary=lambda x: summary_chain_lcel.invoke({
            "review": x["review"],
            "sentiment": x["sentiment"]
        })
    )
    | RunnablePassthrough.assign(
        response=lambda x: response_chain_lcel.invoke({
            "review": x["review"],
            "sentiment": x["sentiment"],
            "summary": x["summary"]
        })
    )
)


def test_chains(review):
    """Test both chain implementations with the given review"""
    print("\n" + "=" * 50)
    print(f"TESTING WITH REVIEW:\n{review[:100]}...\n")

    print("TRADITIONAL CHAIN RESULTS:")
    traditional_results = traditional_chain.invoke({"review": review})
    print(f"Sentiment: {traditional_results['sentiment']}")
    print(f"Summary: {traditional_results['summary']}")
    print(f"Response: {traditional_results['response']}")

    print("\nLCEL CHAIN RESULTS:")
    lcel_results = lcel_chain.invoke({"review": review})
    print(f"Sentiment: {lcel_results['sentiment']}")
    print(f"Summary: {lcel_results['summary']}")
    print(f"Response: {lcel_results['response']}")

    print("=" * 50)


# Run tests
test_chains(positive_review)
test_chains(negative_review)