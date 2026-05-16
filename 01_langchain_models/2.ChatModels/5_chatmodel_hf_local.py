from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm= HuggingFacePipeline.from_model_id(
    model_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    pipeline_kwargs= dict(
        max_length=200,
        temperature=1.0
    )
)
model= ChatHuggingFace(llm=llm)
result = model.invoke("what is the capital of pakistan")
print(result.content)
