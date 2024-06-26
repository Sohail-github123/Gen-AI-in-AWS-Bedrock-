# # Lesson 1 - first generations with Amazon Bedrock
 
# You'll start with using Amazon Bedrock to prompt a model and customize how it generates its response.
# **Note:** To access the `requirements.txt` file, go to `File` and click on `Open`. Here, you will also find all helpers functions and datasets used in each lesson.

import boto3
import json


# ### Setup the Bedrock runtime

bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')

prompt = "Write a one sentence summary of Muumbai."

kwargs = {
    "modelId": "amazon.titan-text-lite-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body": json.dumps(
        {
            "inputText": prompt
        }
    )
}


response = bedrock_runtime.invoke_model(**kwargs)

response

response_body = json.loads(response.get('body').read())

print(json.dumps(response_body, indent=4))

print(response_body['results'][0]['outputText'])


# ### Generation Configuration

prompt = "Write a summary of Mumbai."

kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body" : json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 100,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
    )
}


response = bedrock_runtime.invoke_model(**kwargs)
response_body = json.loads(response.get('body').read())

generation = response_body['results'][0]['outputText']
print(generation)

print(json.dumps(response_body, indent=4))

kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body" : json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 500,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
    )
}

response = bedrock_runtime.invoke_model(**kwargs)
response_body = json.loads(response.get('body').read())

generation = response_body['results'][0]['outputText']
print(generation)

print(json.dumps(response_body, indent=4))


# ### Working with other type of data

from IPython.display import Audio

audio = Audio(filename="dialog.mp3")
display(audio)


with open('transcript.txt', "r") as file:
    dialogue_text = file.read()

print(dialogue_text)


prompt = f"""The text between the <transcript> XML tags is a transcript of a conversation. 
Write a short summary of the conversation.

<transcript>
{dialogue_text}
</transcript>

Here is a summary of the conversation in the transcript:"""


print(prompt)

kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body": json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0,
                "topP": 0.9
            }
        }
    )
}



response = bedrock_runtime.invoke_model(**kwargs)

response_body = json.loads(response.get('body').read())
generation = response_body['results'][0]['outputText']

print(generation)

