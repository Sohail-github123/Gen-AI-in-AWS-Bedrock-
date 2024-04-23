#!/usr/bin/env python
# coding: utf-8

# # Lesson 1 - first generations with Amazon Bedrock

# Lesson 1. 
# 
# You'll start with using Amazon Bedrock to prompt a model and customize how it generates its response.
# 
# **Note:** To access the `requirements.txt` file, go to `File` and click on `Open`. Here, you will also find all helpers functions and datasets used in each lesson.
#  

# ### Import all needed packages

# In[3]:


import boto3
import json


# ### Setup the Bedrock runtime

# In[4]:


bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')


# In[5]:


prompt = "Write a one sentence summary of Muumbai."


# In[6]:


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


# In[7]:


response = bedrock_runtime.invoke_model(**kwargs)


# In[8]:


response


# In[9]:


response_body = json.loads(response.get('body').read())


# In[10]:


print(json.dumps(response_body, indent=4))


# In[11]:


print(response_body['results'][0]['outputText'])


# ### Generation Configuration

# In[12]:


prompt = "Write a summary of Mumbai."


# In[13]:


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


# In[14]:


response = bedrock_runtime.invoke_model(**kwargs)
response_body = json.loads(response.get('body').read())

generation = response_body['results'][0]['outputText']
print(generation)


# In[15]:


print(json.dumps(response_body, indent=4))


# In[16]:


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


# In[17]:


response = bedrock_runtime.invoke_model(**kwargs)
response_body = json.loads(response.get('body').read())

generation = response_body['results'][0]['outputText']
print(generation)


# In[18]:


print(json.dumps(response_body, indent=4))


# ### Working with other type of data

# In[19]:


from IPython.display import Audio


# In[20]:


audio = Audio(filename="dialog.mp3")
display(audio)


# In[21]:


with open('transcript.txt', "r") as file:
    dialogue_text = file.read()


# In[22]:


print(dialogue_text)


# In[23]:


prompt = f"""The text between the <transcript> XML tags is a transcript of a conversation. 
Write a short summary of the conversation.

<transcript>
{dialogue_text}
</transcript>

Here is a summary of the conversation in the transcript:"""


# In[24]:


print(prompt)


# In[25]:


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


# In[26]:


response = bedrock_runtime.invoke_model(**kwargs)


# In[27]:


response_body = json.loads(response.get('body').read())
generation = response_body['results'][0]['outputText']


# In[28]:


print(generation)

