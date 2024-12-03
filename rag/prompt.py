from haystack.dataclasses import ChatMessage

system_message = ChatMessage.from_system(
    "You are a helpful AI assistant using provided supporting documents and conversation history to assist humans")

user_message_template = """Given the conversation history and the provided supporting documents, give a brief answer to the question.
Note that supporting documents are not part of the conversation. If question can't be answered from supporting documents, say so.

    Conversation history:
    {% for memory in memories %}
        {{ memory.content }}
    {% endfor %}

    Supporting documents:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{query}}
    \nAnswer:
"""
