logic_block_llama3 = """
{% set loop_messages = messages %}
{%- if messages[0]['role'] == 'system' %}
    {%- if messages[0]['content'] is string %}
        {%- set system_message = messages[0]['content']|trim %}
    {%- else %}
        {%- set system_message = messages[0]['content'][0]['text']|trim %}
    {%- endif %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- if tools is not none %}
        {%- set system_message = "You are a helpful assistant..." %} {# Simplified for brevity #}
    {%- else %}
        {%- set system_message = "" %}
    {%- endif %}
{%- endif %}
"""

system_message_block_llama3 = """
{% if system_message != "" %}
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ system_message }}<|eot_id|>
{% endif %}
"""


user_assistant_block_llama3 = """
{% for message in loop_messages %}
    <|start_header_id|>{{ message['role'] }}<|end_header_id|>

    {{ message['content']|trim }}<|eot_id|>
{% endfor %}
"""


final_block_llama3 = """
<|start_header_id|>assistant<|end_header_id|>
"""


template_llama3 = """
{{ logic_block_llama3 }}
{{ system_message_block_llama3 }}
{{ user_assistant_block_llama3 }}
{{ final_block_llama3 }}
"""