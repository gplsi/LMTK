# Prompt Template Structure

This document explains how prompt templates are structured in this codebase and how you can create your own custom templates.

## Overview

Prompt templates are organized into four main components, each handling a different part of the prompt construction process:

1. **System Blocks** - Define the initial instructions and context for the model
2. **User Blocks** - Format user inputs and instructions
3. **Assistant Blocks** - Format model responses and continuation prompts
4. **Final Blocks** - Handle end-of-prompt tokens and stopping criteria

## Template Components

### 1. System Blocks (`system_blocks.py`)

System blocks provide the initial context and instructions for the model. They typically include:
- Model's role and behavior guidelines
- Safety instructions
- Response formatting requirements

**Example Classes:**
- `DefaultSystem` - Empty system prompt
- `AlpacaSystem` - Basic instruction-following format
- `Llama2System` - Llama 2's safety-focused system prompt
- `VicunaSystem` - Chat-style system prompt

### 2. User Blocks (`user_blocks.py`)

User blocks format the user's input according to the model's expected format. They handle:
- Instruction formatting
- Optional input/context
- Language-specific formatting

**Example Classes:**
- `DefaultUser` - Passes through input as-is
- `AlpacaUser` - Formats as Alpaca instruction with optional input
- `Llama2User` - Wraps in `[INST]` tags
- `FlanUser` - Supports multiple languages

### 3. Assistant Blocks (`assistant_blocks.py`)

Assistant blocks format the model's responses and continuation prompts. They handle:
- Response formatting
- Prefixes for continued generation
- Model-specific formatting

**Example Classes:**
- `DefaultAssistant` - Simple response format
- `AlpacaAssistant` - Formats as Alpaca response
- `Llama2Assistant` - Handles Llama 2's response format

### 4. Final Blocks (`final_blocks.py`)

Final blocks handle the end-of-prompt tokens and stopping criteria. They define:
- End-of-prompt tokens
- Stop sequences for generation
- Model-specific termination criteria

**Example Classes:**
- `DefaultFinal` - Basic EOS token handling
- `Llama3Final` - Handles Llama 3's stop tokens
- `StableLMFinal` - StableLM specific stop tokens

## Creating Custom Templates

To create a new template format:

1. **Choose a base class** from the appropriate module (e.g., `PromptStyle`)
2. **Implement the required methods** (typically `apply()` and possibly `stop_tokens()`)
3. **Add your new class** to the corresponding `*_blocks.py` file
4. **Register your template** in the appropriate factory or configuration

### Example: Creating a Custom User Block

```python
from src.tasks.instruction.templates.base import PromptStyle

class MyCustomUser(PromptStyle):
    """My custom user prompt format."""
    
    def apply(self, prompt: str, **kwargs: str) -> str:
        """Format the user input with custom formatting."""
        self.logger.debug("Applying MyCustomUser formatting")
        return f"[USER_QUERY] {prompt} [END_QUERY]\n[ASSISTANT_RESPONSE]"
```

## Best Practices

1. **Inherit from `PromptStyle`** for consistency
2. **Use logging** for debugging
3. **Document your template** with docstrings
4. **Handle edge cases** (empty inputs, missing kwargs, etc.)
5. **Add type hints** for better IDE support
6. **Keep formatting consistent** with the model's training data

## Adding New Template Types

If you need to add a completely new type of template:

1. Create a new module (e.g., `my_blocks.py`)
2. Define your template classes
3. Update any factory classes or configuration to include your new templates
4. Add tests for your new templates

## Testing Your Templates

Always test your templates with the target model to ensure they produce the expected format. Consider:
- Different input lengths
- Special characters
- Edge cases (empty strings, very long inputs)
- Multilingual content if applicable
