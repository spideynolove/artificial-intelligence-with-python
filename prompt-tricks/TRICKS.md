# [Chat with ChatGPT](https://platform.openai.com/examples) table of contents

- [Chat with ChatGPT table of contents](#chat-with-chatgpt-table-of-contents)
  - [Persona Pattern](#persona-pattern)
    - [Basic Persona](#basic-persona)
    - [From medium](#from-medium)
      - [Creating a Character](#creating-a-character)
      - [Knowing our Character](#knowing-our-character)
    - [Reddit](#reddit)
    - [Others](#others)
  - [Audience Persona Pattern](#audience-persona-pattern)
  - [Output Automator Pattern](#output-automator-pattern)
  - [Question Refinement Pattern](#question-refinement-pattern)
  - [Context Manager Pattern](#context-manager-pattern)
  - [Cognitive Verifier Pattern](#cognitive-verifier-pattern)
  - [Flipped Interaction Pattern](#flipped-interaction-pattern)
  - [Few-shot Examples](#few-shot-examples)
  - [Recipe Pattern](#recipe-pattern)
  - [Alternative Approaches Pattern](#alternative-approaches-pattern)
  - [Ask for Input Pattern](#ask-for-input-pattern)
  - [Menu Actions Pattern](#menu-actions-pattern)
  - [Semantic Filter Pattern](#semantic-filter-pattern)
  - [Template Pattern](#template-pattern)
  - [The Reflection Pattern](#the-reflection-pattern)
  - [Chain-of-Thought Prompting](#chain-of-thought-prompting)
  - [Zero-Shot Learning](#zero-shot-learning)
  - [One-Shot Learning](#one-shot-learning)
  - [Iterative Prompting](#iterative-prompting)
  - [Negative Prompting](#negative-prompting)
  - [Hybrid Prompting](#hybrid-prompting)
  - [Prompt Chaining](#prompt-chaining)
  - [Other resources](#other-resources)
    - [promptbase](#promptbase)
    - [SEO prompts](#seo-prompts)
    - [COSTAR Framework](#costar-framework)
      - [example:](#example)
    - [Best Prompt Techniques for Best LLM Responses](#best-prompt-techniques-for-best-llm-responses)
    - [The Perfect Prompt: A Prompt Engineering Cheat Sheet](#the-perfect-prompt-a-prompt-engineering-cheat-sheet)
    - [Sectioning with Delimeters](#sectioning-with-delimeters)
    - [CO-STAR and TIDD-EC Frameworks](#co-star-and-tidd-ec-frameworks)

## [Persona](https://character.ai/) Pattern

### Basic Persona

```text
Act as X

You are X

From now on, act as [persona]. Pay close attention to [details to focus on]. Provide outputs that [persona] would regarding the input.

Become a "______________" (Eg: financial advisor) and suggest "______________" (investment strategies for long-term wealth accumulation)

Embody the persona of a "______________" (fantasy author) and craft an "______________" (engaging opening paragraph for a mythical tale)

Take on the persona of "______________" (an ancient historian) and provide insights into the "______________" (significance of the Roman Empire).

```

### From [medium](https://medium.com/@treycwong/prompt-engineering-a-user-persona-with-chatgpt-and-character-ai-cd36c554fa25)

- First, you need to understand the character from ChatGPT and will prompt on Persona(Jake, Max or ...) ’s background. 
- Second, you need to think from the viewpoint of the chatbot that you will be designing. Meaning that you would need to know Persona’s motivations, preference and physical attributes.

#### Creating a Character

#### Knowing our Character

### Reddit 

- [Collection of ChatGPT persona prompts](https://www.reddit.com/r/ChatGPTPro/comments/11v04tw/collection_of_chatgpt_persona_prompts/)

- I created [a list](https://contentwritertools.com/chatygpt-personas-to-enhance-you-prompts) of [ChatGPT Personas](https://www.reddit.com/r/PromptEngineering/comments/15snobt/i_created_a_list_of_chatgpt_personas/)

- [PromptEngineering](https://www.reddit.com/r/PromptEngineering/)

- Introducing the [Persona+ Method](https://www.reddit.com/r/ChatGPT/comments/18pda6e/introducing_the_persona_method_a_revolutionary/): A Revolutionary Approach to AI Interaction

### Others

- [Github](https://github.com/kiil/awesome-chatgpt-personas)

- [delve](https://www.delve.ai/blog/personas-chatgpt)

- [zapier](https://zapier.com/blog/prompt-engineering/)

## Audience Persona Pattern

```text
Explain X to me. Assume that I am Persona Y. You will need to replace "Y" with an appropriate persona, such as "have limited background in computer science" or "a healthcare expert".

Eg:
Explain the use of prompt patterns. Assume the audience is not well aware of this topic.
Explain the plot of the "Game of Thrones" series to me. Assume that I am someone who has never watched a single episode.
Explain large language models to me. Assume that I am a highschool student.
Explain how the supply chains for US grocery stores work to me. Assume that I am a 8 year old boy
```

## Output Automator Pattern

```text
Whenever you produce an output, consider step X or property Y.

Eg:
From now on, whenever I ask you to compare something, use tabular format.
From now on, whenever you provide a summary, include the key points in bullet points.
From now on, whenever I ask you to explain something, provide a detailed description. Use simple English and simple sentences,since I'm not pretty good at English.

```

## Question Refinement Pattern

```text

From now on, whenever I ask a question, suggest a better version of the question to use instead

From now on, whenever I ask a question, suggest a better version of the question and ask me if I would like to use it instead

From now on, when I ask a question, suggest a better version of the question to use that incorporates information specific to [use case] and ask me if I would like to use your question instead.

```

- [libguides](https://libguides.utoledo.edu/home)

```text
A simple question "__________________________________________________" 
  can be answered with a "yes" or "no" 
  contain the answers within themselves. 
  can only be answered by a fact, or a series of facts.

A critical question "__________________________________________________"
  leads to more questions.
  provokes discussion.
  concerns itself with audience and authorial intent.
  derives from a critical or careful reading of the text or understanding of a topic.
  addresses or ties in wider issues.
  moves you out of your own frame of reference ("What does this mean in our context?") to your author's ("What was the author trying to convey when he/she wrote this? How would the audience have responded?").

Ask speculative questions.

Ask What if? questions. 
  What if "__________________________________________________"

Ask how the topic fits into larger contexts.

Ask questions that reflect disagreements with a source.

Ask questions that build on agreements with a source.

Ask questions about the nature of the thing itself, as an independent entity.

Ask questions analogous to those that others have asked about similar topics.

Turn positive questions into negative ones.

Look for questions posed in scholarly articles; ask part of the questions.

Find a Web discussion list on your topic...reading the exchanges to understand the kinds of questions those on the list discuss.
```

- [medium](https://medium.com/@hugoblanc.blend/the-question-refinement-patter-the-art-of-asking-better-questions-b81e27367650)

```text
Daily Work Use Cases

Software Engineer
Whenever I ask a question about implementing a feature, suggest a better version of the question that focuses on best practices and the specific programming language or framework I’m using.

Product Manager
Whenever I ask a question about measuring product success, suggest a better version of the question that considers relevant KPIs and specific aspects of success.

Marketing Strategy Consultation
Whenever I ask a question about social media strategy, suggest a better version of the question that considers my client’s target audience and industry-specific practices.

```

## Context Manager Pattern

```text
Within scope X. Please consider Y. Please ignore Z.

Eg:
For our discussion about AI, only consider its applications in healthcare. Please ignore its applications in other industries such as finance or entertainment. Let's start.

```

## Cognitive Verifier Pattern

```text
When you are asked a question, follow these rules. "__________________________________________________".
Eg:

When you are asked a question, follow these rules. Generate a number of additional questions that would help you more accurately answer the question. Combine the answers to the individual questions to produce the final answer to the overall question.

When I ask you a question, generate three additional questions that would help you give a more accurate answer. When I have answered the three questions, combine the answers to produce the final answers to my original question.

```

## Flipped Interaction Pattern

```text
Prompt structure:
• I would like you to ask me questions to achieve X
• You should ask questions until condition Y is met or to achieve this goal
(alternatively, forever)
• (Optional) ask me the questions one at a time, two at a time, ask me the first
question, etc.
• You will need to replace "X" with an appropriate goal, such as "creating a meal
plan" or "creating variations of my marketing materials."
• You should specify when to stop asking questions with Y. Examples are "until
you have sufficient information about my audience and goals" or "until you know
what I like to eat and my caloric targets."

Eg:
I would like you to ask me questions to help me diagnose a problem with my Internet. Ask me questions until you have enough information to identify the two most likely causes. Ask me one question at a time. Ask me the first question.

```

```text

From now on, I would like you to ask me questions to [do a specific task]. When you have enough information to [do the task], create [output you want]. If you need more information, ask me for it.

I would like you to ask me questions to achieve X. You should ask questions until condition Y is met.

Eg:
I want to improve my knowledge of "Data Structures and Algorithms". Ask me questions about this topic.
Assume you are an interviewer and you interview me for the position of software engineer intern at your company. So, ask me 20 trivial questions. Begin with the first question
```

## Few-shot Examples

```text
A "___________________", while B "___________________". How are C "___________________"?

Eg:
Foundation Models such as GPT-3 are used for natural language processing, while models like DALL-E are used for image generation. How are Foundation Models used in the field of robotics?
```

## Recipe Pattern

```text
Prompt structure:
  - I would like to achieve X
  - I know that I need to perform steps A,B,C
  - Provide a complete sequence of steps for me
  - Fill in any missing steps
  - (Optional) Identify any unnecessary steps
  - You will need to replace "X" with an appropriate task. You will then need to specify the steps A, B, C that you know need to be part of the recipe / complete plan.

Eg:
I would like to create a trading advisor platform. I know that I need to perform steps make an website and build a pricing prediction model. Provide a complete sequence of steps for me. Fill in any missing steps.
```

## Alternative Approaches Pattern

```text
```

## Ask for Input Pattern

```text
```

## Menu Actions Pattern

```text
```

## Semantic Filter Pattern

```text
```

## Template Pattern

- examples:

```text
Crafting email drafts:

Template: “Subject: [Subject line]. Greeting: [Salutation]. Body: [Main content of the email]. Closing: [Closing remarks].”
Example prompt: “Please generate an email draft inviting colleagues to a team-building event.”

Outlining a blog post:

Template: “Introduction: [Introduction to the topic]. Section 1: [Key points for section 1]. Section 2: [Key points for section 2]. Conclusion: [Closing thoughts].”
Example prompt: “Please generate an outline for a blog post discussing the benefits of mindfulness.”

Designing interview questions:

Template: “Question 1: [Specific interview question]. Question 2: [Specific interview question]. Question 3: [Specific interview question].”
Example prompt: “Please generate a set of interview questions for a software engineering position.”
```

## The Reflection Pattern
  
```text

When you provide an answer, please explain the reasoning and assumptions behind your response. If possible, use specific examples or evidence to support your answer of why [prompt topic] is the best. Moreover, please address any potential ambiguities or limitations in your answer, in order to provide a more complete and accurate response.

Whenever you generate an answer. Explain the reasoning and assumptions behind your answer.

Eg:
Whenever I ask you to generate a code in Python, try to provide the reasons for each line, since I'm new to programming using Python.

Whenever I ask you to optimize my code, try to provide reasoning for each of your changes.
```

## Chain-of-Thought Prompting

```text
Describe the process of "___________________", from "___________________" to "___________________".

Eg:
Describe the process of developing a Foundation Model in AI, from data collection to model training
```

## Zero-Shot Learning

```text
Explain what a "___________________" is.

Eg:
Explain what a large language model is.

```

## One-Shot Learning

```text
A "___________________". Explain what it is and how it works

Eg:
A Foundation Model in AI refers to a model like GPT-3, which is trained on a large dataset and can be adapted to various tasks. Explain what BERT is in this context.

```

## Iterative Prompting

```text
Initial
Tell me about the "___________________".
Answer:
1. ...
2. ...

Refined (on 1. or 2. or ...):
Can you provide more details about "___________________"?

```

## Negative Prompting

```text
Explain "___________________" without "___________________".

Eg:
Explain the concept of machine learning without using technical jargon.
```

## Hybrid Prompting

```text

```

## Prompt Chaining

```text
```

## Other resources

### [promptbase](https://promptbase.com/marketplace)

### [SEO prompts](https://searchengineland.com/advanced-ai-prompt-engineering-strategies-seo-436286)

### [COSTAR Framework](https://medium.com/@frugalzentennial/unlocking-the-power-of-costar-prompt-engineering-a-guide-and-example-on-converting-goals-into-dc5751ce9875)

```text
Context : Providing background information helps the LLM understand the specific scenario.

Objective (O): Clearly defining the task directs the LLM’s focus.

Style (S): Specifying the desired writing style aligns the LLM response.

Tone (T): Setting the tone ensures the response resonates with the required sentiment.

Audience (A): Identifying the intended audience tailors the LLM’s response to be targeted to an audience.

Response (R): Providing the response format, like text or json, ensures the LLM outputs, and help build pipelines.
```

#### example:
  
```text
# CONTEXT # 
I am a personal productivity developer. In the realm of personal development and productivity, there is a growing demand for systems that not only help individuals set goals but also convert those goals into actionable steps. Many struggle with the transition from aspirations to concrete actions, highlighting the need for an effective goal-to-system conversion process.

#########

# OBJECTIVE #
Your task is to guide me in creating a comprehensive system converter. This involves breaking down the process into distinct steps, including identifying the goal, employing the 5 Whys technique, learning core actions, setting intentions, and conducting periodic reviews. The aim is to provide a step-by-step guide for seamlessly transforming goals into actionable plans.

#########

# STYLE #
Write in an informative and instructional style, resembling a guide on personal development. Ensure clarity and coherence in the presentation of each step, catering to an audience keen on enhancing their productivity and goal attainment skills.

#########

# Tone #
 Maintain a positive and motivational tone throughout, fostering a sense of empowerment and encouragement. It should feel like a friendly guide offering valuable insights.

# AUDIENCE #
The target audience is individuals interested in personal development and productivity enhancement. Assume a readership that seeks practical advice and actionable steps to turn their goals into tangible outcomes.

#########

# RESPONSE FORMAT #
Provide a structured list of steps for the goal-to-system conversion process. Each step should be clearly defined, and the overall format should be easy to follow for quick implementation. 

#############

# START ANALYSIS #
If you understand, ask me for my goals.

# [GOAL]
My goal is to write better articles for medium.com readers

```

```text
# CONTEXT #
You are tasked with transforming a general goal into a SMART goal. This could be in a professional setting, such as work or personal development.

#######

# OBJECTIVE #
The goal is to convert a broad or vague objective into a SMART goal, ensuring it is Specific, Measurable, Achievable, Relevant, and Time-bound.

#######

# STYLE # 
Maintain a clear and concise style, focusing on the key components of a SMART goal. This could emulate a guide or instructional format.

#######

# TONE #
Maintain a neutral and informative tone, providing guidance on how to refine the goal without adding unnecessary complexity.

#######

# AUDIENCE #
The target audience is individuals or teams seeking to set clear and actionable goals. The explanation should be accessible to a general readership.

#######

# RESPONSE FORMAT #
Provide a structured explanation or list format to guide the user in converting their goal into a SMART goal.

#######

# START #
When you are ready, ask me for a goal to convert to SMART goal.

# [User]
I have a store and sell t-shirts, would like to increase my sales

```

### [Best Prompt Techniques for Best LLM Responses](https://medium.com/the-modern-scientist/best-prompt-techniques-for-best-llm-responses-24d2ff4f6bca)

- Prompt Principles

  - Prompt Structure and Clarity: Integrate the intended audience in the prompt.
  - Specificity and Information: Implement example-driven prompting (Use few-shot prompting)
  - User Interaction and Engagement: Allow the model to ask precise details and requirements until it has enough information to provide the needed response
  - Content and Language Style: Instruct the tone and style of response
  - Complex Tasks and Coding Prompts: Break down complex tasks into a sequence of simpler steps as prompts.

- CO-STAR Prompt Framework

- Prompt Types and Tasks

  - Text summarization
  - Zero and few shot learning
  - Information extractions
  - Question answering
  - Text and image classification
  - Conversation
  - Reasoning
  - Code generation

### The Perfect Prompt: A [Prompt Engineering Cheat Sheet](https://medium.com/the-generator/the-perfect-prompt-prompt-engineering-cheat-sheet-d0b9c62a2bba)

- the AUTOMAT and the CO-STAR framework
- output format definition
- few-shot learning
- chain of thoughts
- prompt templates
- RAG, retrieval-augmented generation
- formatting and delimiters and
- the multi-prompt approach.

### Sectioning with [Delimeters](https://medium.com/@thomasczerny/alternative-prompt-engineering-approach-sectioning-with-delimeters-112ff9d7953f)


### [CO-STAR and TIDD-EC Frameworks](https://vivasai01.medium.com/mastering-prompt-engineering-a-guide-to-the-co-star-and-tidd-ec-frameworks-3334588cb908)
