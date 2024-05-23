# [Prompt engineering](https://platform.openai.com/examples) table of contents

- [Prompt engineering table of contents](#prompt-engineering-table-of-contents)
  - [Google Reference](#google-reference)
  - [Prompt examples](#prompt-examples)
    - [Provide an English sentence for a given non-English sentence.](#provide-an-english-sentence-for-a-given-non-english-sentence)
    - [Content summary - maybe a very long paragraph](#content-summary---maybe-a-very-long-paragraph)
    - [Create tables from unstructured text.](#create-tables-from-unstructured-text)
    - [Translate regular text into emoji text.](#translate-regular-text-into-emoji-text)
    - [Calculate time complexity](#calculate-time-complexity)
    - [Extract keywords from a block of text.](#extract-keywords-from-a-block-of-text)
  - [Generate product names from a description and seed words.](#generate-product-names-from-a-description-and-seed-words)
  - [Six strategies](#six-strategies)
    - [Write clear instructions](#write-clear-instructions)
    - [Provide reference text](#provide-reference-text)
    - [Split complex tasks into simpler subtasks](#split-complex-tasks-into-simpler-subtasks)
    - [Give the model time to "think"](#give-the-model-time-to-think)
    - [Use external tools](#use-external-tools)
    - [Test changes systematically](#test-changes-systematically)
  - [More Prompt examples](#more-prompt-examples)
    - [Extracting information](#extracting-information)
    - [Generating text](#generating-text)
    - [Transforms](#transforms)
    - [Code](#code)
    - [Structured data](#structured-data)
    - [Natural language understanding](#natural-language-understanding)

## [Google Reference](https://www.google.com/search?q=Prompt+examples&sourceid=chrome&ie=UTF-8)

## [Prompt examples](https://platform.openai.com/examples)

### Provide an English sentence for a given non-English sentence.

```txt
"__________________________________________________"
You will be provided with statements, and your task is to convert them to standard English.

```

### Content summary - maybe a very long paragraph

```txt
"__________________________________________________"
Summarize the content you just provided to a 13 year old child.
Summarize content you are provided with for a second-grade student.

```

### Create tables from unstructured text.

```txt
"__________________________________________________"
You will be provided with unstructured data, and your task is to parse it into CSV format.
```

### Translate regular text into emoji text.

```txt
Eg: Artificial intelligence is a technology with great promise.

"__________________________________________________"
You will be provided with text, and your task is to translate it into emojis. Do not use any regular text. Do your best with emojis only.

```

### Calculate time complexity

```python
prompt = """

You will be provided with Python code, and your task is to calculate its time complexity.

You will be provided with a piece of code, and your task is to explain it in a concise way.

"""


def add_percentage_change(df, column_name, periods):
    period_map = {'W': 5,'M': 21, 'Y': 252, '3Y': 252 * 3}
    for period in periods:
        if period == 'YTD':
            df['YTD'] = df[column_name] / df.iloc[0][column_name] - 1
        else:
            period_value = period_map.get(period, period)
            new_column_name = f'Chg{period}'
            df[new_column_name] = df[column_name].pct_change(periods=period_value) * 100
            
    return df

```

### Extract keywords from a block of text.

```txt
Eg:

The Rolls Royce of productivity laptops, Lenovo’s X1 Carbon combines a lightweight build, an industry-leading keyboard, strong battery life, and all the ports that most people need. The X1 Carbon (Gen 12) makes some noteworthy improvements over its predecessors, shrinking the laptop’s dimensions and peeling about 0.08 pounds off its weight while upgrading to an Intel Core Ultra processor, its first consumer chip to feature a Neural Processing Unit (NPU) to help with AI. 

You will be provided with a block of text, and your task is to extract a list of keywords from it. The block of text looks like this: "__________________________________________________"


```

## Generate product names from a description and seed words.

```txt
Eg:

Product description: A home milkshake maker. Seed words: fast, healthy, compact.

Use the following text separated by quotation marks to provide the product description and seed words, and your task is to generate product names. "__________________________________________________"

```

## Six strategies

### Write clear instructions

- [Include details in your query](https://platform.openai.com/docs/guides/prompt-engineering/tactic-include-details-in-your-query-to-get-more-relevant-answers) to get more relevant answers.

```txt  

Before:

Write code to calculate the Fibonacci sequence.	

Summarize the meeting notes.	

After:

Write a Python function to efficiently calculate the Fibonacci sequence. Comment the code liberally to explain what each piece does and why it's written that way.

Summarize the meeting notes in a single paragraph. Then write a markdown list of the speakers and each of their key points. Finally, list the next steps or action items suggested by the speakers, if any.

```

- Ask the model to [adopt a persona](https://platform.openai.com/docs/guides/prompt-engineering/tactic-ask-the-model-to-adopt-a-persona):

```txt

When I ask for help to write something, you will reply with a document that contains at least one joke or playful comment in every paragraph.

Eg:

"__________________________________________________". Write an answer that contains at least one joke or playful comment _________________

```

- Use delimiters to clearly indicate distinct parts of the input (Delimiters like triple quotation marks, XML tags, section titles, etc. can help demarcate sections of text to be treated differently)
  
```txt

"__________________________________________________". Summarize the text delimited by quotes with _________________.

Eg:
Summarize the text delimited by triple quotes with a haiku.

------------------------------------------------------------------------

You will be provided with a pair of articles (delimited with XML tags) about the same topic. First summarize the arguments of each article. Then indicate which of them makes a better argument and explain why.

<article> __________________________________________________ </article>
<article> __________________________________________________ </article>

```

- Specify the steps required to complete a task (Some tasks are best specified as a sequence of steps. Writing the steps out explicitly can make it easier for the model to follow them.)

```txt
Eg:

Use the following step-by-step instructions to complete the task: Step 1 - Summarize this text in one sentence with a prefix that says "Summary: ": "__________________________________________________". Step 2 - Translate the summary from Step 1 into Spanish, with a prefix that says "Translation: ". Step 3 - _________________

```

- Provide examples

```txt

Teach me about _________________ . Answer in a consistent style. _________________

```

- Specify the desired length of the output

```txt
"__________________________________________________". Summarize the text delimited by quotes with _________________ in about _______ words | paragraphs | bullet points.

```

### Provide reference text

- Instruct the model to answer using a reference text

```txt

Use the provided articles delimited by quotes to answer questions. "__________________________________________________".  Question: ________________ .

```

- Instruct the model to answer with citations from a reference text (need to insert input document)

### Split complex tasks into simpler subtasks

- Use intent classification to identify the most relevant instructions for a user query

```txt
You will be provided with ________________ queries. Classify each query into a primary category and a secondary category. Provide your output in json format with the keys: primary and secondary.

You will be provided with ________________ inquiries that require ________________. Help me by: 1. ________________. 2. ________________. 3. ________________. 4. ________________. 5. ________________. 6. ________________. 7. ________________. 8. ________________. 9. ________________. 10. ________________.

```

- For dialogue applications that require **very long conversations**, **summarize or filter previous dialogue**

- Summarize long documents piecewise and construct a full summary recursively. (
a further trick that can be useful is to include a running summary of the text that precedes any given point in the book while summarizing content at that point
)

### Give the model time to "think"

- Instruct the model to work out its own solution before rushing to a conclusion (
Sometimes we get better results when we explicitly instruct the model to reason from first principles. simply ask the model if ________________ is correct or not.
)
```txt

Determine if ________________ is correct or not: Problem Statement: "________________". Solution: "________________". Please explain why you came to that conclusion in about _______ bullet points.

------------------------------------------------------------------------------------------------------

First, find your own solution to the problem. Then compare your solution to the following solution and evaluate if the following solution is correct or not. Don't decide if the following solution is correct until you have done the problem yourself. Problem: "________________". Solution: "________________".

```

- Use inner monologue or a sequence of queries to hide the model's reasoning process (partial display, all except the last have their output hidden from the end user, there is no chance that the model’s solution will be biased by user’s attempted solution)

```txt

Step 1:

Follow these steps to answer the question: ________________. 1. First work out your own solution to the problem. Don't rely on the solution was created before, enclose all your work for this step within triple quotes ("""). 2. Compare your solution to the solution was created before and evaluate if that solution is correct or not. Enclose all your work for this step within triple quotes ("""). 3. If the solution was created before made a mistake, determine what ________________ you could give ________________ without giving away the answer. Enclose all your work for this step within triple quotes ("""). 4. If the solution was created before made a mistake, provide the hint from the previous step to ________________ (outside of triple quotes). Instead of writing "Step 4 - ______" write "Hint: __________".

------------------------------------------------------------------------------------------------------

Step 2:

Using 2 prompts:

Problem: "________________". Can you give a solution to the above problem? Then summarize the answer in a single sentence.
(Can be combined with other prompt tricks to improve answer efficiency.)

Problem: "________________".  Bot's answer (which has been summarized): "________________". The given solution: "________________". Compare your solution to the given solution and evaluate if the given solution is correct or not.

Step 3:

let the model use its own analysis to construct a reply in the persona:

You are a ________________. If the given solution made an error, offer a hint to the user in a way that does not reveal the answer. If the given solution did not make an error, simply offer them an encouraging comment.
(Just an example, can be modified to suit the user's needs.)

```

- [Ask the model if it missed anything on previous passes](https://platform.openai.com/docs/guides/prompt-engineering/tactic-ask-the-model-if-it-missed-anything-on-previous-passes)


### Use external tools

### Test changes systematically

## More Prompt examples

### Extracting information

### Generating text

### Transforms

### Code

### Structured data

### Natural language understanding