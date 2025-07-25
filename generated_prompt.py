def prompt_template(data: str, num_records: int = 5):

    return f"""You are an expert data curator assisting a machine learning engineer in creating a high-quality instruction tuning dataset. Your task is to transform 
    the provided data chunk into diverse question and answer (Q&A) pairs that will be used to fine-tune a language model. 

    For each of the {num_records} entries, generate one or two well-structured questions that reflect different aspects of the information in the chunk. 
    Ensure a mix of longer and shorter questions, with shorter ones typically containing 1-2 sentences and longer ones spanning up to 3-4 sentences. Each 
    Q&A pair should be concise yet informative, capturing key insights from the data.

    IMPORTANT: Return ONLY valid JSON in the exact format below. Do not include any explanatory text, markdown formatting, or code blocks.

    Required JSON format:
    {{
        "generated": [
            {{
                "question": "Your question here...",
                "answer": "Your answer here..."
            }},
            {{
                "question": "Your second question here...",
                "answer": "Your second answer here..."
            }}
        ]
    }}

    Focus on creating clear, relevant, and varied questions that encourage the model to learn from diverse perspectives. Avoid any sensitive or biased 
    content, ensuring answers are accurate and neutral.

    Data to process:
    {data}
    """


if __name__ == "__main__":
    print(prompt_template("nicholas renotte", 10))
