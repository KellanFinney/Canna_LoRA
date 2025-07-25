import json
from pathlib import Path
from pydantic import BaseModel
from litellm import completion
from colorama import Fore 
from config_loader import get_config

class Score(BaseModel):
    score: int
    explanation: str

class Rank(BaseModel):
    accuracy: Score
    style: Score

def llm_call(record: str, model: str = None) -> dict:
    """
    Assess quality of instruction tuning record using local LLM
    
    Args:
        record: The Q&A record to assess
        model: LLM model to use (if None, uses config)
    """
    if model is None:
        config = get_config()
        model = config.quality.get('local_model', 'ollama_chat/qwen2.5:14b')
    
    stream = completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"""Classify this instruction tuning record between 1-10 in terms of accuracy and style, provide explanations as to your reasoning. The answer should be completely self contained. 
                                If the question is not in fact a question it should receive 0 for accuracy. If the answer does not adequately answer the question it should receive 1 for accuracy. If the question 
                                or answer are in any way harmful, not helpful or not honest it should be classified as 1 for style. If the question or answer are blank or ..., return 1 for both. \n\n Record: {record} Return as JSON""",
            }
        ],
        stream=True,
        options={"num_predict": 2000, "temperature": 0.2},
        format=Rank.model_json_schema(),
    )
    data = ""
    for x in stream: 
        delta = x['choices'][0]["delta"]["content"]
        if delta is not None: 
            print(Fore.LIGHTBLUE_EX+ delta + Fore.RESET, end="") 
            data += delta 
    return json.loads(data)


def main():
    """Main quality assessment function using configuration"""
    config = get_config()
    
    # Get settings from config
    input_file = config.quality.get('input_file', './data/instruction.json')
    filtered_output = config.quality.get('filtered_output', './data/instructionquality.json')
    quality_report = config.quality.get('quality_report', './qualityresults.json')
    min_accuracy = config.quality.get('min_accuracy_score', 6)
    min_style = config.quality.get('min_style_score', 6)
    model = config.quality.get('local_model', 'ollama_chat/qwen2.5:14b')
    
    # Convert to Path objects
    input_path = Path(input_file)
    filtered_path = Path(filtered_output)
    quality_path = Path(quality_report)
    
    # Ensure input file exists
    if not input_path.exists():
        print(f"{Fore.RED}❌ Input file not found: {input_path}{Fore.RESET}")
        return
    
    print(f"{Fore.CYAN}🔍 Starting Quality Assessment{Fore.RESET}")
    print(f"{Fore.CYAN}📄 Input: {input_path}{Fore.RESET}")
    print(f"{Fore.CYAN}🤖 Model: {model}{Fore.RESET}")
    print(f"{Fore.CYAN}🎯 Thresholds: Accuracy ≥{min_accuracy}, Style ≥{min_style}{Fore.RESET}")
    
    quality = []
    instructions = []
    
    with open(input_path, 'r', encoding='utf-8') as f: 
        data = json.load(f)
        
        total_records = len(data)
        processed = 0
        
        for pair in data: 
            processed += 1
            print(f"\n{Fore.YELLOW}📋 Processing {processed}/{total_records}: {str(pair)[:100]}...{Fore.RESET}") 
            
            try:
                result = llm_call(pair, model) 
                
                accuracy_score = result['accuracy']['score']
                style_score = result['style']['score']
                
                if accuracy_score >= min_accuracy and style_score >= min_style:
                    instructions.append(pair)
                    quality.append({**pair, 'quality': result})
                    print(f"{Fore.GREEN}✅ PASSED (Accuracy: {accuracy_score}, Style: {style_score}){Fore.RESET}")
                else:
                    print(f"{Fore.RED}❌ FILTERED (Accuracy: {accuracy_score}, Style: {style_score}){Fore.RESET}")
                    
            except Exception as e:
                print(f"{Fore.RED}❌ ERROR: {str(e)}{Fore.RESET}")
                continue

    # Save results
    print(f"\n{Fore.CYAN}💾 Saving Results{Fore.RESET}")
    
    with open(filtered_path, 'w', encoding='utf-8') as f: 
        json.dump(instructions, f, indent=2, ensure_ascii=False)
    print(f"{Fore.GREEN}✅ Filtered instructions: {filtered_path} ({len(instructions)} records){Fore.RESET}")

    with open(quality_path, 'w', encoding='utf-8') as f: 
        json.dump(quality, f, indent=2, ensure_ascii=False)
    print(f"{Fore.GREEN}✅ Quality report: {quality_path} ({len(quality)} records){Fore.RESET}")
    
    # Summary
    pass_rate = (len(instructions) / len(data)) * 100 if data else 0
    print(f"\n{Fore.MAGENTA}📊 Quality Assessment Complete{Fore.RESET}")
    print(f"{Fore.MAGENTA}📈 Pass Rate: {pass_rate:.1f}% ({len(instructions)}/{len(data)}){Fore.RESET}")


if __name__ == "__main__": 
    main()