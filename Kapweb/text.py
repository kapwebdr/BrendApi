from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForSeq2SeqLM, pipeline
from ctransformers import AutoModelForCausalLM

from os import path

current_file = path.realpath(__file__)

def Translate(modelname,text):
    tokenizer = AutoTokenizer.from_pretrained(modelname, cache_dir=path.join(path.dirname(current_file),"..", "Cache"))
    model = AutoModelForSeq2SeqLM.from_pretrained(modelname)
    return tokenizer.decode(model.generate(tokenizer.encode(text, return_tensors='pt')).squeeze(), skip_special_tokens=True)

def Helsinki(from_lang,to_lang,text):
    modelname = "Helsinki-NLP/opus-mt-"+from_lang+"-"+to_lang
    return Translate(modelname,text)

def Sentiment(prompt,modelname="nlptown/bert-base-multilingual-uncased-sentiment"):
    model       = AutoModelForSequenceClassification.from_pretrained(modelname, cache_dir=path.join(path.dirname(current_file),"..", "Cache"))
    tokenizer   = AutoTokenizer.from_pretrained(modelname)
    classifier  = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    results     = classifier(prompt)
    return results

def Resume(prompt,modelname= "pszemraj/led-large-book-summary",min_length=16,max_length=256, device='mps'):
    summarizer = pipeline("summarization",modelname,device=device)
    result = summarizer(
        prompt,
        min_length=min_length,
        max_length=max_length,
        no_repeat_ngram_size=3,
        encoder_no_repeat_ngram_size=3,
        repetition_penalty=3.5,
        num_beams=4,
        early_stopping=True,
    )
    return result

def Fill(prompt,modelname= "xlm-roberta-base", device='mps'):
    unmasker = pipeline('fill-mask', model=modelname, device=device)
    return unmasker(prompt)

def Classify(prompt,modelname= "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",candidate_labels = ["politics", "economy", "entertainment", "environment"],multi_label=True, device='mps'):
    classifier = pipeline("zero-shot-classification", model=modelname,device=device)
    sequence_to_classify = prompt
    result = classifier(sequence_to_classify, candidate_labels, multi_label=multi_label)
    return result

def Qa(prompt, context,modelname= "deepset/roberta-base-squad2", device='mps'):
    nlp = pipeline('question-answering', model=modelname, tokenizer=modelname, device=device)
    QA_input = {
        'question': prompt,
        'context': context
    }
    result = nlp(QA_input)
    return result

def Emotion(prompt,modelname= "j-hartmann/emotion-english-distilroberta-base", device='mps'):
    classifier = pipeline("text-classification", model=modelname, top_k=None, device=device)
    return classifier(prompt)

def LLM(prompt,modelname="tiiuae/falcon-7b-instruct"):
    tokenizer = AutoTokenizer.from_pretrained(modelname, cache_dir=path.join(path.dirname(current_file),"..", "Cache"))
    pipeline = transformers.pipeline(
        "text-generation",
        model=modelname,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    sequences = pipeline(
        prompt,
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    return sequences

def LLObjectMetal(modelname="TheBloke/zephyr-7B-beta-GGUF",model_file="zephyr-7b-beta.Q4_K_M.gguf",model_type="mistral"):
    llm = AutoModelForCausalLM.from_pretrained(modelname, model_file=model_file, model_type=model_type, gpu_layers=50) # , cache_dir=path.join(path.dirname(current_file),"..", "Cache")
    return llm

def LLMetal(prompt,modelname="TheBloke/zephyr-7B-beta-GGUF",model_file="zephyr-7b-beta.Q4_K_M.gguf",model_type="mistral"):
    llm = LLObjectMetal(modelname, model_file, model_type) # , cache_dir=path.join(path.dirname(current_file),"..", "Cache")
    response = llm(prompt)
    return response
