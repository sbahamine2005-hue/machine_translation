from transformers import MarianMTModel, MarianTokenizer
from nltk import sent_tokenize
import nltk
import torch

nltk.download('punkt_tab')

def translate_large_text(text,model_name="Helsinki-NLP/opus-mt-en-fr"):
    #now we loading the tokenizer and the model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    #Now let's break the large text to sentence
    sentences = sent_tokenize(text)

    #translat text 
    translated_texts = []
    for sentence in sentences:
        if not sentence.strip():
            continue

        input_sentence = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            translate_sentence = model.generate(**input_sentence)

        translate_sentence = tokenizer.decode(translate_sentence[0], skip_special_tokens = True)
        translated_texts.append(translate_sentence)

    return " ".join(translated_texts)    


if __name__ == "__main__" :
    large_text = """
            Artificial intelligence is changing the world rapidly. Machine translation allows people 
            from different cultures to communicate effortlessly. However, long documents pose a challenge 
            for small translation models due to strict token length limitations. By splitting the text into 
            individual sentences, we can process indefinitely large files without cutting off any information.
            """
    french_translation =translate_large_text(text= large_text, model_name="Helsinki-NLP/opus-mt-en-fr") 
    print("---------THE French Translation---------")
    print(french_translation)

    arabic_translation =translate_large_text(text= large_text, model_name="Helsinki-NLP/opus-mt-en-ar")
    print("\n\n---------THE Arabic Translation---------")
    print(arabic_translation)


"""
الإستخبارات الإصطناعية تغير العالم بسرعة وتتيح الترجمة الآلية للناس من مختلف الثقافات التواصل دون جهد.
 غير أن الوثائق الطويلة تشكل تحديا لنماذج الترجمة الصغيرة بسبب القيود الصارمة على طول الوثائق. وبتقسيم النص إلى جمل فردية،
 يمكننا أن نعالج ملفات كبيرة إلى أجل غير مسمى دون أن نقطع أي معلومات
"""    