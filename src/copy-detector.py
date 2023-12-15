import openai
import tiktoken
import sys
import xml.etree.ElementTree as ET
from Levenshtein import ratio
from bleurt import score
from rouge_metric import PyRouge
import numpy as np

model = "gpt-4"
openai.api_key = ""
checkpoint = "BLEURT-20"
scorer = score.BleurtScorer(checkpoint)
rouge = PyRouge()

def splitter(
    point:str
) -> tuple:
    enc = tiktoken.encoding_for_model(model)
    q = enc.encode(point)
    q1 = enc.decode(q[:len(q)//2])
    q2 = enc.decode(q[len(q)//2:])
    return q1,q2

def sim(
   completion:str,
   q2:str     
)-> float:
    return ratio(q2, completion), scorer.score(references=[q2], candidates=[completion])[0], rouge.evaluate([completion], [[q2]])['rouge-1']['f'] ### for BLEURT I am using the recommended checkpoint and for ROUGE unigram overlap, but I could also compute the longest common sequence or weighted

def complete(
    document:str,
    q1:str,
    q2:str
) -> str:
    enc = tiktoken.encoding_for_model(model)
    prompt = f'You are provided with the a first part of a sentence from a {document}. Finish the sentence as exactly appeared in the {document}. Only rely on the original {document} to finish the sentence.\nFirst half of the sentence:{q1}\nGenerated second halgo of the sentence:' 
    print(prompt)
    if model=="text-davinci-003" or model=="text-davinci-002":
        text = openai.Completion.create(
            model=model,
            prompt=prompt,
            # max_tokens=len(enc.encode(q2)),
            temperature=0
        )
        return text['choices'][0]['text']

    else:
        response = openai.ChatCompletion.create(
              model=model,
              messages=[
                  {"role": "system", "content": "You are a chatbot"},
                  {"role": "user", "content": prompt},
              ],
            #   max_tokens=len(enc.encode(q2)),
              temperature=0 # to ensure reproducibility
          )

        result = ''
        for choice in response.choices:
            result += choice.message.content
        return result

if __name__ == "__main__":
    doc = sys.argv[1]
    sentence = sys.argv[2]
    model = sys.argv[3]

    q1, q2 = splitter(sentence) ### we split narrative field
    compl = complete(doc,q1,q2)    
    print("Real second half:```", q2,"```")
    leven,bleurt,rou = sim(compl,q2)
    print("Syntactic overlap:",leven, "Semantic overlap:",bleurt, "Lexical overlap:", rou)
    print()