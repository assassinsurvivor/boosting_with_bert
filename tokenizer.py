import re
import sys
import string
from collections import defaultdict



global_vocab=["##"+i for i in string.ascii_lowercase]

with open(sys.argv[1],"r") as filer:
    text=filer.readlines()[0]
    


def basic_process(text):
    '''
    Input: basic text/corpus
    Output: returns a dictionary consisting of words separated by whitespace as key and their frequency in the corpus as value. example: {'p i z z a':25}
    
    '''
    
    
    vocab_dict=defaultdict(int)
    text=pattern_ap.sub(" 's",text.strip().lower()).split()
    text=[pattern_text.sub("",k) for k in text]

    for word in text:
        vocab_dict[' '.join(k for k in word)]+=1

    return vocab_dict


def frequent_bichar(vocab_dict,min_freq=-(sys.maxsize)):

    '''
    Input:
        1.A dictionary consisting of words separated by whitespace as key and their frequency in the corpus as value. example: {'p i z z a':25}
        2.Minimum frequency that the bigrams should occur

    Output: returns highest frequency matching bigrams that maximizes the likelihood of occurence.

    '''
    
    global global_vocab

    sub_class=[]

    new_vocab=defaultdict(int)
    freq_dict=defaultdict(int)

    for key,value in vocab_dict.items():
        word=key.split()
        for i in range(len(word)-1):
            if i==0:
                sub_class.append(word[i]+word[i+1])
            freq_dict[word[i],word[i+1]]+=value
    
    max_freq=max(freq_dict,key=freq_dict.get)

    if freq_dict[max_freq]<min_freq:
        print(as_use)
        

    if ''.join(v for v in max_freq) in sub_class:
        global_vocab.append(''.join(v for v in max_freq))
    else:
        global_vocab.append('##'+''.join(v for v in max_freq))

    pattern_new=re.compile(' '.join(max_freq))

    for key,value in vocab_dict.items():

        new_key=pattern_new.sub(''.join(max_freq),key)
        new_vocab[new_key]+=value

    return new_vocab


def decode(vocab_dict,text):

    

    '''
    Input:
        1.the vocab dictionary obtained after running the function 'frequent_bichar'
        2.the word or a string of words sep by spaces example: transformer
    Output:
        ['trans','form','er'] assuming 'trans','form' & 'er' were present in the dictionary and the word transformer was not present.

    '''

    final_decode=[]

    for word in text.split():

        store=[]
        start=0
        no_match_flag=False
        token=[i for i in word]
        

        while(start<len(token)):
            current_find=None
            end=len(token)
            

            while(start<end):

                if start>0:
                    sub_str="##"+"".join(token[start:end])
                else:
                    sub_str="".join(token[start:end])

                if sub_str in vocab_dict:
                    current_find=sub_str
                    break

                end=end-1

            if current_find is None:
                no_match_flag=True
                break
            else:
                store.append(current_find)
            start=end
        final_decode+=store
        

    return final_decode
            
        
        
num_iterations=sys.argv[2] #maximum length of the dictionary --user input


#initializing the dictionary
vocab_dict=basic_process(text)
#print(vocab_dict)

for _ in range(num_iterations):
    try:
        vocab_dict=frequent_bichar(vocab_dict,min_freq=sys.argv[3])
    except:
        break

print(global_vocab)
