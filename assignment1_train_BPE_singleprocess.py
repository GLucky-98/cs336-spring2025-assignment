# type tips: bytes in python just like a list
# a=b'r'  a[0]=114
# and when you visit a byte using a index it will return a int 
# a=b'ra' a[0]=114
# bytes([114])=b'r'

import regex as re
from collections import Counter
import os

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # single processing

    # read file
    with open(input_path,'r') as f:
        text=f.read() #str

    # pre-tokenizer
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pre_tokens=pre_tokenization(text,PAT,special_tokens)

    
    # get words counts and byte pair counts
    # words_counts the words count          counter{bytes:counts}    Eg :(b' low':1000)
    # words        the words in pre_tokens  dict{bytes:list[bytes]]  Eg :(b' low':[b' ',b'l',b'o',b'w'])
    # bp_counts    the pair count           dict{(bytes,bytes):int}  Eg :(b'l',b'o'):30
    words,words_count,bp_count=get_count(pre_tokens)

    
    #initial vocab & merges 
    vocab={i:bytes([i]) for i in range(256)} # int:bytes
    
    cu_vocab_index=256 #current vocab size
    
    # special tokens from index 256
    for st in special_tokens:
        vocab[cu_vocab_index]=st.encode('utf-8')
        cu_vocab_index+=1

    merges=[]
    
    
    # find -> merge -> update vocab  & merge  & words & bp_count
    while cu_vocab_index<vocab_size:
        if not bp_count:
            break

        # find max byte pair counts
        max_bp=max(bp_count.items(), key=lambda item: (item[1], item[0]))[0] # the most important part to pass the test because of the 'lexicographically'
        # add max_pair to merges and vocab 
        merges.append(max_bp)
        vocab[cu_vocab_index]=max_bp[0]+max_bp[1]

        words,bp_count=merge_pair(words,words_count,bp_count,max_bp)
        

        cu_vocab_index+=1

    return vocab,merges


# Tool function


def pre_tokenization(text,PAT,special_tokens):
    # handle special tokens
    special_tokens=sorted(list(map(re.escape,special_tokens)),key=len,reverse=True)

    #text=re.split(f'({'|'.join(special_tokens)})',raw_text) # list[str]  capture group pattern
    text=re.split('|'.join(special_tokens),text) # list[str]

    pre_tokens=[]
    for t in text:
        pre_tokens+=(re.findall(PAT,t))
    
    return pre_tokens


def get_count(pre_tokens:list[str]):
    # words_counts counter{bytes:counts}    E.g :(b' low':1000)
    # words        dict{bytes:list[bytes]]  E.g :(b' low':[b' ',b'l',b'o',b'w'])
    words_count=Counter([i.encode('utf-8') for i in pre_tokens]) # words_counts: bytes(word):counts
    words={}
    for _,bs in enumerate(words_count):
        words[bs]=[bytes([i]) for i in bs]
    
    # bp_counts    (b1(bytes),b2(bytes)):counts
    bp_count={} 
    for word,_ in words.items():
          for b1,b2 in zip(word,word[1:]):
            word_count=words_count[word] 
            pair=(bytes([b1]),bytes([b2]))
            bp_count[pair]=bp_count.get(pair,0)+word_count

    return words,words_count,bp_count

def merge_pair(words,words_count,bp_count,max_pair):
    for idx,word in words.items():
        # the idx actually is the complete word like b' low'
        # the word is list type of bytes like [b' ',b'l',b'o',b'w'])
        count=words_count[idx]
        i=0
        while i < len(word):
            if i+1<len(word) and word[i]==max_pair[0] and word[i+1]==max_pair[1]:
                # left
                if i-1>=0:
                    # old pair
                    pair=(word[i-1],word[i])
                    bp_count[pair]-=count
                    # new pair
                    pair=(word[i-1],max_pair[0]+max_pair[1])
                    bp_count[pair]=bp_count.get(pair,0)+count
                # right
                if i+2<len(word):
                    # old pair
                    pair=(word[i+1],word[i+2])
                    bp_count[pair]-=count
                    # new pair
                    pair=(max_pair[0]+max_pair[1],word[i+2])
                    bp_count[pair]=bp_count.get(pair,0)+count
                
                # update word
                word[i]=word[i]+word[i+1]
                del word[i+1]
            
            i+=1

        # update word
        words[idx]=word
    
    # set zero to the max_pair count
    del bp_count[max_pair]
    
    return words,bp_count


vocab,_=run_train_bpe(
        input_path='data/corpus.en',
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

print(vocab)





# old version without tool function
# and some part is different from above version 
# import os
# import regex as re
# from collections import Counter
# def run_train_bpe(
#     input_path: str | os.PathLike,
#     vocab_size: int,
#     special_tokens: list[str],
#     **kwargs,
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#     """Given the path to an input corpus, run train a BPE tokenizer and
#     output its vocabulary and merges.

#     Args:
#         input_path (str | os.PathLike): Path to BPE tokenizer training data.
#         vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
#         special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
#             These strings will never be split into multiple tokens, and will always be
#             kept as a single token. If these special tokens occur in the `input_path`,
#             they are treated as any other string.

#     Returns:
#         tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#             vocab:
#                 The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
#                 to bytes (token bytes)
#             merges:
#                 BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
#                 representing that <token1> was merged with <token2>.
#                 Merges are ordered by order of creation.
#     """

#     # single processing

#     # read file
#     with open(input_path,'r',encoding='utf-8') as f:
#         raw_text=f.read() #str

#     #initial vocab & merges
#     vocab={i:bytes([i]) for i in range(256)} # int:bytes
#     cu_vocab_index=256 #current vocab size
#     for st in special_tokens:
#         vocab[cu_vocab_index]=st.encode('utf-8')
#         cu_vocab_index+=1

#     merges=[]

#     # handle special tokens
#     special_tokens=sorted(list(map(re.escape,special_tokens)),key=len,reverse=True)
#     #text=re.split(f'({'|'.join(special_tokens)})',raw_text) # list[str]
#     text=re.split('|'.join(special_tokens),raw_text) # list[str]
#     # pre-tokenizer
#     PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

#     words=[]
#     for t in text:
#         words+=(re.findall(PAT,t))

#     # firsttime construct
#     # words_counts bytes(word):counts
#     # words        list[bytes]
#     # words_list   list[list[bytes]]
#     words=[i.encode('utf-8') for i in words]
#     words_counts=Counter(words) # words_counts: bytes(word):counts
#     words=[]
#     words_list=[]
#     for _,bs in enumerate(words_counts):
#         words.append(bs)
#         words_list.append([bytes([i]) for i in bs])  
    
#     # bp_counts    (b1(bytes),b2(bytes)):counts
#     # bp_pos       (b1(bytes),b2(bytes)):[]
#     bp_counts={} 
#     bp_pos={} 
#     for i in range(len(words)):
#         word=words[i]
#         for b1,b2 in zip(word,word[1:]):
#             word_count=words_counts[word] 
#             pair=(bytes([b1]),bytes([b2]))
#             if bp_pos.get(pair) is None:
#                 bp_pos[pair]=[i]
#             elif i not in bp_pos.get(pair):
#                 bp_pos[pair].append(i)
#             bp_counts[pair]=bp_counts.get(pair,0)+word_count
   

#     while cu_vocab_index<vocab_size:
#         if not bp_counts:
#             break

#         # find max byte pair counts
#         max_bp=max(bp_counts.items(), key=lambda item: (item[1], item[0]))[0]
#         #print(max(bp_counts.items(), key=lambda item: (item[1], item[0])))
#         merges.append((max_bp[0],max_bp[1]))
#         vocab[cu_vocab_index]=max_bp[0]+max_bp[1]

#         # merge 处理bp_pos bp_counts words_list
#         for i in bp_pos[max_bp]:
#             word=words_list[i]
#             word_count=words_counts[words[i]]
#             new_word=[]
#             j=0
#             while j < len(word):
#                 if j<len(word)-1 and word[j]==max_bp[0] and word[j+1]==max_bp[1]:
#                     new_word.append(vocab[cu_vocab_index])
#                     j+=2
#                 else:
#                     new_word.append(word[j])
#                     j+=1
            
#             for b1,b2 in zip(word,word[1:]):
#                 pair=(b1,b2)
#                 if bp_counts.get(pair,0)!=0:
#                     bp_counts[pair]-=word_count
#                     if bp_counts[pair]==0:
#                         del bp_counts[pair] 
                
#             for b1,b2 in zip(new_word,new_word[1:]):
#                 pair=(b1,b2)
#                 if bp_pos.get(pair) is None:
#                     bp_pos[pair]=[i]
#                 elif i not in bp_pos.get(pair):
#                     bp_pos[pair].append(i) 
#                 bp_counts[pair]=bp_counts.get(pair,0)+word_count

#             words_list[i]=new_word
                          
                    
#         del bp_pos[max_bp]      #del merge pair
#         if bp_counts.get(max_bp) is not None:
#             del bp_counts[max_bp]   #del merge pair

#         cu_vocab_index+=1

#     return vocab,merges