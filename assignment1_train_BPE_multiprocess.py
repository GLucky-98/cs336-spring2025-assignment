# 16 process 64 chunk
# cpu:12600kf
# RAM:32G
# Tiny Story train
# vocab:10000
# pre_tokenization time:23.687570333480835
# merge time: 235.42702770233154
# merge speed: 41.38437330279182 / s
# owt_train
# vocab:32000
#
#
#

import regex as re
from collections import Counter
import os
import multiprocessing
from typing import BinaryIO
import time
import json
import pickle

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_process=1,
    num_chunk=1,
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
    # multi processing
    split_special_token=b'<|endoftext|>'

    # read file
    with open(input_path,'rb') as f:
        boundaries=find_chunk_boundaries(f,num_chunk,split_special_token) #str

    # construct multiprocess task
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # words={}
    # words_count={}
    # bp_count={}
    # for start,end in zip(boundaries,boundaries[1:]):
    #     chunk_words,chunk_words_count,chunk_bp_count=process_worker((input_path,start,end,PAT,special_tokens))
    #     words.update(chunk_words)
    #     words_count.update(chunk_words_count)
    #     bp_count.update(chunk_bp_count)

    start_time=time.time()

    start_end=[(i,j) for i,j in zip(boundaries,boundaries[1:])]
    words={}
    words_count={}
    bp_count={}
    iters=num_chunk//num_process
    for i in range(iters):
        tasks=[]
        for j in range(num_process):
            args=(input_path,start_end[i*num_process+j][0],start_end[i*num_process+j][1],PAT,special_tokens)
            tasks.append(args)
        with multiprocessing.Pool(processes=num_process) as pool:
            results = pool.map(process_worker, tasks)
        for result in results:
            chunk_words,chunk_words_count,chunk_bp_count=result
            words.update(chunk_words)
            words_count.update(chunk_words_count)
            bp_count.update(chunk_bp_count)
    
    end_time=time.time()
    print(f'pre_tokenization time:{end_time-start_time}')
 
    #initial vocab & merges 
    vocab={i:bytes([i]) for i in range(256)} # int:bytes
    
    cu_vocab_index=256 #current vocab size
    
    # special tokens from index 256
    for st in special_tokens:
        vocab[cu_vocab_index]=st.encode('utf-8')
        cu_vocab_index+=1

    merges=[]
    
    start_time=time.time()
    # find -> merge -> update vocab  & merge  & words & bp_count
    while cu_vocab_index<vocab_size:
        if not bp_count:
            break

        # find max byte pair counts
        max_bp=max(bp_count.items(), key=lambda item: (item[1], item[0]))[0] # the most important part to pass the test because of the 'lexicographically'
        # print(max_bp)
        # add max_pair to merges and vocab 
        merges.append(max_bp)
        vocab[cu_vocab_index]=max_bp[0]+max_bp[1]

        words,bp_count=merge_pair(words,words_count,bp_count,max_bp)
        

        cu_vocab_index+=1

    end_time=time.time()
    print(f'merge time: {end_time-start_time}')
    print(f'merge speed: {len(merges)/(end_time-start_time)} / s')

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

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_worker(args):
    
    input_path,start,end,PAT,special_tokens=args

    with open(input_path,'rb') as f:
        f.seek(start)
        chunk_length=end-start
        binary_data=f.read(chunk_length)
        text=binary_data.decode('utf-8',errors='ignore')

    pre_tokens=pre_tokenization(text,PAT,special_tokens)

    words,words_count,bp_count=get_count(pre_tokens)

    return (words,words_count,bp_count)

def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

if __name__ == '__main__':
        
    vocab,merges=run_train_bpe(
            input_path='/home/gl/Desktop/cs336/assignment1-basics-main/data/owt_train.txt',
            vocab_size=32000,
            special_tokens=["<|endoftext|>"],
            num_process=16,
            num_chunk=512
        )
    
    # save file for encoder and decoder 
    with open('vocab1.pkl','wb') as f:
        pickle.dump(vocab,f)
    
    with open('merges1.pkl','wb') as f:
        pickle.dump(merges,f)
    

    # save file for people reading 
    d=gpt2_bytes_to_unicode()

    for k,v in vocab.items():
        v=''.join([d[i] for i in v])
        vocab[k]=v

    new_merges=[]
    for merge in merges:
        i,j=merge
        i=''.join([d[k] for k in i])
        j=''.join([d[k] for k in j])
        new_merges.append([i,j])
    
    merges=new_merges


    with open('vocab1.json','w',encoding='utf-8') as f:
        json.dump(vocab,f,ensure_ascii=False,indent=4)

    with open('merges1.json','w',encoding='utf-8') as f:
        json.dump(merges,f,ensure_ascii=False,indent=0)








