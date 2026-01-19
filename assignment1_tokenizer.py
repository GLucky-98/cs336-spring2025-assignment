from collections.abc import Iterator, Iterable
import pickle
import os,json
import regex as re
import tiktoken
import time

class tokenizer():
    def __init__(self,vocab,merges,special_tokens=None):
        self.vocab=vocab # eg: {996: b'oun', 997: b'ren'}
        self.special_tokens=special_tokens
        if self.special_tokens:
            for special_token in self.special_tokens:
                if special_token.encode('utf-8') not in vocab.values():
                    vocab[len(vocab)]=special_token.encode('utf-8')
        self.byte2id={ v:k for k,v in vocab.items()}
        
        # input merges eg: [(b' ',b't'),(b' th',b'e')]
        # construct self.merges for encoder eg:{(b' ',b't'):1,(b' th',b'e'):2}
        self.merges={}
        for index,merge in enumerate(merges):
            self.merges[merge]=index
        

    
    @classmethod
    def from_files(cls,vocab_path,merges_path,special_tokens=None):
        # read the pkl file 
        with open(vocab_path,'rb') as file:
            vocab=pickle.load(file)
        with open(merges_path,'rb') as file:
            merges=pickle.load(file)
        
        return cls(vocab,merges,special_tokens)

    def encode(self,text:str) -> list[int]:
        
        def pre_tokenizer(text:str,special_tokens:list[str]):
            PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

            if self.special_tokens:
                split_special_tokens=sorted(list(map(re.escape,special_tokens)),key=len,reverse=True)

                text=re.split(f'({'|'.join(split_special_tokens)})',text) # list[str]  capture group pattern
            
            pre_tokens=[]
            if isinstance(text,list):
                for t in text:
                    if self.special_tokens and t in self.special_tokens:
                        pre_tokens.append(t)
                    else:
                        pre_tokens.extend(re.findall(PAT,t))
            else:
                pre_tokens=re.findall(PAT,text)

            return pre_tokens
                   
        def get_bp_set(tokens:list[int]) -> list[tuple[int,int]]:
            bp_set=[]
            for b1,b2 in zip(tokens,tokens[1:]):
                pair=(b1,b2)
                if pair not in bp_set:
                    bp_set.append(pair)
            return bp_set

        def merge_pair(tokens:list[int],pair:tuple[int,int],id) -> list[int]:
            i=0
            new_tokens=[]
            while i<len(tokens):
                if i<len(tokens)-1 and tokens[i]==pair[0] and tokens[i+1]==pair[1]:
                    new_tokens.append(id)
                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
            return new_tokens 
        
        pretokens=pre_tokenizer(text,self.special_tokens)


        def find_merge_index(p:list[int,int]):
            merge=(self.vocab[p[0]],self.vocab[p[1]])
            x=self.merges.get(merge,float('inf'))
            return x
        
        
        tokens=[]
        for token in pretokens:
            if self.special_tokens and token in self.special_tokens:
                tokens.append(self.byte2id[token.encode('utf-8')])
            else:
                token=token.encode('utf-8')
                token=[self.byte2id[bytes([b])] for b in token]
                while len(token)>1:
                    bp_set=get_bp_set(token)
                    pair=min(bp_set,key=find_merge_index)
                    if (self.vocab[pair[0]],self.vocab[pair[1]]) not in self.merges:
                        break
                    id=self.byte2id[self.vocab[pair[0]]+self.vocab[pair[1]]]
                    token=merge_pair(token,pair,id)
                tokens.extend(token)
        return tokens
    
    def encode_iterable(self,iterable:Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            if not chunk:
                continue
            
            yield from self.encode(chunk)
    
    def decode(self,ids:list[int]) -> str:
        text_b=b''.join([self.vocab[id] for id in ids])
        text=text_b.decode('utf-8',errors='replace')
        return text

VOCAB_PATH = './data/vocab.pkl'

MERGES_PATH ='./data/merges.pkl' 

tk=tokenizer.from_files(VOCAB_PATH,MERGES_PATH)

with open('../data/TinyStoriesV2-GPT4-valid.txt') as f:
    text=f.read()

start_time=time.time()

tokens=tk.encode(text)

end_time=time.time()

with open('./data/tokens.pkl','wb') as f:
    pickle.dump(tokens,f)

print(f'encode speed {len(tokens)/(end_time-start_time)} byte/s')