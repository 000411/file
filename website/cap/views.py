from django.shortcuts import render
from flask import Flask, render_template, request
import sys
import re
from konlpy.tag import Komoran, Okt
import pandas as pd
import math
import numpy as np  
from sklearn.preprocessing import normalize
import random
import googletrans #구글번역 라이브러리
import spacy
import torch
import diffusers
import os
from diffusers import StableDiffusionPipeline
from PIL import Image
from .models import Image
from django.core.files import File
from cap.models import ImageModel
# import time

application = Flask(__name__)

@application.route("/")
def main(request):
    return render(request, 'main.html')

@application.route("/capstone/write")
def write(request):
    return render(request, '/capstone/write.html')

@application.route("/file")
def file(request):
    return render(request, 'file.html')


@application.route("/applyfile/", methods=['GET', 'POST'])
def applyfile(request):
    if request.method == 'POST':
        text_bytes = request.FILES['file'].read()
        text = text_bytes.decode('utf-8')
        text = re.sub(r"\n+", " ", text)  # 줄바꿈 없애기
        text = re.sub('"',"",text) #큰 따옴표 없애기
        sentences = re.split('[.?!]\s+', text)  #한문장 단위로 list 생성하기 : 구분 기준 . ? ! 

        #형태소 단위로 나누기(코모란 이용)

        komoran = Komoran()
        data = []
        for sentence in sentences: 
            temp_dict = dict() 
            temp_dict['sentence'] = sentence
            temp_dict['token_list'] = komoran.morphs(sentence) # 형태소 단위로 나누기

            data.append(temp_dict)

        df = pd.DataFrame(data) # DataFrame에 넣어 깔끔하게 보기
        print(df)

        # 문장들간의 유사도 계산
        similarity_matrix = []   # 빈 리스트 생성
        for i, row_i in df.iterrows(): # 유사도 그래프의 행 index
            i_row_vec = []  # i_row_vec가 유사도 그래프의 1행에 관한 모든 열 가짐 [열1, 열2, 열3, ...]
            for j, row_j in df.iterrows(): 
                    
                if i == j :
                    i_row_vec.append(0.0) # 같은 문장은 유사도 0으로 지정

                else:
                    intersection = len(set(row_i['token_list']) & set(row_j['token_list'])) # 문장 i와 문장 j에서 동시에 등장하는 단어의 수 
                    log_i = math.log(len(set(row_i['token_list']))) # 문장 i에 등장하는 단어의 수 
                    log_j = math.log(len(set(row_j['token_list']))) # 문장 j에 등장하는 단어의 수
                    if(log_i + log_j != 0) :
                      similarity = intersection / (log_i + log_j) # 유사도 계산 
                    if(similarity > 0.3) : # https://lovit.github.io/nlp/2019/04/30/textrank/ 에 따르면 min_sim이 0.3
                      i_row_vec.append(similarity) # min_sim  넘는 similarity만 edge로 설정 
                    else : 
                      i_row_vec.append(0.0) # 안넘으면 0으로 추가
            similarity_matrix.append(i_row_vec)  # 행 정보 추가 


        def pagerank(x, df=0.85, max_iter=100):
            assert 0 < df < 1   #assert(가정설정문) : assert는 뒤의 조건이 True가 아니면 AssertError 발생, 함수의 입력이 맞는지 확인하려고 

            # initialize
            A = normalize(x, axis=0, norm='l1') # pagerank 알고리즘의 페이지 중요도를 다 더해서 1로 만들어주는 작업
            R = np.ones(A.shape[0]).reshape(-1,1) # R : 각 문장의 rank값을 의미하고 맨처음은 1로 초기화
            bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1) # bias : 만족하지 못하고 페이지를 떠나는 확률로 (1 - damping factor)를 의미

            # iteration
            for _ in range(max_iter):  # 각 rank(score)는 weighted graph * rank*damping factor + (1-damping factor)로 계산한다.
             #(이론상 각 랭크 값이 어느정도 수렴할때까지 무한 루프를 돌리는데 이건 max iteration 만큼 동작.)
                R = df * (A * R) + bias

            return R
       

#         N = len(sentences)//3
        N = 10 

        weightedGraph = np.array(similarity_matrix) #데이터 타입 변환
        R = pagerank(weightedGraph) # pagerank를 돌려서 rank matrix 반환
        R = R.sum(axis=1) # 반환된 matrix를 row 별로 sum

        indexs = R.argsort()[-N:] # 해당 rank 값을 sort, 값이 높은 N개의 문장 index를 반환
        
        data_sent = []

        # rank값이 높은 문장을 프린트
        for index in sorted(indexs): # sorted 하는 이유는 원래 문장 순서에 맞춰 보여주기 위함
            temp_sent = dict()
            temp_sent['sentence'] = df['sentence'][index]
            data_sent.append(temp_sent)


        global df2
        df2 = pd.DataFrame(data_sent) # DataFrame에 넣어 깔끔하게 보기

        html = df2.to_html(index=False).replace('<table', '<table class="dataframe"')
        html = df2.to_html(justify='center')
        context={'결과': df2, 'html': html}

    return render(request, 'applyfile.html', context)

@application.route("/result_write/")
def result_write(request):
    
    if request.method == 'POST':
        if request.POST.get('change_type') == 'restart':
            sent = request.POST.get('sent') 
        else:
            sent = request.POST['sent']
    
    translator = googletrans.Translator() #번역기 객체 생성

    Outstr = translator.translate(sent, dest = 'en', src = 'ko') #번역
    eng_sent = Outstr.text
    
    def find_V(sent):
      nlp = spacy.load('en_core_web_sm')
      doc=nlp(sent)

      v_list=[]
      modi_v={}
      v_dic={}
      cnt=0

      for token in doc:
        if (token.pos_ == 'VERB') & (token.dep_ not in  ['xcomp','ccomp']):     #began to eat에서 eat 빼고 began 만 하려고 
          if (token.dep_ == 'conj') & (token.head.text not in v_list) & (token.head.pos_ != 'AUX'):  #수식어구에 있는 동사는 빼고 찾으려고 
            continue
          else:
            v_list.append(str(token.text))

      #동사 없을 때만 AUX찾아주기 
      if len(v_list) == 0:
        for token in doc:
          if token.pos_ in ['AUX','VERB']:   
            v_list.append(str(token.text))

     #수정할 부분 찾아두기 
      neg_case = [0 for i in range(len(v_list))]
      case_num = [0 for i in range(len(v_list))]
      for token in doc:
        for i in range(len(v_list)):
          #부정조사 있는 경우 
          if (token.dep_ == 'neg') & (str(token.head.text) == v_list[i]):
            neg_case[i] = 1
            case_num[i] = 1
            modi_v[v_list[i]] = v_list[i]

          #case1 : 수동태 수정
          elif (token.dep_ == 'auxpass') & (str(token.head.text) == v_list[i]) & (token.pos_ == 'AUX'):
            modi_v[v_list[i]] = str(token.text)+' '+v_list[i]
            case_num[i] = 1
          #case2 : 뒤에 부사/전치사 붙는 2글자짜리 숙어
          elif (token.dep_ in ['prt','advmod']) & (str(token.head.text) == v_list[i]) & (token.pos_ != 'SCONJ'):
            if (str(token.text)[-2:] != 'ly') & (str(token.text) not in ['So', 'Once']) :
              modi_v[v_list[i]] = v_list[i]+' '+str(token.text)
              case_num[i] = 2

      #동사 수정하기 
      for i in range(len(v_list)):
        if v_list[i] in modi_v:
          if neg_case[i] == 1:
            v_list[i] = ['not '+modi_v[v_list[i]], case_num[i]]
          else:
            v_list[i] = [modi_v[v_list[i]], case_num[i]]

        else:
          v_list[i] = [v_list[i], case_num[i]]

      return v_list

    #전치사 연결 명사 찾기
    def fine_conj(sent, n):
      nlp = spacy.load('en_core_web_sm')
      doc=nlp(sent)
      add_n =""

      for token in doc:
        if (token.pos_ == "NOUN") & (token.dep_ == "conj") & (str(token.head.text) == n):
          add_n = str(token.text)

      return add_n

    def modi_noun(sent, n, pos):
      nlp = spacy.load('en_core_web_sm')
      doc=nlp(sent)

      adj_dep = ['compound','amod','nummod', 'det']
      prep_dep = ['pobj']
      adj = []
      prep = []
      modi_n = ''
      sec_modi_n = ''

      for token in doc:
        if token.dep_ in adj_dep:
          adj.append(str(token.text))
        elif token.dep_ in prep_dep:
          prep.append(str(token.text))

      # 앞에 수식어구 붙여주기 
      if pos in ['NOUN','PROPN']:
        for token in doc:
          if str(token.text) == n:
            child = [str(child) for child in token.children if str(child) in adj]
            if child:
              modi_n = ' '.join(child)+' ' +str(token.text)
            else:
              modi_n = str(token.text)
      elif pos == 'ADP':
        for token in doc: 
          if (str(token.text) == n) & (token.head.pos_ in ['VERB', 'ADJ']):
            child = [str(child) for child in token.children if str(child) in prep]
            if child:
              new_n = modi_noun(sent, child[0], 'NOUN')
              modi_n = str(token.text)+' ' +new_n
            else:
              modi_n = str(token.text)
          elif (str(token.head.text) == n) & (token.pos_ == 'NOUN'):  #명사 + 전치사 + 명사일 때 
            modi_n = n+' '+str(token.text)
            return modi_n


      #뒤에 수식어구 붙여주기 : 앞 수식어(modi_n)도 포함해서 붙여주기

      for token in doc:
        if (str(token.head.text) == n) & (token.pos_ == 'ADP') & (token.dep_ == 'prep'):
          sec_modi_n = modi_n +' '+modi_noun(sent, str(token.text), 'ADP')

      if sec_modi_n == '':
        final_n = modi_n 
      else:
        final_n = sec_modi_n
        
      return final_n

    def key_phr(sent):
      nlp = spacy.load('en_core_web_sm')
      doc=nlp(sent)

      #반복되는 단어 찾기 
      word_n = []
      word_v = []
      word_sp = []
      word_det = []
      count_n = 0
      count_v = 0
      count_sp = 0
      count_det = 0
      for token in doc:
        if token.pos_ == 'NOUN':
          if str(token.text) in word_n:
            dup_n = str(token.text)
            count_n += 1 
          else:
            word_n.append(str(token.text))
        elif token.pos_ in ['AUX','VERB']:
          if str(token.text) in word_v:
            dup_v = str(token.text)
            count_v += 1 
          else:
            word_v.append(str(token.text))
        elif token.pos_ in ['DET']:
          if str(token.text) in word_det:
            dup_det = str(token.text)
            count_det += 1 
          else:
            word_det.append(str(token.text))  
        else :
          if str(token.text) in word_sp :
            dup_sp = str(token.text)
            count_sp += 1 
          else:
            word_sp.append(str(token.text))

      #반복되는 단어 구분 (ex side라는 단어가 2개면, side1, side2)
      #명사 구분 
      if count_n > 0:
        change_n = dup_n
        max_n = count_n+1
        while max_n != 0:
          sent = sent.replace(change_n, dup_n+str(max_n), max_n)
          change_n = dup_n+str(max_n)
          max_n -= 1
      #동사 구분  
      if count_v > 0:
        change_v = dup_v
        max_v = count_v+1
        while max_v != 0:
          sent = sent.replace(change_v, dup_v+str(max_v), max_v)
          change_v = dup_v+str(max_v)
          max_v -= 1
      #det 구분 
      if count_det > 0:
        change_det = dup_det
        max_det = count_det+1
        while max_det != 0:
          sent = sent.replace(change_det, dup_det+str(max_det), max_det)
          change_det = dup_det+str(max_det)
          max_det -= 1
      #다른 단어 구분
      if count_sp > 0:
        change_sp = dup_sp
        max_sp = count_sp+1
        while max_sp != 0:
          sent = sent.replace(change_sp, dup_sp+str(max_sp), max_sp)
          change_sp = dup_sp+str(max_sp)
          max_sp -= 1

      #동사 찾기 
      v_list=find_V(sent) 
      if v_list==[]:
        sent = re.sub(r"[0-9]", "", sent)
        v_list=find_V(sent) 

      child_list=[]
      num = 1
      s=""
      o=""
      c=""
      spare_v=""
      spare_n=""
      mark=""
      add_s=""
      add_c=""
      add_o=""
      add_I_o=""
      add_spare_n=""


      phrase = [0 for i in range(len(v_list))] 
      for i in range(len(v_list)):
        # 숙어 케이스일 경우 : 주어는 앞에 동사로 찾고 목적어는 뒤에 부사/전치사로 찾는다 
        if v_list[i][1] == 2:    
          temp = v_list[i][0].split()
          v1 = temp[0]
          v2 = temp[1] 
        # 수동태 케이스의 경우 : 둘다 뒤에 있는 동사로 찾아준다 
        elif (v_list[i][1] == 1): 
          temp = v_list[i][0].split()
          v1 = temp[-1]
          v2 = temp[-1] 
        else : 
          v1 = v_list[i][0]
          v2 = v_list[i][0]

        doc=nlp(sent)
        for token in doc:
          #주어 찾기 
          if (token.dep_ in ['nsubjpass','nsubj']) & (str(token.head.text) == v1): 
            s = modi_noun(sent,str(token.text),token.pos_) #수식어구 
            add_s = fine_conj(sent, str(token.text))
          elif (str(token.text) == v1) & (token.dep_ == 'relcl') & (s == ''):
            s = modi_noun(sent,str(token.head.text),token.head.pos_) #수식어구 
            add_s = fine_conj(sent, str(token.text))

          #보어 찾기    
          elif (token.dep_ in ['attr', 'oprd']) & (str(token.head.text) == v2):  
            num = 2  #2형식 
            c = modi_noun(sent,str(token.text),token.pos_) #수식어구 
            add_c = fine_conj(sent, str(token.text))
          elif (token.dep_ == 'acomp') & (str(token.head.text) == v2) & (token.head.pos_ == 'AUX'): 
            num = 2  #2형식 
            mark = str(token.text)

          #목적어 찾기 
          if (token.dep_ in ['pobj','dobj']) & (str(token.head.text) == v2): 
            num = 3  #3형식
            o = modi_noun(sent,str(token.text),token.pos_) #수식어구 
            add_o = fine_conj(sent, str(token.text))
          elif (token.dep_ in ['prep','agent']) & (str(token.head.text) == v2) & (o == ''):
            num = 3 #3형식
            o = modi_noun(sent, str(token.text),token.pos_) #수식어구 
            add_o = fine_conj(sent, str(token.text))

          #간접목적어 찾기  
          elif (token.dep_ == 'dative') & (str(token.head.text) == v2): 
            num = 4
            I_o = modi_noun(sent,str(token.text),token.pos_) #수식어구 
            add_I_o = fine_conj(sent, str(token.text))

          #보어구 찾기
          elif (token.dep_ in ['xcomp','ccomp']) & (str(token.head.text) == v2):
            spare_v = str(token.text)
            num = 5

          if spare_v != "":
            if (token.dep_ == 'dobj') & (str(token.head.text) == spare_v): 
              spare_n = modi_noun(sent,str(token.text),token.pos_) #수식어구 
              add_spare_n = fine_conj(sent, str(token.text))
            elif (token.dep_ in ['acomp']) & (str(token.head.text) == spare_v):
              spare_n = modi_noun(sent,str(token.text),token.pos_) #수식어구 
              add_spare_n = fine_conj(sent, str(token.text))

          if mark != "":    
            if (token.dep_ in ['prep','agent']) & (str(token.head.text) == mark) & (num == 2):
              c = mark+' '+modi_noun(sent, str(token.text),token.pos_) #수식어구 
              add_c= fine_conj(sent, str(token.text))

        print('num:',num)
        #명사 결정
        if add_s != "":
          s += " and "+add_s
        if add_c != "":
          c += " and "+add_c
        if add_o != "":
          o += " and "+add_o
        if add_I_o != "":
          I_o += " and "+add_I_o
        if add_spare_n != "":
          spare_n += " and "+add_spare_n


        if num == 1:
          phrase[i] = re.sub(r"[0-9]", "", s +' who '+v_list[i][0])
        elif num == 2:
          phrase[i] = re.sub(r"[0-9]", "", s +' who '+v_list[i][0]+' '+c)
        elif num == 3:  
          phrase[i] = re.sub(r"[0-9]", "", s+' who '+v_list[i][0]+' '+o)  
        elif num == 4:
          phrase[i] = re.sub(r"[0-9]", "", s+' who '+v_list[i][0]+' '+o+' to '+I_o)  
        elif num == 5:
          phrase[i] = re.sub(r"[0-9]", "", s+' who '+v_list[i][0]+' to '+spare_v+' '+spare_n )  


      return phrase
    
    phrase = key_phr(eng_sent)
    
    if request.method != 'POST':
        def dec_phrase(phrase):
          error = 0
          result = ' '.join(s for s in phrase)
          if len(result.split())<=3 :
            global df2
            df2 = df2.drop(i, axis=0) # 해당 행 삭제
            error=1

          return error

        error = dec_phrase(phrase)

        while error == 1:
            #다시 구하기 
            sent = df2.iloc[i]['sentence']
            Outstr = translator.translate(sent, dest = 'en', src = 'ko') #번역
            eng_sent = Outstr.text
            phrase = key_phr(eng_sent)
            error = dec_phrase(phrase)

    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    
    num_inference_steps = 200

    prompt = phrase

    print(sent)
    print(eng_sent)
    print(prompt)
    Negative_prompt = "((NSFW)),strange face, lowres, (bad anatomy, bad hands:1.1), text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, blurry, artist name, b&w, weird colors, (cartoon, 3d, bad art, poorly drawn, close up, word, blurry:1.5), (disfigured, deformed, extra limbs:1.5)"
    image = pipe(prompt).images[0]
#     image.show()    
    image.save('./cap/static/cap/image_result.png')
#     time.sleep(15)
 
#     return render(request, 'result_write.html', {'image' : image, 'sent' : sent, 'eng_sent' : eng_sent, 'prompt' : prompt})   
    return render(request, 'result_write.html', {'image' : image, 'sent' : sent})   

@application.route("/applydiff")
def applydiff(request):    
    global i

    if request.method == "POST":
        print(request.POST.get('change_type'))
        if request.POST.get('change_type') == 'restart':
            i += 0
        elif request.POST.get('change_type') == 'save':
            i += 1
            if i >= 10:
                i = int(request.POST.get('i_value'))
                sent_0 = request.POST.get('sent_0')
                sent_1 = request.POST.get('sent_1')
                sent_2 = request.POST.get('sent_2')
                sent_3 = request.POST.get('sent_3')
                sent_4 = request.POST.get('sent_4')
                sent_5 = request.POST.get('sent_5')
                sent_6 = request.POST.get('sent_6')
                sent_7 = request.POST.get('sent_7')
                sent_8 = request.POST.get('sent_8')
                sent_9 = request.POST.get('sent_9')
                sent = request.POST.get('sent')
                return render(request, 'end.html', {'sent' : sent, 'i' : i, 'sent_0' : sent_0, 'sent_1' : sent_1, 'sent_2' : sent_2, 'sent_3' : sent_3, 'sent_4' : sent_4, 'sent_5' : sent_5, 'sent_6' : sent_6, 'sent_7' : sent_7, 'sent_8' : sent_8, 'sent_9' : sent_9})
    else:
        i=0

    sent = df2.iloc[i]['sentence']
    sents_10 = df2[df2.columns[0]].to_list()
    
    translator = googletrans.Translator() #번역기 객체 생성

    Outstr = translator.translate(sent, dest = 'en', src = 'ko') #번역
    eng_sent = Outstr.text
    
    def find_V(sent):
      nlp = spacy.load('en_core_web_sm')
      doc=nlp(sent)

      v_list=[]
      modi_v={}
      v_dic={}
      cnt=0

      for token in doc:
        if (token.pos_ == 'VERB') & (token.dep_ not in  ['xcomp','ccomp']):     #began to eat에서 eat 빼고 began 만 하려고 
          if (token.dep_ == 'conj') & (token.head.text not in v_list) & (token.head.pos_ != 'AUX'):  #수식어구에 있는 동사는 빼고 찾으려고 
            continue
          else:
            v_list.append(str(token.text))

      #동사 없을 때만 AUX찾아주기 
      if len(v_list) == 0:
        for token in doc:
          if token.pos_ in ['AUX','VERB']:   
            v_list.append(str(token.text))

     #수정할 부분 찾아두기 
      neg_case = [0 for i in range(len(v_list))]
      case_num = [0 for i in range(len(v_list))]
      for token in doc:
        for i in range(len(v_list)):
          #부정조사 있는 경우 
          if (token.dep_ == 'neg') & (str(token.head.text) == v_list[i]):
            neg_case[i] = 1
            case_num[i] = 1
            modi_v[v_list[i]] = v_list[i]

          #case1 : 수동태 수정
          elif (token.dep_ == 'auxpass') & (str(token.head.text) == v_list[i]) & (token.pos_ == 'AUX'):
            modi_v[v_list[i]] = str(token.text)+' '+v_list[i]
            case_num[i] = 1
          #case2 : 뒤에 부사/전치사 붙는 2글자짜리 숙어
          elif (token.dep_ in ['prt','advmod']) & (str(token.head.text) == v_list[i]) & (token.pos_ != 'SCONJ'):
            if (str(token.text)[-2:] != 'ly') & (str(token.text) not in ['So', 'Once']) :
              modi_v[v_list[i]] = v_list[i]+' '+str(token.text)
              case_num[i] = 2

      #동사 수정하기 
      for i in range(len(v_list)):
        if v_list[i] in modi_v:
          if neg_case[i] == 1:
            v_list[i] = ['not '+modi_v[v_list[i]], case_num[i]]
          else:
            v_list[i] = [modi_v[v_list[i]], case_num[i]]

        else:
          v_list[i] = [v_list[i], case_num[i]]

      return v_list

    #전치사 연결 명사 찾기
    def fine_conj(sent, n):
      nlp = spacy.load('en_core_web_sm')
      doc=nlp(sent)
      add_n =""

      for token in doc:
        if (token.pos_ == "NOUN") & (token.dep_ == "conj") & (str(token.head.text) == n):
          add_n = str(token.text)

      return add_n

    def modi_noun(sent, n, pos):
      nlp = spacy.load('en_core_web_sm')
      doc=nlp(sent)

      adj_dep = ['compound','amod','nummod', 'det']
      prep_dep = ['pobj']
      adj = []
      prep = []
      modi_n = ''
      sec_modi_n = ''

      for token in doc:
        if token.dep_ in adj_dep:
          adj.append(str(token.text))
        elif token.dep_ in prep_dep:
          prep.append(str(token.text))

      # 앞에 수식어구 붙여주기 
      if pos in ['NOUN','PROPN']:
        for token in doc:
          if str(token.text) == n:
            child = [str(child) for child in token.children if str(child) in adj]
            if child:
              modi_n = ' '.join(child)+' ' +str(token.text)
            else:
              modi_n = str(token.text)
      elif pos == 'ADP':
        for token in doc: 
          if (str(token.text) == n) & (token.head.pos_ in ['VERB', 'ADJ']):
            child = [str(child) for child in token.children if str(child) in prep]
            if child:
              new_n = modi_noun(sent, child[0], 'NOUN')
              modi_n = str(token.text)+' ' +new_n
            else:
              modi_n = str(token.text)
          elif (str(token.head.text) == n) & (token.pos_ == 'NOUN'):  #명사 + 전치사 + 명사일 때 
            modi_n = n+' '+str(token.text)
            return modi_n


      #뒤에 수식어구 붙여주기 : 앞 수식어(modi_n)도 포함해서 붙여주기

      for token in doc:
        if (str(token.head.text) == n) & (token.pos_ == 'ADP') & (token.dep_ == 'prep'):
          sec_modi_n = modi_n +' '+modi_noun(sent, str(token.text), 'ADP')

      if sec_modi_n == '':
        final_n = modi_n 
      else:
        final_n = sec_modi_n
        
      return final_n

    def key_phr(sent):
      nlp = spacy.load('en_core_web_sm')
      doc=nlp(sent)

      #반복되는 단어 찾기 
      word_n = []
      word_v = []
      word_sp = []
      word_det = []
      count_n = 0
      count_v = 0
      count_sp = 0
      count_det = 0
      for token in doc:
        if token.pos_ == 'NOUN':
          if str(token.text) in word_n:
            dup_n = str(token.text)
            count_n += 1 
          else:
            word_n.append(str(token.text))
        elif token.pos_ in ['AUX','VERB']:
          if str(token.text) in word_v:
            dup_v = str(token.text)
            count_v += 1 
          else:
            word_v.append(str(token.text))
        elif token.pos_ in ['DET']:
          if str(token.text) in word_det:
            dup_det = str(token.text)
            count_det += 1 
          else:
            word_det.append(str(token.text))  
        else :
          if str(token.text) in word_sp :
            dup_sp = str(token.text)
            count_sp += 1 
          else:
            word_sp.append(str(token.text))

      #반복되는 단어 구분 (ex side라는 단어가 2개면, side1, side2)
      #명사 구분 
      if count_n > 0:
        change_n = dup_n
        max_n = count_n+1
        while max_n != 0:
          sent = sent.replace(change_n, dup_n+str(max_n), max_n)
          change_n = dup_n+str(max_n)
          max_n -= 1
      #동사 구분  
      if count_v > 0:
        change_v = dup_v
        max_v = count_v+1
        while max_v != 0:
          sent = sent.replace(change_v, dup_v+str(max_v), max_v)
          change_v = dup_v+str(max_v)
          max_v -= 1
      #det 구분 
      if count_det > 0:
        change_det = dup_det
        max_det = count_det+1
        while max_det != 0:
          sent = sent.replace(change_det, dup_det+str(max_det), max_det)
          change_det = dup_det+str(max_det)
          max_det -= 1
      #다른 단어 구분
      if count_sp > 0:
        change_sp = dup_sp
        max_sp = count_sp+1
        while max_sp != 0:
          sent = sent.replace(change_sp, dup_sp+str(max_sp), max_sp)
          change_sp = dup_sp+str(max_sp)
          max_sp -= 1

      #동사 찾기 
      v_list=find_V(sent) 
      if v_list==[]:
        sent = re.sub(r"[0-9]", "", sent)
        v_list=find_V(sent) 

      child_list=[]
      num = 1
      s=""
      o=""
      c=""
      spare_v=""
      spare_n=""
      mark=""
      add_s=""
      add_c=""
      add_o=""
      add_I_o=""
      add_spare_n=""


      phrase = [0 for i in range(len(v_list))] 
      for i in range(len(v_list)):
        # 숙어 케이스일 경우 : 주어는 앞에 동사로 찾고 목적어는 뒤에 부사/전치사로 찾는다 
        if v_list[i][1] == 2:    
          temp = v_list[i][0].split()
          v1 = temp[0]
          v2 = temp[1] 
        # 수동태 케이스의 경우 : 둘다 뒤에 있는 동사로 찾아준다 
        elif (v_list[i][1] == 1): 
          temp = v_list[i][0].split()
          v1 = temp[-1]
          v2 = temp[-1] 
        else : 
          v1 = v_list[i][0]
          v2 = v_list[i][0]

        doc=nlp(sent)
        for token in doc:
          #주어 찾기 
          if (token.dep_ in ['nsubjpass','nsubj']) & (str(token.head.text) == v1): 
            s = modi_noun(sent,str(token.text),token.pos_) #수식어구 
            add_s = fine_conj(sent, str(token.text))
          elif (str(token.text) == v1) & (token.dep_ == 'relcl') & (s == ''):
            s = modi_noun(sent,str(token.head.text),token.head.pos_) #수식어구 
            add_s = fine_conj(sent, str(token.text))

          #보어 찾기    
          elif (token.dep_ in ['attr', 'oprd']) & (str(token.head.text) == v2):  
            num = 2  #2형식 
            c = modi_noun(sent,str(token.text),token.pos_) #수식어구 
            add_c = fine_conj(sent, str(token.text))
          elif (token.dep_ == 'acomp') & (str(token.head.text) == v2) & (token.head.pos_ == 'AUX'): 
            num = 2  #2형식 
            mark = str(token.text)

          #목적어 찾기 
          if (token.dep_ in ['pobj','dobj']) & (str(token.head.text) == v2): 
            num = 3  #3형식
            o = modi_noun(sent,str(token.text),token.pos_) #수식어구 
            add_o = fine_conj(sent, str(token.text))
          elif (token.dep_ in ['prep','agent']) & (str(token.head.text) == v2) & (o == ''):
            num = 3 #3형식
            o = modi_noun(sent, str(token.text),token.pos_) #수식어구 
            add_o = fine_conj(sent, str(token.text))

          #간접목적어 찾기  
          elif (token.dep_ == 'dative') & (str(token.head.text) == v2): 
            num = 4
            I_o = modi_noun(sent,str(token.text),token.pos_) #수식어구 
            add_I_o = fine_conj(sent, str(token.text))

          #보어구 찾기
          elif (token.dep_ in ['xcomp','ccomp']) & (str(token.head.text) == v2):
            spare_v = str(token.text)
            num = 5

          if spare_v != "":
            if (token.dep_ == 'dobj') & (str(token.head.text) == spare_v): 
              spare_n = modi_noun(sent,str(token.text),token.pos_) #수식어구 
              add_spare_n = fine_conj(sent, str(token.text))
            elif (token.dep_ in ['acomp']) & (str(token.head.text) == spare_v):
              spare_n = modi_noun(sent,str(token.text),token.pos_) #수식어구 
              add_spare_n = fine_conj(sent, str(token.text))

          if mark != "":    
            if (token.dep_ in ['prep','agent']) & (str(token.head.text) == mark) & (num == 2):
              c = mark+' '+modi_noun(sent, str(token.text),token.pos_) #수식어구 
              add_c= fine_conj(sent, str(token.text))

        print('num:',num)
        #명사 결정
        if add_s != "":
          s += " and "+add_s
        if add_c != "":
          c += " and "+add_c
        if add_o != "":
          o += " and "+add_o
        if add_I_o != "":
          I_o += " and "+add_I_o
        if add_spare_n != "":
          spare_n += " and "+add_spare_n


        if num == 1:
          phrase[i] = re.sub(r"[0-9]", "", s +' who '+v_list[i][0])
        elif num == 2:
          phrase[i] = re.sub(r"[0-9]", "", s +' who '+v_list[i][0]+' '+c)
        elif num == 3:  
          phrase[i] = re.sub(r"[0-9]", "", s+' who '+v_list[i][0]+' '+o)  
        elif num == 4:
          phrase[i] = re.sub(r"[0-9]", "", s+' who '+v_list[i][0]+' '+o+' to '+I_o)  
        elif num == 5:
          phrase[i] = re.sub(r"[0-9]", "", s+' who '+v_list[i][0]+' to '+spare_v+' '+spare_n )  


      return phrase
    
    phrase = key_phr(eng_sent)
    
    if request.method != 'POST':
        def dec_phrase(phrase):
          error = 0
          result = ' '.join(s for s in phrase)
          if len(result.split())<=3 :
            global df2
            df2 = df2.drop(i, axis=0) # 해당 행 삭제
            error=1

          return error

        error = dec_phrase(phrase)

        while error == 1:
            #다시 구하기 
            sent = df2.iloc[i]['sentence']
            Outstr = translator.translate(sent, dest = 'en', src = 'ko') #번역
            eng_sent = Outstr.text
            phrase = key_phr(eng_sent)
            error = dec_phrase(phrase)

    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    
    num_inference_steps = 200

    prompt = phrase

    print(sent)
    print(eng_sent)
    print(prompt)
    Negative_prompt = "((NSFW)),strange face, ((bad anatomy)), ((text)), extra digit, fewer digits, cropped, worst quality, jpeg artifacts, weird colors, bad art, poorly drawn, close up"
    image = pipe(prompt).images[0]
#     image.show()    
    image.save('./cap/static/cap/image_result.png')
    image.save('./cap/static/cap/image_result%d.png' %i)
    sent_0 = sents_10[0]
    sent_1 = sents_10[1]
    sent_2 = sents_10[2]
    sent_3 = sents_10[3]
    sent_4 = sents_10[4]
    sent_5 = sents_10[5]
    sent_6 = sents_10[6]
    sent_7 = sents_10[7]
    sent_8 = sents_10[8]
    sent_9 = sents_10[9]    
#     time.sleep(15)

#     return render(request, 'applydiff.html', {'image' : image, 'sent' : sent, 'eng_sent' : eng_sent, 'prompt' : prompt, 'i' : i})
    return render(request, 'applydiff.html', {'image' : image, 'sent' : sent, 'i' : i, 'sent_0' : sent_0, 'sent_1' : sent_1, 'sent_2' : sent_2, 'sent_3' : sent_3, 'sent_4' : sent_4, 'sent_5' : sent_5, 'sent_6' : sent_6, 'sent_7' : sent_7, 'sent_8' : sent_8, 'sent_9' : sent_9})

@application.route("/cartoon/")
def cartoon(request):
    if request.method == "POST":
        print(request.POST.get('i_value'))
        i = int(request.POST.get('i_value'))
        sent_0 = request.POST.get('sent_0')
        sent_1 = request.POST.get('sent_1')
        sent_2 = request.POST.get('sent_2')
        sent_3 = request.POST.get('sent_3')
        sent_4 = request.POST.get('sent_4')
        sent_5 = request.POST.get('sent_5')
        sent_6 = request.POST.get('sent_6')
        sent_7 = request.POST.get('sent_7')
        sent_8 = request.POST.get('sent_8')
        sent_9 = request.POST.get('sent_9')
        print(sent_0)
        return render(request, 'cartoon.html', {'i' : i, 'sent_0' : sent_0, 'sent_1' : sent_1, 'sent_2' : sent_2, 'sent_3' : sent_3, 'sent_4' : sent_4, 'sent_5' : sent_5, 'sent_6' : sent_6, 'sent_7' : sent_7, 'sent_8' : sent_8, 'sent_9' : sent_9})

@application.route("/end")
def end(request):  
    return render(request, 'end.html' , {'i' : i, 'sent_0' : sent_0, 'sent_1' : sent_1, 'sent_2' : sent_2, 'sent_3' : sent_3, 'sent_4' : sent_4, 'sent_5' : sent_5, 'sent_6' : sent_6, 'sent_7' : sent_7, 'sent_8' : sent_8, 'sent_9' : sent_9})

if __name__ == "__main__":
    application.run(host='0.0.0.0')
