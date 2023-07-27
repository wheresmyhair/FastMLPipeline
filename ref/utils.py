import numpy as np
import pandas as pd
import jieba.analyse
import xgboost as xgb
import pickle

from tqdm import tqdm
from typing import Tuple
from os import makedirs
from os.path import (join, exists)
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional, Union, List, Dict, Any, Callable, Iterable, Set


def extract_global_topwords(df, topK:int=20, allowPOS:tuple=(), cust_vocab=None):
    '''
    ## Desc
        返回topK词语dataframe,每一列为一个词语,每一行为一个文本,值为该词语的权重
        总会返回topk个词
        如: 只返回v,则是返回top20的动词
    ## Args
        df: 一维文本list
        allowPOS: 词性过滤,如allowPOS=('ns', 'n', 'vn', 'v')，只保留地名、名词、动名词、动词
    ## Return
        pd.DataFrame
    '''
    if cust_vocab is not None:
        with open('./vocab_spm.txt','r',encoding='utf-8') as f:
            vocab = f.readlines()
        vocab = [v.strip() for v in vocab]
        for idx,v in enumerate(vocab):
            try:
                jieba.add_word(v.split(' ')[0], freq=int(v.split(' ')[1]))
            except:
                print(f'[FAIL] Failed to load vocab {idx}: {v}')
    top_words = []
    for i in tqdm(range(len(df))):
        top_words.append(dict(jieba.analyse.extract_tags(df[i], topK=topK, allowPOS=allowPOS, withWeight=True)))
    return pd.DataFrame(top_words).fillna(0)


class Preprocess:
    def __init__(self, df_train, df_test, colname_target, colname_manual_features, encode_y=True, custom_vocab=None) -> None:
        self.df_train_raw = df_train
        self.df_test_raw = df_test
        self.df_test = df_test[colname_manual_features]
        self.df_train_x = self.df_train_raw[colname_manual_features]
        self.df_train_y = self.df_train_raw[colname_target]
        self.encody_y = encode_y
        self.nlp_features = {}

        if custom_vocab is not None:
            self.__init_vocabdict()
            print(f'[INFO] Loading custom dict: {custom_vocab}')
            jieba.load_userdict(custom_vocab)

        if self.encody_y:
            self.le = LabelEncoder()
            self.df_train_y = self.le.fit_transform(self.df_train_y)


    @staticmethod
    def __init_vocabdict():
        print(f'[INFO] Init custom dict')
        with open('../data_0512/vocab/keywords.txt', 'r', encoding='utf-8') as f:
            keywords = f.readlines()
        keywords = [x.strip() for x in keywords]
        keywords = list(set(keywords))
        with open('./vocabs/vocab_custom.txt', 'w', encoding='utf-8') as f:
            for kw in keywords:
                f.write(kw + ' 999\n')
                

    def __get_sentence_vector(self, cutted_sentence:list):
        sentence_vectors = []
        for word in cutted_sentence:
            try:
                word_vector = self.model_w2v.wv[word]
                sentence_vectors.append(word_vector)
            except KeyError:
                # ignore out-of-vocabulary words
                pass
        # average the word embeddings to create a sentence vector
        if sentence_vectors:
            sentence_vector = sum(sentence_vectors) / len(sentence_vectors)
            return sentence_vector
        else:
            return np.array([0]*self.model_w2v.vector_size)

                
    def topk(self, colnames_nlp:list, topk, allowPOS=(), cust_vocab=None):
        for col in colnames_nlp:
            df_kw = extract_global_topwords(
                pd.concat([self.df_train_raw[col], self.df_test_raw[col]], axis=0, ignore_index=True), 
                topK=topk, 
                allowPOS=allowPOS,
                cust_vocab=cust_vocab,
            )
            
            df_kw = df_kw.add_prefix(f'topk_{col}_')
            df_kw_train = df_kw.iloc[:len(self.df_train_raw)]
            df_kw_test = df_kw.iloc[len(self.df_train_raw):].reset_index(drop=True)
            
            self.nlp_features[f'topk_{col}'] = {'train': df_kw_train, 'test': df_kw_test}
            print(f'[INFO] Topk features saved to self.nlp_features["topk_{col}"]')


    def tfidf(self, colnames_nlp, mode='seperate', stop_words=None):
        print('[INFO] Using jieba cut kernel')
        for col in colnames_nlp:
            tfidf = TfidfVectorizer(
                tokenizer=jieba.lcut, 
                token_pattern=None, 
                norm='l2', 
                use_idf=True, 
                smooth_idf=True, 
                sublinear_tf=True, 
                stop_words=stop_words)
            
            if mode == 'seperate':
                print('[INFO] Seperate mode')
                df_tfidf_train = tfidf.fit_transform(self.df_train_raw[col])
                df_tfidf_train = pd.DataFrame(df_tfidf_train.toarray(), columns=tfidf.get_feature_names_out())
                df_tfidf_test = tfidf.transform(self.df_test_raw[col])
                df_tfidf_test = pd.DataFrame(df_tfidf_test.toarray(), columns=tfidf.get_feature_names_out())
            elif mode =='concat':
                print('[INFO] Concat mode')
                df_tfidf = tfidf.fit_transform(
                    pd.concat(
                        [self.df_train_raw[col], self.df_test_raw[col]],
                        axis=0, 
                        ignore_index=True
                    ))
                df_tfidf = pd.DataFrame(df_tfidf.toarray(), columns=tfidf.get_feature_names_out())
                df_tfidf_train = df_tfidf.iloc[:len(self.df_train_raw)]
                df_tfidf_test = df_tfidf.iloc[len(self.df_train_raw):].reset_index(drop=True)
            else:
                raise ValueError(f'[ERROR] Invalid mode: {mode}')
            
            df_tfidf_train = df_tfidf_train.add_prefix(f'tfidf_{col}_')
            df_tfidf_test = df_tfidf_test.add_prefix(f'tfidf_{col}_')
            
            self.nlp_features[f'tfidf_{col}'] = {'train': df_tfidf_train, 'test': df_tfidf_test}
            print(f'[INFO] Tfidf features saved to self.nlp_features["tfidf_{col}"]')
       
        
    def ngram(self, colnames_nlp, mode='seperate', max_features=4000, stop_words=None):
        print('[INFO] Using default ngram kernel')
        for col in colnames_nlp:
            tfidf = TfidfVectorizer(ngram_range=(1, 4), max_features=max_features, stop_words=stop_words)
            
            if mode == 'seperate':
                print('[INFO] Seperate mode')
                df_ngram_train = tfidf.fit_transform(self.df_train_raw[col])
                df_ngram_train = pd.DataFrame(df_ngram_train.toarray(), columns=tfidf.get_feature_names_out())
                df_ngram_test = tfidf.transform(self.df_test_raw[col])
                df_ngram_test = pd.DataFrame(df_ngram_test.toarray(), columns=tfidf.get_feature_names_out())
            elif mode =='concat':
                print('[INFO] Concat mode')
                df_ngram = tfidf.fit_transform(
                    pd.concat(
                        [self.df_train_raw[col], self.df_test_raw[col]],
                        axis=0, 
                        ignore_index=True
                    ))
                df_ngram = pd.DataFrame(df_ngram.toarray(), columns=tfidf.get_feature_names_out())
                df_ngram_train = df_ngram.iloc[:len(self.df_train_raw)]
                df_ngram_test = df_ngram.iloc[len(self.df_train_raw):].reset_index(drop=True)
            else:
                raise ValueError(f'[ERROR] Invalid mode: {mode}')
            
            df_ngram_train = df_ngram_train.add_prefix(f'ngram_{col}_')
            df_ngram_test = df_ngram_test.add_prefix(f'ngram_{col}_')
            
            self.nlp_features[f'ngram_{col}'] = {'train': df_ngram_train, 'test': df_ngram_test}
            print(f'[INFO] Ngram features saved to self.nlp_features["ngram_{col}"]')
        
        
    def w2v(self, colnames_nlp, model_path, stop_words=None):
        import gensim
        tokenizer = jieba.lcut
        self.model_w2v = gensim.models.Word2Vec.load(model_path)
        
        for col in colnames_nlp:
            print('[INFO] Processing train data')
            sentence_vectors = []
            for i in tqdm(range(len(self.df_train_raw))):
                cutted_sentence = tokenizer(self.df_train_raw[col][i])
                cutted_sentence = [x for x in cutted_sentence if x not in stop_words]
                sentence_vectors.append(self.__get_sentence_vector(cutted_sentence))
            df_w2v_train = pd.DataFrame(sentence_vectors)
            
            print('[INFO] Processing test data')
            sentence_vectors = []
            for i in tqdm(range(len(self.df_test_raw))):
                cutted_sentence = tokenizer(self.df_test_raw[col][i])
                cutted_sentence = [x for x in cutted_sentence if x not in stop_words]
                sentence_vectors.append(self.__get_sentence_vector(cutted_sentence))
            df_w2v_test = pd.DataFrame(sentence_vectors)
            
            df_w2v_train.columns = [f'w2v_{col}_{i}' for i in range(df_w2v_train.shape[1])]
            df_w2v_test.columns = [f'w2v_{col}_{i}' for i in range(df_w2v_test.shape[1])]
            
            self.nlp_features[f'w2v_{col}'] = {'train': df_w2v_train, 'test': df_w2v_test}
            print(f'[INFO] W2v features saved to self.nlp_features["w2v_{col}"]')
            
   
    def finalize(self, which_to_output:Union[str, List]='all', encode_columns:bool=False):
        if which_to_output == 'all':
            df_train_x = pd.concat([self.df_train_x] + [self.nlp_features[k]['train'] for k in self.nlp_features.keys()], axis=1)
            df_test = pd.concat([self.df_test] + [self.nlp_features[k]['test'] for k in self.nlp_features.keys()], axis=1)
        elif which_to_output == 'manual':
            df_train_x = self.df_train_x
            df_test = self.df_test
        elif isinstance(which_to_output, str):
            if which_to_output not in self.nlp_features.keys():
                raise ValueError(f'[ERROR] Invalid which_to_output: {which_to_output}')
            df_train_x = pd.concat([self.df_train_x] + [self.nlp_features[which_to_output]['train']], axis=1)
            df_test = pd.concat([self.df_test] + [self.nlp_features[which_to_output]['test']], axis=1)
        elif isinstance(which_to_output, list):
            if not set(which_to_output).issubset(set(self.nlp_features.keys())):
                raise ValueError(f'[ERROR] Invalid which_to_output: {which_to_output}')
            df_train_x = pd.concat([self.df_train_x] + [self.nlp_features[k]['train'] for k in which_to_output], axis=1)
            df_test = pd.concat([self.df_test] + [self.nlp_features[k]['test'] for k in which_to_output], axis=1)
            
        if encode_columns:
            print('[INFO] Encoding columns')
            colnames_origin = df_train_x.columns
            colnames_encode = {}
            for i in range(len(colnames_origin)):
                colnames_encode[colnames_origin[i]] = str(i)

            df_train_x.rename(columns=colnames_encode, inplace=True)
            df_test.rename(columns=colnames_encode, inplace=True)

        res = (df_train_x, df_test, self.df_train_y,)

        if self.encody_y:
            res += (self.le,)
        if encode_columns:
            res += (colnames_encode,)

        return res


class CustomEval:
    @staticmethod
    def f1_macro_xgb(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
        y_true = dtrain.get_label()
        # y_pred = np.argmax(predt, axis=1)
        y_pred = predt
        score = f1_score(y_true, y_pred, average='macro')
        return 'f1_macro', -score
    
    @staticmethod
    def rmsle_xgb(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
        ''' Root mean squared log error metric.'''
        y = dtrain.get_label()
        predt[predt < -1] = -1 + 1e-6
        elements = np.power(np.log1p(y) - np.log1p(predt), 2)
        return 'PyRMSLE', float(np.sqrt(np.sum(elements) / len(y)))
    
    @staticmethod
    def f1_macro_lgb(y_pred, data):
        y_true = data.get_label()
        y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)  
        return 'f1_macro', f1_score(y_true, y_pred, average='macro'), True

    
def _save_importance_excel(model, idx, out_dir, model_type:str='xgb', importance_type:str='gain', name_dict:dict=None):
    '''
    model_type: choose from 'xgb', 'lgb'
    '''
    df_importance = pd.DataFrame(columns=['feature', 'importance'])
    if model_type == 'xgb':
        for keys, values in model.get_score(importance_type=importance_type).items():
            df_importance = pd.concat([df_importance, pd.DataFrame({'feature': keys, 'importance': values}, index=[0])], axis=0)
    
    elif model_type == 'lgb':
        importance = model.feature_importance(importance_type=importance_type)
        names = model.feature_name() if name_dict is None else [name_dict[i] for i in model.feature_name()]
        df_importance = pd.DataFrame({'feature': names, 'importance': importance})
    
    else:
        raise ValueError(f'[ERROR] Invalid model_type: {model_type}')    

    df_importance.reset_index(drop=True, inplace=True)
    df_importance.sort_values(by='importance', ascending=False, inplace=True)
    if not exists(join(out_dir, 'importance')):
        makedirs(join(out_dir, 'importance'))
    df_importance.to_excel(join(out_dir, f'importance/feature_importance_{importance_type}_{idx}.xlsx'), index=False)
    print(f'[INFO] Feature importance saved to {out_dir}/feature_importance_{importance_type}_{idx}.xlsx')
    

def report_model(path, res, colnames_decode, no_val_mode=False, additional_info=None):
    '''
    Logging. If there's no validation set, set no_val_mode=True. Then 
    this function will only save the feature importances.    
    '''
    # models
    pickle.dump(res, open(join(path, 'result.pkl'), 'wb'))
    print(f'[INFO] Model and reports saved to {path}/result.pkl')
    
    # feature importance
    for i in range(len(res)):
        _save_importance_excel(res[i]['model'], i, path, model_type='lgb', importance_type='gain', name_dict=colnames_decode)
        _save_importance_excel(res[i]['model'], i, path, model_type='lgb', importance_type='split', name_dict=colnames_decode)
    print(f'[INFO] Feature importance saved to {path}/importance')
    
    
    if not no_val_mode:
        # log
        with open(join(path, 'report.txt'), 'w', encoding='utf-8') as f:
            f.write(f'[INFO] Additional info: {additional_info}\n')
            f.write('='*20 + '\n')
            total_f1 = []
            for i in range(len(res)):
                total_f1.append(res[i]['f1_score'])
            print(f'[INFO] 5 fold avg f1: {np.mean(total_f1)}')
            print(f'[INFO] 5 fold var f1: {np.var(total_f1)}')
            f.write(f'[INFO] 5 fold avg f1: {np.mean(total_f1)}\n')
            f.write(f'[INFO] 5 fold var f1: {np.var(total_f1)}\n')
            for i in range(len(res)):
                f.write('='*20 + '\n')
                f.write(f'[INFO] Fold {i} f1: {res[i]["f1_score"]}\n')
                f.write(f'[INFO] Fold {i} confmat: \n{res[i]["confmat"]}\n')
                f.write(f'[INFO] Fold {i} report: \n{res[i]["report"]}\n')
        print(f'[INFO] Report text file saved to {path}/report.txt')
                
        # failed samples
        df_fails = pd.DataFrame()
        for i in range(len(res)):
            df_fails = pd.concat([df_fails, res[i]['fails'].reset_index(drop=False)], axis=0)
        df_fails.to_excel(join(path, 'fails.xlsx'), index=False)
        print(f'[INFO] Failed cases saved to {path}/fails.xlsx')
    else:
        with open(join(path, 'report.txt'), 'w', encoding='utf-8') as f:
            f.write(f'[INFO] Additional info: {additional_info}\n')
            f.write('='*20 + '\n')