#!usr/bin/env/python
"""
Authors: Fatih Uenal & Shashi Badloe
Description: Segments the texts by making lists of paragraphs and
removing empty lines. Also filters for meaningful words. 
Second step in the pipeline.
Source of the model: https://link.springer.com/article/10.1007%2Fs10113-020-01677-8
"""
import os
import re
import time
import nltk
from nltk.tag.perceptron import PerceptronTagger
from pdf_parser import create_folder
from pdf_parser import reset_folder

def open_texts(directory):
    """Returns one string containing textfile from .txt files

    directory: string, directory of the textfiles
    """
    for sub_folder in os.listdir(directory):
        sub_directory = os.path.join(directory,sub_folder)
        for txt_file in os.listdir(sub_directory):
            target_dir = os.path.join(sub_directory,txt_file)
            txt = open(target_dir,'r',encoding = 'utf-8').read()
            yield txt,txt_file
               
def split_blocks(text):
    """Returns text splitted on blocks (double newlines)

    text: string, raw, unedited textfile
    """
    splitted_blocks = text.split('\n\n')
    return splitted_blocks

def filter_fn(condition,link_list):
    """Comprehensive function to filter a list where elements are linked"""
    filtered_lists = list(map(list,zip(*[[block,link_list[1][i]] for i,block \
                               in enumerate(link_list[0]) if condition(block)])))
    return filtered_lists

def map_fn(condition, link_list):
    """Comprehensive function to apply function to first element in link list"""
    mapped_lists = [[condition(block) for i,block in \
		     enumerate(link_list[0])],link_list[1]]
    return mapped_lists

def filter_blocks(splitted_text,min_block_length):
    """Returns filtered text splitted on blocks

    block_length: integer, minimum length allowed for a block to be included
    splitted_text: list of two lists, both containing the text splitted on blocks
    """
    #remove empty blocks
    filtered_text = filter_fn(lambda block: block != '', splitted_text)
    #remove blocks only consisting of a single line (headers/footers)
    filtered_text = filter_fn(lambda string: string.count('\n') > 1 ,\
                              filtered_text)
    #remove blocks that contain irrelevant information
    filtered_text = filter_fn(lambda string: string.__contains__('www') == False,filtered_text)
    filtered_text = filter_fn(lambda string: string.__contains__('....') == False,filtered_text)
    filtered_text = filter_fn(lambda string: string.__contains__('http') == False,filtered_text)
    filtered_text = filter_fn(lambda string: string.__contains__('@') == False,filtered_text)
    filtered_text = filter_fn(lambda string: string.__contains__('Fax') == False,filtered_text)
    #remove newline characters
    filtered_text = map_fn(lambda string: string.replace('\n',' '), filtered_text)
    #split '-' conjoined words up to reduce dimensionality
    filtered_text = map_fn(lambda string: string.replace('-',' '), filtered_text)
    #pre-filter for small blocks (saves processing time during later steps)
    filtered_text = filter_fn(lambda string: len(string.split()) \
                           >= min_block_length, filtered_text)
    return filtered_text

def split_words(filtered_blocks):
    """Returns texts splitted on blocks and words within the blocks

    filtered_blocks: list of strings, strings are entire blocks 
    """
    splitted_text = map_fn(lambda string : string.split(),filtered_blocks)
    return splitted_text

def filter_words(splitted_words,forbidden_word_types,max_word_length,min_word_length,min_block_length,max_block_length):
    """Returns text where words are filtered for several things

    splitted_words: list of lists, text splitted on blocks and words
    word_length: integer, minimum allowed word length
    """
    #remove 's at the end of words
    filtered_words =  map_fn(lambda block: list(map(lambda string: \
        string.rstrip("'s") if string.endswith("'s") else string,block))\
                             ,splitted_words)
    #remove punctuation marks around words
    filtered_words =  map_fn(lambda block: list(map(lambda string: \
        re.sub(r'[^A-Za-z0-9°]','',string),block))\
                             ,filtered_words)
    #remove numbers around words
    filtered_words =  map_fn(lambda block: list(map(lambda string: \
        re.sub(r'[0-9]','',string) if re.match(r'[A-Za-z]',string) else string,\
                                                    block)) ,filtered_words)
    #remove every word containing number
    #except when its used as year or temperature indicator
    filtered_words =  map_fn(lambda block: list(filter(lambda string: \
        re.match(r'(?:^(?:19|20)\d{2}$)|(?:[a-zA-Z])|(?:^\d{1,2}°(?:C|F)$)',\
                 string),block)) ,filtered_words)
    #remove empty elements in the bag of words (necessary for pos_tag)
    filtered_words =  map_fn(lambda block: list(filter(lambda string: \
        string != '',block)) ,filtered_words)
    #tokenize text and filter adpositions, pronouns,particles,
    #conjunctions, determiners
    filtered_words1 = map(lambda block: list(filter(lambda string: \
        tagger.tag([string])[0][1] not in forbidden_word_types,\
                                                   block)) ,filtered_words[0])
    filtered_words = [filtered_words1,filtered_words[1]]
    #remove words of length beneath min_word_length and above max_word_length
    filtered_words = map_fn(lambda block: list(filter(\
        lambda word: max_word_length >= len(word) >= min_word_length,block))\
                         ,filtered_words)
    #turn all uppercase to lowercase letters
    filtered_words = map_fn(lambda block: list(map(lambda string : \
        string.lower(), block)), filtered_words)
    #Finally remove all blocks that are now less than block length
    #(resulted from filtering)
    filtered_words = list(filter_fn(lambda block: max_block_length >= len(block) >= min_block_length,\
                                 filtered_words))
    return filtered_words

def save_structure(structured_data,folder_name,directory,txt_file):
    """Writes structured_data to a text file

    structured_data: list of lists, lists contain words from a single block
    folder_name: string, name of the main target folder
    directory: string, full path with all subfolders
    txt_file: string, filename of txt file to write
    """
    create_folder(folder_name)
    for folder in os.listdir(directory):
        target_dir = os.path.join(folder_name,folder)
        create_folder(target_dir)
        for file in os.listdir(os.path.join(directory,folder)):
            if file == txt_file:
                with open(os.path.join(target_dir,file),'w',encoding='utf-8') as struc_txt:
                    struc_txt.write(str(list(structured_data)))

    
if __name__== '__main__':
    #define source and target folders
    directory = '../parsed_files'
    folder_name = "../structured_files"
    #word types we do not want in our bag of words
    #'IN' = preposition, 'CC' = conjunction, 'DT' = determiner, 'PRP($)' = preposition
    #for more info check https://en.oxforddictionaries.com/grammar/word-classes-or-parts-of-speech
    forbidden_word_types = ['IN','CC','DT','PRP','PRP$','TO']
    #Load the tagger
    nltk.download('averaged_perceptron_tagger')
    tagger = PerceptronTagger()
    #Remove existing folder
    reset_folder(folder_name)
    start_time = time.perf_counter()
    for text,txt_file in open_texts(directory):
        splitted_text = split_blocks(text)
        try:
            filtered_blocks = filter_blocks([splitted_text,splitted_text],10)
            splitted_words = split_words(filtered_blocks)
        except IndexError:
            print('No eligable blocks in "{}". Skipping document.'\
                  .format(txt_file))
            continue
        structured_data = filter_words(splitted_words,forbidden_word_types,20,3,10,200)
        save_structure(structured_data,folder_name,directory,txt_file)                                  
    print('Data took {} seconds to structurize'.format(int(time.perf_counter() - start_time)))
