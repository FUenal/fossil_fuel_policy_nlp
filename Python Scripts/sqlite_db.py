#!usr/bin/env/python
"""
Authors: Fatih Uenal & Shashi Badloe
Description: Creates SQLite database and populates it with structured textual data
Third step in the pipeline.
Source of the model: https://link.springer.com/article/10.1007%2Fs10113-020-01677-8
"""
import sqlite3
import os
import time
from text_cleanup import open_texts


def init_db(db_name):
    """Creates a db with db_name

    db_name: string, name of the database
    """
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    return c,conn

def create_table(table_entity,cursor):
    """Creates an empty table within the database

    table_entity: string, name of the table
    cursor: db cursor, used to send queries to the database
    
    table parameters:
    doc_nr: int, unique identifier
    year: int, year of publication
    block: list, bag of words content of the block
    original: string, original unprocessed content of the block
    source: string, filename where the block originated from 
    class: string, type of document, either adaptation, mitigation or unrelated
    """
    if table_entity == 'Labeled_data':
        cursor.execute('''CREATE TABLE IF NOT EXISTS {}
              (id INT PRIMARY KEY, block TEXT, original TEXT,
              source TEXT, class TEXT)'''.format(table_entity))
        
    elif table_entity == 'Unlabeled_data':
        cursor.execute('''CREATE TABLE IF NOT EXISTS {}
              (id INT PRIMARY KEY, dept TEXT, date TEXT, block TEXT, original TEXT,
              source TEXT, pred_class TEXT)'''.format(table_entity))
    

def get_labeled_data(directory):
    """Returns all necessary data from the training set

    directory: string, directory where the parsed files are stored
    """
    for sub_folder in os.listdir(directory):
        if sub_folder.split()[0] == "Mitigation" or \
           sub_folder.split()[0] == "Adaptation" or \
           sub_folder.split()[0] == "Non-climate":            
            sub_directory = os.path.join(directory,sub_folder)
            for txt_file in os.listdir(sub_directory):
                target_dir = os.path.join(sub_directory,txt_file)
                txt = eval(open(target_dir,'r',encoding = 'utf-8').read())
                blocks = txt[0]
                original = txt[1]
                yield blocks,original,txt_file,sub_folder.split()[0]
    
def get_unlabeled_data(directory):
    """Returns all necessary data from the testing set

    directory: string, directory where the parsed files are stored
    """
    for sub_folder in os.listdir(directory):
        if sub_folder.split()[0] == "Mixed":            
            sub_directory = os.path.join(directory,sub_folder)
            for txt_file in os.listdir(sub_directory):
                target_dir = os.path.join(sub_directory,txt_file)
                txt = eval(open(target_dir,'r',encoding = 'utf-8').read())
                try:
                    blocks = txt[0]
                    original = txt[1]
                except IndexError:
                    print('"{}" has no blocks'.format(txt_file))
                    continue
                yield blocks,original,txt_file,sub_folder.split()[0]
                
def read_metadata(filename,directory):
    """Reads metadata for unlabeled data obtained from webscraper

    filename: string, name of the metadata textfile
    directory: string, directory where the parsed files are stored
    """
    if os.path.isfile(filename):
        metadata = eval(open(filename,'r').read())
    else:
        metadata = {}
        for sub_folder in os.listdir(directory):
            if sub_folder.split()[0] == "Mixed":            
                sub_directory = os.path.join(directory,sub_folder)
                for txt_file in os.listdir(sub_directory):
                    metadata[txt_file[:-4]] = (None,None)
        print('metadata.txt not found, please run the web scraper or provide one')
    return metadata

def insert_data(table_entity,cursor,blocks,original,filename,data_class,i):
    """Adds data to the database tables and returns the entry number

    table_entity: string, name of the table
    cursor: db cursor, used to send queries to the database
    blocks: list, bag of words of the blocks
    original: string, original text of the blocks
    filename: string, document title
    data_class: string, type of block Adaptation/Mitigation/Non-climate/Mixed
    i: integer, id number of the block
    """
    
    if filename.endswith('.txt'):
        filename = filename[:-4]+'.pdf'
        
    if data_class != "Mixed":
        block_list = []
        block_index = []
        for j,block in enumerate(blocks):
            if block not in block_list:
                block_list.append(block)
                block_index.append((block,j))
                
        for block,j in block_index:
            parameters = (i,str(block)[1:-1],str(original[j]),filename,\
                                                                  data_class)
            try:
                cursor.execute("INSERT INTO {} VALUES (?,?,?,?,?)"\
                                       .format(table_entity),parameters)
                i += 1
            except sqlite3.IntegrityError:
                i += 1
                continue
            
                
    elif data_class == "Mixed":
        year = metadata[filename][0]
        dept = metadata[filename][1]
        block_list = []
        block_index = []
        for j,block in enumerate(blocks):
            if block not in block_list:
                block_list.append(block)
                block_index.append((block,j))
                
        for block,j in block_index:        
            parameters = (i,dept,year,str(block)[1:-1],str(original[j]),\
                                                        filename,None)
            try:
                cursor.execute("INSERT INTO {} VALUES (?,?,?,?,?,?,?)"\
                               .format(table_entity),parameters)
                i += 1
            except sqlite3.IntegrityError:
                i += 1
                continue 
    return i

    
if __name__ == '__main__':
    directory = '../structured_files'
    
    metadata = read_metadata('metadata.txt',directory)
    db_name = 'climate.db'
    start_time = time.perf_counter()
    c,conn = init_db(db_name)
    create_table('Labeled_data',c)
    create_table('Unlabeled_data',c)          
    
    i = 1                
    for blocks,original,txt_file,data_type in get_labeled_data(directory):
        i = insert_data('Labeled_data',c,blocks,original,txt_file,data_type,i)
    i = 1    
    for blocks,original,txt_file,data_type in get_unlabeled_data(directory):
        i = insert_data('Unlabeled_data',c,blocks,original,txt_file,data_type,i)
        if i % 1000 == 0:
            print(i)
    conn.commit()
    print('Database took {} seconds to create'.format(int(time.perf_counter() - start_time)))
