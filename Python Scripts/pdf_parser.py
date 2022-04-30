#!usr/bin/env/python
"""
Authors: Fatih Uenal & Shashi Badloe
Description: Script for retrieving raw data from pdf files.
First step in the pipeline.
Source of the model: https://link.springer.com/article/10.1007%2Fs10113-020-01677-8
"""
import os
import tika
tika.initVM()
from tika import parser
import re
import shutil

def open_pdf(directory):
    """Opens PDF file and converts it into text

    directory: string, folder containing subfolders with climate documents
    
    In this case 'PDF_files' is the directory
    """
    for sub_folder in os.listdir(directory):
        sub_directory = os.path.join(directory,sub_folder)
        for pdf_file in os.listdir(sub_directory):
            full_path = os.path.join(sub_directory,pdf_file)
            try:
                pdf_content = pdf_to_txt(full_path)
                if isinstance(pdf_content, str) and len(pdf_content) > 1000:
                    yield full_path, pdf_content
                else:
                    print('No text found, skipping "{}"..'.format(pdf_file))
                    continue
            except Exception as e:
                print(e)
                print('Failed to parse "%s"' % pdf_file)
            
                
def pdf_to_txt(full_path):
    """Turns all text from pdf to raw string

    full_path: string, full path to pdf file to convert
    """
    file = open(full_path,'rb')
    extracted_text = parser.from_buffer(file)
    return extracted_text['content']

def write_pdf_content(pdf_content, target_dir):
    """Writes convert pdf content to plain text file

    pdf_content: string, raw text from pdf file
    target_dir: string, directory to save the text file in
    """
    txt_file = open(target_dir,'w', encoding = 'utf-8')
    txt_file.write(pdf_content)
    txt_file.close()
    
def corrected_text(pdf_content):
    """Returns string contaning corrected pdf text

    pdf_content: string, raw text from pdf file
    
    This step is necessary because some strings get chopped up
    into tiny pieces. These need to be recognized and glued back
    together
    """
    def fix_text(damaged_txt):
        result = damaged_txt.split('\n')
        result = ''.join(result)
        result = re.sub(r'([a-z])([A-Z])',r'\1 \2',result)
        fixed_txt = re.sub(r'([.,])([A-Z])',r'\1 \2',result)
        return fixed_txt

    replaced = re.sub(r"(^.{1,4}\n)+",\
                lambda text: '\n\n'+fix_text(text.group())+'\n\n',\
                      pdf_content,flags=re.MULTILINE|re.DOTALL)
    return replaced

def lines_to_blocks(text):
    """Returns original text without whitespace between every single line

    txt: string, raw, unedited textfile
    txt_file: string, name of the textfile

    Checks if every line is seperated by whitespace based on experimental value
    and if true combines them
    """
    n_sep = text.count('\n\n')
    n_lines = text.count('\n')
    #approximate ratio of double newlines vs single newline: 40
    if int(n_sep/n_lines*100) > 40:
        text = re.sub('\n\n', '\n',text)
    #try to split it up with topic indicators such as numbers or bullet points
        text = re.sub(r'[0-9]+[.]', '\n',text)
        text = re.sub('•', '\n',text)
    return text 

def create_folder(folder_name):
    """Creates a folder to hold parsed files if it doesn't exists

    folder_name: string, full path of the folder you want to make
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
def reset_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        
if __name__== '__main__':
    #destination
    folder_name = os.path.join("..","parsed_files")
    reset_folder(folder_name)
    create_folder(folder_name)
    #files to parse
    directory = os.path.join("..","PDF_files")
    for full_path, pdf_content in open_pdf(directory):
        target_dir = full_path.replace(directory,folder_name)\
                     .replace('.pdf','.txt')
        create_folder(os.path.dirname(target_dir))

        #fixes for broken text
        pdf_content = corrected_text(pdf_content)
        pdf_content = lines_to_blocks(pdf_content)
        #write to file
        write_pdf_content(pdf_content,target_dir)
    print("Finished parsing PDF files")

    
