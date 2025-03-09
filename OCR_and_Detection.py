import os
import shutil
import time
from pathlib import Path
from PIL import Image
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
import warnings
warnings.filterwarnings("ignore")

def process_single_page(input_doc_dir,page,text_dir,table_dir,figure_dir, converter):
    conv_res = converter.convert(os.path.join(input_doc_dir,page))
    full_doc_md = conv_res.document.export_to_markdown()

    table_counter = 0
    picture_counter = 0
    page_name = page.split('.jpg')[0]

    with open(os.path.join(text_dir, f"{page_name}.txt"), "w",encoding='utf-8') as fp:
        fp.write(full_doc_md)

    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            element_image_filename = Path(os.path.join(table_dir, f"{page_name}_{table_counter-1}.png"))
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

        if isinstance(element, PictureItem):
            picture_counter += 1
            element_image_filename = Path(os.path.join(figure_dir, f"{page_name}_{picture_counter-1}.png"))
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

def process_doc(input_doc_dir:str,output_doc_dir:str,converter):
    os.makedirs(output_doc_dir,exist_ok=True)
    text_dir = os.path.join(output_doc_dir,'text')
    table_dir = os.path.join(output_doc_dir,'table')
    figure_dir = os.path.join(output_doc_dir,'figure')

    os.makedirs(text_dir,exist_ok=True)
    os.makedirs(table_dir,exist_ok=True)
    os.makedirs(figure_dir,exist_ok=True)

    pages  = os.listdir(input_doc_dir)
    for page in pages:
        process_single_page(input_doc_dir,page,text_dir,table_dir,figure_dir, converter)

def check_and_adjust(dir,min_num = 20):
    for doc in os.listdir(dir):
        doc_dir = os.path.join(dir,doc)
        text_dir = os.path.join(doc_dir,'text')
        table_dir = os.path.join(doc_dir,'table')
        figure_dir = os.path.join(doc_dir,'figure')

        # Remove empty text
        if len(os.listdir(text_dir)) == 0:
            print(f"Empty text dir: {doc_dir}")
            shutil.rmtree(doc_dir)

        else:
            for file in os.listdir(text_dir):
                file_path = os.path.join(text_dir,file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().replace('<!-- image -->','').replace('<!-- formula-not-decoded -->','')
                if len(content.split()) < min_num:
                    os.remove(file_path)
                else :
                    with open(file_path , 'w', encoding='utf-8') as f:
                        f.write(content)

        # Remove useless table and figure
        if len(os.listdir(table_dir)):
            for img in os.listdir(table_dir):
                img_path = os.path.join(table_dir,img)
                with Image.open(img_path) as image:
                    width, height = image.size
                if width <=200 or height <= 200:
                        os.remove(img_path)

        if len(os.listdir(figure_dir)):
            for img in os.listdir(figure_dir):
                img_path = os.path.join(figure_dir,img)
                with Image.open(img_path) as image:
                    width, height = image.size
                if width <=200 or height <= 200:
                        os.remove(img_path)



# check_and_adjust('OCR_DOC')
# os.makedirs('Final_OCR',exist_ok=True)
# converter = load_converter()
#
# dir1 = 'Final_Doc/WebDocs250'
# dir2 = 'Final_Doc/PDF200'
# os.makedirs(dir1,exist_ok=True)
# os.makedirs(dir2,exist_ok=True)
# tgt_dir1 = 'Final_OCR/WebDocs250'
# tgt_dir2 = 'Final_OCR/PDF200'
#
# webdocs = os.listdir(dir1)
# for doc in tqdm(webdocs):
#     doc_dir = os.path.join(dir1, doc)
#     source_path = os.path.join(tgt_dir1, doc)
#     tgt_path = os.path.join(tgt_dir1, doc)
#     process_doc(doc_dir, tgt_path, converter)
#
# pdf_docs = os.listdir(dir2)
# for doc in tqdm(pdf_docs):
#     doc_dir = os.path.join(dir2, doc)
#     source_path = os.path.join(tgt_dir2, doc)
#     tgt_path = os.path.join(tgt_dir2, doc)
#     process_doc(doc_dir, tgt_path, converter)