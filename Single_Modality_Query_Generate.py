import os
import json
import random
from PIL import Image
from Utils import load_openai_client, pil_images_to_base64, json_parser


def generate_single_modality_qa_json(doc_dir,text_dir,modality_dir,modality,client):
    messages = [
        {"role": "system", "content": "You are a helpful document assistant"},
    ]

    modality_prompt = f'''
    You are required to read the context and {modality} provided to you. 
    The contextual information serves only as a supplementary and background reference for the {modality}. 
    You must formulate a single-hop global query and corresponding answer 
    **based on the content of the {modality}, not on the contextual information**'''

    pure_text_prompt = f'''
    Read the text provided to you.You must formulate a single-hop global query and corresponding answer'''

    bade_prompt = '''
    You are encouraged to generate diverse question, such as "which," "how," "why.",etc,try to advoid 'what was' or 'what is'.
    (A global query is a specific and clear question that can be answered, whereas a non-global query lacks sufficient context and cannot be answered meaningfully.
    Examples of non-global queries include:
    "Which team performed the best?" (insufficient context),
    "According to the chart, what are the four business cooperation activities?" (no chart provided).
    Examples of global queries include:
    "In the men's team table tennis event at the 2024 Olympics, which team performed the best?"
    "What is the percentage of registered voters who support or lean toward the candidate from the party with the higher total percentage of good policy ideas and high ethical standards and closely follow congressional elections in their district in the survey of U.S. adults conducted April 25 - May 1, 2018?")

    Your response must be in JSON dictionary format **pure text** without any extra labels (for instance, *** do not add “```json” *** at the beginning or end). The following information is required in the JSON:

    - question (string)
    - answer (string, no more than 50 words)

    Example responses:
    {
        "question": "Which companies demonstrate competitive advantages in the electric vehicle market?",
        "answer": "Companies such as BYD, Tesla, and Li Auto are leading in electric vehicle technology and sales.",
    }

    {
        "question": "What were the medal counts for China and the US in the 2024 Olympics?",
        "answer": "China secured 91 medals, whereas the US obtained 126 medals.",
    }
    '''

    if modality == 'text':
        txt_files = [f for f in os.listdir(text_dir)]
        prompt = pure_text_prompt + bade_prompt
        random_file = random.choice(txt_files)
        evidence = os.path.basename(random_file).split('.')[0]
        txt = os.path.join(text_dir, random_file)
        with open(txt, 'r', encoding='utf-8') as file:
            content = file.read()

        if len(content) < 100:
            return None

    else :
        img_files = [f for f in os.listdir(modality_dir)]
        img_name = random.choice(img_files)
        evidence = txt_idx = int(img_name.split('_')[0])
        img = os.path.join(modality_dir, img_name)
        img = Image.open(img)
        img = pil_images_to_base64(img)
        prompt = modality_prompt + bade_prompt + f'''
        \nYour question and answer must **based on the content of the {modality}, not on the contextual information**
    '''
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url":{"url": f"data:image/jpeg;base64,{img}", "detail": "high"}
                    },
                ],
            }
        )

        content = ''
        for idx in range(txt_idx-1,txt_idx+2):
            if idx<0:
                break
            with open(os.path.join(text_dir,str(txt_idx)+'.txt'), 'r', encoding='utf-8') as file:
                content += file.read()+ '\n'

    prompt += '\nContext:\n' + content
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    json_dict = json_parser(response.choices[0].message.content)
    if json_dict:
        json_dict['modality'] = modality
        json_dict['doc'] =  os.path.basename(doc_dir)
        json_dict['evidence_page'] = str(evidence)

    return  json_dict

def generate_doc_question(doc_dir,client,data):
    modality_list = ['text']
    text_dir = os.path.join(doc_dir, 'text')

    if len(os.listdir(os.path.join(doc_dir,'table'))) > 0:
        modality_list.append('table')

    if len(os.listdir(os.path.join(doc_dir,'figure'))) > 0:
        modality_list.append('figure')

    text_num =  5 if len(os.listdir(text_dir)) > 20 else 3
    for i in range(text_num):
        json_dict = generate_single_modality_qa_json(doc_dir,text_dir,None,'text',client)
        if json_dict:
            data.append(json_dict)

    for modality in modality_list[1:]:
        if modality in modality_list:
            length = len(os.listdir(os.path.join(doc_dir, modality)))
            num = 4 if length > 5 else (3 if length > 3 else 1)
            for i in range(num):
                json_dict = generate_single_modality_qa_json(doc_dir,text_dir,os.path.join(doc_dir,modality),modality,client)
                if json_dict != None and json_dict != 0:
                    data.append(json_dict)

#
# data = []
# client = load_openai_client()
# foler_path = 'OCR_DOC'
# docs = os.listdir(foler_path)
# for doc in docs:
#     doc_dir = os.path.join(foler_path, doc)
#     generate_doc_question(doc_dir,client,data)
#
# with open('Data/data.json', 'w') as f:
#     json.dump(data, f, indent=4)












