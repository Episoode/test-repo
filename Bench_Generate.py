import json
import os
import time
import random
from Utils import get_embedding, load_converter, find_most_similar_index, load_embed_model, load_openai_client, \
    json_parser
from Single_Modality_Query_Generate import generate_doc_question
from OCR_and_Detection import  process_doc


def generate_base_bench(tgt_dir,source_dir,data_path):
    data = []
    client = load_openai_client()
    time1 = time.time()
    converter = load_converter()
    os.makedirs(tgt_dir, exist_ok=True)
    docs = os.listdir(source_dir)
    for doc in docs:
        process_doc(os.path.join(source_dir,doc),os.path.join(tgt_dir,doc),converter)
        generate_doc_question(os.path.join(tgt_dir,doc),load_openai_client(),data)
    time2 = time.time()
    print(f"Time taken: {time2 - time1} seconds")
    with open(data_path, 'w') as f:
        f.write(json.dumps(data, indent=4))
    print('Data saved to', data_path)

def generate_hard_question(doc_data,hard_data,client,model,tokenizer,device):
    print(f"Generating hard questions for {doc}")
    text_ids = [i for i,item in enumerate(doc_data) if item['modality'] == 'text']
    table_ids = [i for i,item in enumerate(doc_data) if item['modality'] == 'table']
    figure_ids = [i for i,item in enumerate(doc_data) if item['modality'] == 'figure']
    print(f'Text:{text_ids} Table:{table_ids} Figure:{figure_ids}')
    all_questions = [item['question'] for item in doc_data]
    all_embeds = get_embedding(all_questions,model,tokenizer, device, return_tensor=True)

    modalities = {"text": text_ids,"table": table_ids,"figure": figure_ids,}
    available_modalities = [modality for modality, ids in modalities.items() if len(ids) > 2]
    if not available_modalities:
        return None

    q_num = random.randint(2, 4)

    for i in range(q_num):
        print('------------------------------------------------')
        print(f"Generating hard question {i+1}/{q_num}")
        num_iterations = random.randint(1, 3) #融合迭代次数
        selected_modality = random.choice(available_modalities)

        selected_idx = random.choice(modalities[selected_modality])
        current_embedding = all_embeds[selected_idx]
        current_question = all_questions[selected_idx]

        json_dict = None
        question_ids = list(range(len(doc_data)))
        question_ids.remove(selected_idx)
        for j in range(num_iterations):
            questions_embed = [all_embeds[idx] for idx in question_ids]
            idx_in_select_ids = find_most_similar_index(current_embedding,questions_embed)
            sim_idx = question_ids[idx_in_select_ids]
            question_ids.remove(sim_idx)

            print(f'Iteration {j+1}/{num_iterations}')
            print(f'Question1 {current_question}')
            print(f'Answer1 {doc_data[selected_idx]["answer"]}')
            print(f'Question2 {doc_data[sim_idx]["question"]}')
            print(f'Answer2 {doc_data[sim_idx]["answer"]}')
            cur_item = json_dict if json_dict else doc_data[selected_idx]
            json_dict = fuse_question(cur_item,doc_data[sim_idx],json_dict,client)

            if json_dict:
                question_ids.append(sim_idx)
                current_question = json_dict['question']
                current_embedding = get_embedding([current_question],model,tokenizer,device)[0]
                print(f'Fused question: {current_question}')
                print(f'Fused answer: {json_dict["answer"]}')
                print()

            hard_data.append(json_dict)
        print('------------------------------------------------\n')
    return hard_data


def fuse_question(item1,item2,json_dict,client):
    question1, answer1 = item1['question'], item1['answer']
    question2, answer2 = item2['question'], item2['answer']
    prompt = '''
    Carefully analyze both questions and answers. You are tasked with merging two Q&A pairs into a single, cohesive Q&A pair. 
    The new question should combine the key information from both original questions, 
    and the new answer should synthesize the content of both original answers while maintaining clarity and conciseness.
    Ensure the resulting Q&A pair is coherent and concise.

    Here are the two original Q&A pairs:
    1. Question 1: %s\n
       Answer 1: %s\n
    2. Question 2: %s\n
       Answer 2: %s\n

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
    '''%(question1,answer1,question2,answer2)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a master at analyzing questions and statements."},
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    qa_dict = json_parser(response.choices[0].message.content)
    if not qa_dict:
        print('json_parser failed')
        return None

    if not json_dict:
        json_dict = qa_dict
        json_dict['modality'] = list(set([item1['modality'], item2['modality']]))
        json_dict['evidence_page'] = list(set([item1['evidence_page'],item2['evidence_page']]))
        json_dict['doc'] = item1['doc']
    else:
        json_dict['question'] = qa_dict['question']
        json_dict['answer'] = qa_dict['answer']
        json_dict['modality'].append(item2['modality'])
        json_dict['evidence_page'].append(item2['evidence_page'])

    return json_dict

with open('Data/data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

hard_data = []
client = load_openai_client()
model, tokenizer ,device = load_embed_model('BAAI/bge-m3')

dir = 'OCR_DOC'
for doc in os.listdir(dir):
    doc_data = [item for item in data if item['doc'] == doc]
    generate_hard_question(doc_data, hard_data, client, model, tokenizer,device)

with open('Data/hard_data.json', 'w') as f:
    f.write(json.dumps(hard_data, indent=4))