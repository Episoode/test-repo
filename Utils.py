'''

                           _ooOoo_
                          o8888888o
                          88" . "88
                          (| -_- |)
                          O\  =  /O
                       ____/`---'\____
                     .'  \\|     |//  `.
                    /  \\|||  :  |||//  \
                   /  _||||| -:- |||||-  \
                   |   | \\\  -  /// |   |
                   | \_|  ''\---/''  |   |
                   \  .-\__  `-`  ___/-. /
                 ___`. .'  /--.--\  `. . __
              ."" '<  `.___\_<|>_/___.'  >'"".
             | | :  `- \`.;`\ _ /`;.`/ - ` : | |
             \  \ `-.   \_ __\ /__ _/   .-` /  /
        ======`-.____`-.___\_____/___.-`____.-'======
                           `=---='
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            Buddha bless us. Bugs never appear

'''
import base64
import io
import json
import warnings
import torch
import torch.nn.functional as F
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
warnings.filterwarnings("ignore")

def load_embed_model(model_name = "BAAI/bge-m3"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return  model, tokenizer ,device

def load_ppl_model(model_path = "BAAI/bge-m3"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
    return model,tokenizer,device

def get_embedding(string_list, model, tokenizer, device, return_tensor=True):

    inputs = tokenizer(string_list, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
    if return_tensor:
        return embeddings  # 返回 PyTorch 张量
    else:
        return [embedding.cpu().numpy() for embedding in embeddings]

def find_most_similar_index(target_tensor,tensor_list):
    if isinstance(tensor_list,list):
        tensor_list = torch.stack(tensor_list)
    similarities = F.cosine_similarity(target_tensor,tensor_list,dim=1)
    sim_idx = torch.argmax(similarities).item()
    return sim_idx

def load_openai_client():
    client = OpenAI()
    return client

def pil_images_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def json_parser(s):
    s = s.replace('```json','').replace('```','')
    try:
        json_dict =  json.loads(s)
        return json_dict
    except json.JSONDecodeError:
        print('JSONDecodeError')
        return None

def load_converter(scale = 2.0) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = scale
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    return converter

