from mednlp.service.base_request_handler import BaseRequestHandler
import torch
from transformers import BertConfig, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
mask_padding_with_zero=True
pad_token=0
pad_token_segment_id=0
max_len = 128
import pdb

def convert_examples_to_feature(query):
    inputs = tokenizer.encode_plus(query, add_special_tokens=True, max_length=max_len)
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    padding_length = max_len - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
    return torch.tensor([input_ids], dtype=torch.long),torch.tensor(
        [attention_mask], dtype=torch.long), torch.tensor(
        [token_type_ids], dtype=torch.long)

label_num_dict = {line.split('=')[0]: index for index, line in
                          enumerate(open(file='dept.txt', mode='r', encoding='utf-8').readlines())}
print(label_num_dict)
num_label_dict = {v: k for k, v in label_num_dict.items()}
print(num_label_dict)

class Control(object):

    def __init__(self, name):
        w_path = 'history/1/'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('load start...')
        if self.device == 'cuda':
            self.model = torch.load(w_path + name)
        else:
            self.model = torch.load(w_path + name, map_location='cpu')

        print('load finish...')

    def predict(self, q):
        result = {}
        # pdb.set_trace()
        batch = convert_examples_to_feature(q)
        batch = tuple(t.to(self.device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'token_type_ids': batch[2],
                  'attention_mask': batch[1]}
        outputs = self.model(**inputs)
        logits = outputs[0]
        preds = torch.argmax(logits.detach().cpu(), dim=1).item()
        result['pred'] = preds
        result['label'] = num_label_dict[preds]
        return result
print('control...')
control = Control(name='dept_6.pkl')

class InquirySuggest(BaseRequestHandler):

    def initialize(self, runtime=None, **kwargs):
        super(InquirySuggest, self).initialize(runtime, **kwargs)

    def post(self):
        self.get()

    def get(self):
        self.asynchronous_get()

    def _get(self):
        q = self.get_argument('q')
        print(q)
        result = {}
        result = control.predict(q)
        return result

# if __name__ == '__main__':
    # result  = control.predict('头痛挂什么科')
    # print(result)
if __name__ == '__main__':
    handlers = [(r'/bert_dept', InquirySuggest, dict(runtime={}))]
    import ailib.service.base_service as base_service

    base_service.run(handlers)
