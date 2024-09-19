import os
import transformers
import json
__version__ = '0.1'

model_name ="Xenova/Meta-Llama-3.1-Tokenizer"

llama_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

class TokenChanger:
    def __init__(self, huggingface_tokenizer):
        self.huggingface_tokenizer = huggingface_tokenizer
        huggingface_tokenizer.save_pretrained("/tmp/__tmp_tokenizer/")
        
        self.tokenizer_json = json.load(open("/tmp/__tmp_tokenizer/tokenizer.json", "r"))
        self.tokenizer_config_json = json.load(open("/tmp/__tmp_tokenizer/tokenizer_config.json", "r"))
        self.special_tokens_map_json = json.load(open("/tmp/__tmp_tokenizer/special_tokens_map.json", "r"))

        self.__update_from_json_files()


    def __update_from_json_files(self):
        self.vocab = self.tokenizer_json["model"]["vocab"]
        self.token_dicts_only_for_merges = {} # tokens that do not need a merge rule.
        self.merges = self.tokenizer_json["model"]["merges"]
        self.added_tokens = self.tokenizer_json["added_tokens"]
        self.added_tokens_decoder = self.tokenizer_config_json["added_tokens_decoder"]
    
    def save_tokenizer(self, save_path):
        # create updated vocab.
        new_vocab = {}
        cnt = 0
        print(self.token_dicts_only_for_merges)
        for k in self.vocab:
            new_vocab[k] = cnt
            cnt += 1
            
            if k in self.token_dicts_only_for_merges:
                for kk in self.token_dicts_only_for_merges[k]:
                    new_vocab[kk] = cnt
                    #print(kk, cnt)
                    cnt += 1
                    
        
        # updated added_tokens and create new_added_token_decoder
        new_added_token_decoder = {}
        for at in self.added_tokens:
            old_id = at["id"]
            at["id"] = cnt
            new_added_token_decoder[str(at["id"])] = self.added_tokens_decoder[str(old_id)]
            cnt += 1

        # assert len(self.added_tokens.keys()) == len(new_added_token_decoder.keys()) 
        self.tokenizer_json["model"]["vocab"] = new_vocab
        self.tokenizer_json["model"]["merges"] = self.merges
        self.tokenizer_json["added_tokens"] = self.added_tokens
        self.tokenizer_config_json["added_tokens_decoder"] = new_added_token_decoder

        # update special tokens' id
        template_processing = None
        for p in self.tokenizer_json["post_processor"]["processors"]:
            if p["type"] == "TemplateProcessing":
                template_processing = p
                break

        for k, v in template_processing["special_tokens"].items():
            assert len(v["ids"]) == 1
            updated = False
            for at in self.added_tokens:
                if at["content"] == k:
                    v["ids"] = [at["id"]]
                    updated = True
                    break
            assert updated

        if os.path.exists(save_path) == False:
            os.makedirs(save_path, exist_ok=True)

        json.dump(self.tokenizer_json, open(f"{save_path}/tokenizer.json", "w"), indent=4, ensure_ascii=False)
        json.dump(self.tokenizer_config_json, open(f"{save_path}/tokenizer_config.json", "w"), indent=4, ensure_ascii=False)
        json.dump(self.special_tokens_map_json, open(f"{save_path}/special_tokens_map.json", "w"), indent=4, ensure_ascii=False)

        self.__update_from_json_files()

    def add_token(self, plain_token):
        encoded_token = self.huggingface_tokenizer.tokenize(plain_token)
        encoded_token = "".join(encoded_token)

        self.vocab[encoded_token] = -1

        rule_added = False
        for k in self.vocab.keys():
            if len(k) < len(encoded_token) and k == encoded_token[:len(k)]:
                # assume a merge rule for k exists
                new_rule = f"{k} {encoded_token[len(k):]}"

                if encoded_token not in self.token_dicts_only_for_merges:
                    self.token_dicts_only_for_merges[encoded_token] = []

                self.token_dicts_only_for_merges[encoded_token].append(encoded_token[len(k):])
                self.add_merge(new_rule)
                rule_added = True

        if rule_added == False:
            new_rule = f"{encoded_token[0]} {encoded_token[1:]}"
            self.tokens_only_for_merges.append(encoded_token[1:])
            self.add_merge(new_rule)
    
    def remove_token(self, plain_token):
        token = "".join(self.huggingface_tokenizer.tokenize(plain_token))
        if token in self.vocab:
            del self.vocab[token]
        
        if token in self.token_dicts_only_for_merges:
            del self.token_dicts_only_for_merges[token]

        index = 0
        while True:
            if len(self.merges) <= index:
                break
            rule = self.merges[index].split(" ")
            if "".join(rule) == token or rule[0] == token or rule[1] == token:
                self.merges.pop(index)
            else:
                index += 1
    
    def add_merge(self, rule):
        self.merges.append(rule)


if __name__ == "__main__":
    token_changer = TokenChanger(llama_tokenizer)
    
    token_changer.add_token("안농")
    token_changer.remove_token("안")
    token_changer.remove_token("안농")
    token_changer.remove_token("안농")
    token_changer.save_tokenizer("updated_tokenizer")

    new_tokenizer = transformers.AutoTokenizer.from_pretrained("updated_tokenizer")
    import pdb
    pdb.set_trace()