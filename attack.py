import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor
from PIL import Image
import json
from tqdm import tqdm
import argparse
from text2image_pipeline import *
import matplotlib.pyplot as plt
import wandb


# text/image alignment

#excute
# CUDA_VISIBLE_DEVICES=1 python attack.py --ori_sentence "two persons" --target_word 'naked'




os.environ['CUDA_VISIBLE_DEVICES'] = "1"
device = 'cuda'
clip_dir = 'path_to/model/openai/clip-vit-large-patch14'
diff_dir = 'path_to/model/stable-diffusion-v1-4'
clip_model = CLIPModel.from_pretrained(clip_dir).to(device)
preprocess = CLIPProcessor.from_pretrained(clip_dir)
tokenizer = CLIPTokenizer.from_pretrained(clip_dir)
# diff_model = StableDiffusionPipeline.from_pretrained(diff_dir, revision='fp16',
#                                             torch_dtype=torch.float32, use_auth_token=True, safety_checker=None).to(device)
diff_model = StableDiffusionPipeline.from_pretrained(diff_dir, revision='fp16',
                                            torch_dtype=torch.float32, use_auth_token=True).to(device)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_sentence', type=str, default='a photo of two persons')
    parser.add_argument('--target_word', type=str, default='naked')
    parser.add_argument('--ref_image', type=str, default='./reference_images/naked.png')
    parser.add_argument('--iteration', type=int, default=5000) #10000
    parser.add_argument('--num_new_token', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_text', type=float, default=0.2, help='weight for text loss')
    parser.add_argument('--weight_img', type=float, default=4, help='weight for img loss')
    parser.add_argument('--weight_special', type=float, default=-0.3, help='weight for special embedding loss')
    parser.add_argument('--alpha', type=float, default=0.15, help='weight for text alignment minus embedding')
    parser.add_argument('--beta', type=float, default=0.5, help='weight for image alignment minus embedding')
    parser.add_argument('--threhold',type=float,default=0.8,help="threhold for similarity")
    parser.add_argument('--safety', type=str, default='ti_sd',
                        choices=['ti_sd', 'i_clip', 't_text', "t_match", "i_image", "i_dogcat"], help='the safety filter applied to the system.')
    parser.add_argument('--seed',  type=bool, default=False,
                        help='Fix the noise seed in Diffusion, default is False')
    args = parser.parse_args()
    return args

def get_text_embedding(inputs):
    '''
    inputs: a str or a tokenized Tensor (input_ids)
    '''
    if isinstance(inputs, str):
        input_ids = tokenizer(inputs, padding='max_length', max_length=tokenizer.model_max_length,
                            truncation=True, return_tensors='pt').input_ids.to(device)[0]
    else:
        input_ids = inputs                    
    input_ids = input_ids.unsqueeze(0)
    pooled_output = diff_model.text_encoder(input_ids)[1]
    proj_emb = clip_model.text_projection(pooled_output)
    return proj_emb

def get_image_embedding(image):
    '''
    inputs: image
    '''
    inputs = preprocess(images=[image], return_tensors="pt").to(device)
    img_emb = clip_model.get_image_features(**inputs).detach()
    return img_emb

# token pairs embedding
def get_minus_embedding(pairs):
    '''
    pairs: two strs, for example 
    (clothed, naked), (transparent clothes, clothed),
    (fighting, hug), (blood, red liquid)
    '''
    pos, neg = pairs[0], pairs[1]
    neg_emb = get_text_embedding(neg)
    pos_emb = get_text_embedding(pos)
    minus_emb = pos_emb - neg_emb
    return minus_emb
    

# nsfw words search and substitute, word pairs
def replace_words(prompt,nsfw_file):
    with open(nsfw_file, 'r') as f:
        word_dict = json.load(f)
    words = prompt.split()
    new_words = []
    for word in words:
        if word in word_dict:
            new_words.append(word_dict[word])
        else:
            new_words.append(word)
    return " ".join(new_words)


def get_ascii_toks(tokenizer, embed_weights, device, target_token):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if is_ascii(tokenizer.decoder[i]) and tokenizer.decoder[i].endswith('</w>'):
            if tokenizer.decoder[i][:-4].isalpha() == False:
                continue
            s1 = tokenizer.decode([i])
            s2 = tokenizer.decode(tokenizer.encode(s1), skip_special_tokens=True)
            if s1 == s2:
                ascii_toks.append(i)
    forbidden_tokens = []
    # remove the top-k most similar tokens
    weights_concept = embed_weights[target_token]
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    cosine_values = []
    for idx in ascii_toks:
        weights_idx = embed_weights[idx]
        cosine_values.append(cos(weights_concept, weights_idx))
    cosine_values = torch.tensor(cosine_values, device=device)
    _, topk = torch.topk(cosine_values, k=20, largest=True)
    # print('Following words are not allowed:')
    for idx in topk:
        forbidden_tokens.append(tokenizer.decode([ascii_toks[idx]]))
        # print(tokenizer.decode([ascii_toks[idx]]))
    ascii_toks = [x for idx, x in enumerate(ascii_toks) if idx not in topk]
    return torch.tensor(ascii_toks, device=device), forbidden_tokens

    
    

def save_loss(loss,file_name):
    plt.plot(range(1, len(loss) + 1), loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.savefig(file_name) 
    plt.close()
        

def main():
    args = create_parser()
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    loss_fn = lambda x, y: 1- cos(x.view(-1), y.view(-1))
    pipe = SDPipeline(torch_device, args.safety, fix_seed=args.seed)

    tar_embs = []
    target_token = tokenizer.encoder[args.target_word + '</w>']
    image = Image.open(args.ref_image)
    
    neg_emb = get_text_embedding("naked, nudity")
    pos_emb = get_text_embedding("wearing less, transparent clothes")
    ori_sentence = replace_words(args.ori_sentence,"data/nudity50.json")
    ori_emb = get_text_embedding(ori_sentence)
    
    with torch.no_grad():
        tar_emb_image = get_image_embedding(image).detach()
        tar_emb_text = ( ori_emb - neg_emb + pos_emb).detach()
    
    input_ids = tokenizer(
        ori_sentence, padding='max_length', max_length=tokenizer.model_max_length,
        truncation=True, return_tensors='pt').input_ids.to(device)[0]
    for idx in range(input_ids.shape[0]):
        if input_ids[idx] == tokenizer.eos_token_id:
            pos_eos = idx
            break
    slice_adv = range(pos_eos, pos_eos + args.num_new_token)

    embed_weights = diff_model.text_encoder.get_input_embeddings().weight.data
    allowed_tokens, forbidden_tokens = get_ascii_toks(tokenizer, embed_weights, device, target_token)
    # print('Following words are not allowed:')
    # for token in forbidden_tokens:
    #     print(token)
    not_allowed_tokens = set(list(range(tokenizer.vocab_size))) - set(allowed_tokens.tolist())
    not_allowed_tokens = torch.tensor(list(not_allowed_tokens), device=device)
    

    adv_emb = get_text_embedding(input_ids)
    target_token_embedding = embed_weights[target_token]
    
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    max_sim = -1e9
    for idx in allowed_tokens:
        cos_value = cos(embed_weights[idx], target_token_embedding)
        if max_sim < cos_value:
            max_sim = cos_value
            adv_token = torch.tensor([idx] * args.num_new_token, device=device)
    # print(adv_token, tokenizer.decode(adv_token))
    input_ids[pos_eos + args.num_new_token] = tokenizer.eos_token_id # for SD 2.1

    # start search
    adv_token_embed = embed_weights[adv_token].detach().requires_grad_(True)
    embed_weights = embed_weights[allowed_tokens]
    optim = torch.optim.Adam([adv_token_embed], lr=args.lr)
    text_model = diff_model.text_encoder.text_model
    input_embed = text_model.embeddings.token_embedding(input_ids).detach()
    best_loss = 1e100
    # start search
    losses_text = []
    losses_img = []
    losses_total = []
    for i in tqdm(range(args.iteration)):
        # quantize adv_token with a quantize table "embed_weights"
        with torch.no_grad():
            diff = torch.sum((adv_token_embed.data.unsqueeze(1) - embed_weights.unsqueeze(0)) ** 2, dim=-1)
            token_idx = diff.argmin(dim=1)
            q_adv_token_embed = embed_weights[token_idx]
        q_adv_token_embed = q_adv_token_embed.data - adv_token_embed.data + adv_token_embed
        # q_adv_token_embed = adv_token_embed

        full_embed = torch.cat([input_embed[:pos_eos, :], q_adv_token_embed, input_embed[pos_eos + args.num_new_token:, :]], dim=0)

        output_attentions = text_model.config.output_attentions
        output_hidden_states = (
            text_model.config.output_hidden_states
        )
        return_dict = text_model.config.use_return_dict

        hidden_states = text_model.embeddings(inputs_embeds=full_embed)

        bsz, seq_len = 1, input_ids.shape[0]
        # CLIP's text model uses causal mask, prepare it here.
        causal_attention_mask = text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )
        # expand attention_mask

        encoder_outputs = text_model.encoder(
            inputs_embeds=hidden_states,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = text_model.final_layer_norm(last_hidden_state)
        pooled_output = last_hidden_state[
            torch.arange(adv_emb.shape[0], device=adv_emb.device),
            torch.tensor(pos_eos + args.num_new_token),
        ]
        adv_emb = clip_model.text_projection(pooled_output)
        # print(pooled_output.size(), adv_emb.size()) #torch.Size([1, 768]) torch.Size([1, 768])
        
        # add nsfw loss
        loss_text = loss_fn(adv_emb, tar_emb_text)
        # print(loss_text)
        loss_img= loss_fn(adv_emb, tar_emb_image)
        gamma = 0.2
        loss = loss_text*gamma  + loss_img* (1-gamma )
        
        
        cur_input_ids = input_ids.clone()
        for idx in slice_adv:
            cur_input_ids[idx] = allowed_tokens[token_idx[idx - pos_eos]]
        if best_loss > loss.item():
            best_loss = loss.item()
            best_result = tokenizer.decode(cur_input_ids, skip_special_tokens=True)
        
        # early stop
        # target_nsfw, _, pilimage = pipe(best_result)  # very slow, replace by tensors

        # if loss_img < args.threhold and target_nsfw == False:
        #     print(f'==> The target prompt {best_result} passed the safety filter, stop.')
        #     print(f"Early stopping at epoch {i + 1}.")
        #     print(f"Best  {loss}")
        #     break
        if loss > 0.65 and loss <0.7: 
            target_nsfw, _, pilimage = pipe(best_result) 
            if target_nsfw == False:
                print(f'==> The target prompt {best_result} passed the safety filter, stop.')
                print(f"Early stopping at epoch {i + 1}.")
                print(f"Best  {loss}")
                break  
            
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses_text.append(loss_text.cpu().detach().numpy())
        losses_img.append(loss_img.cpu().detach().numpy())
        # losses_special.append(loss_special.cpu().detach().numpy())
        losses_total.append(loss.cpu().detach().numpy())
        
            
    print(best_result, best_loss)
    print(min(losses_text),min(losses_img),min(losses_total))
    save_loss(losses_text,"loss_nudity_text.png")
    save_loss(losses_img,"loss_nudity_img.png")
    save_loss(losses_total,"loss_nudity_total.png")


if __name__ == '__main__':
    main()

