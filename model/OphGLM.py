import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BertTokenizer, BertModel
from torchvision import models
from .convnextv1.convnext_ml_sb3 import convnext_base_ml_sb3


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim):
        super(ImageEncoder, self).__init__()
        self.encoder = convnext_base_ml_sb3()
        self.encoder.head = nn.Linear(1024, embed_dim)  
        self.embed_dim = embed_dim

    def forward(self, images):
        features = self.encoder(images)
        features = F.normalize(features, p=2, dim=-1)
        return features


class OphGLM(nn.Module):
    def __init__(self, image_embed_dim, text_embed_dim, fusion_dim):
        super(OphGLM, self).__init__()
        self.image_encoder = ImageEncoder(embed_dim=image_embed_dim)
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.text_model = AutoModelForCausalLM.from_pretrained("THUDM/chatglm-6b")
        self.fusion_layer1 = nn.Linear(image_embed_dim + text_embed_dim, fusion_dim)
        self.fusion_layer2= nn.Linear(fusion_dim, fusion_dim)
        self.fusion_layer3= nn.Linear(fusion_dim, fusion_dim)
        self.fusion_layer4= nn.Linear(fusion_dim, fusion_dim)
        self.fusion_layer5= nn.Linear(fusion_dim, fusion_dim)
        self.fusion_layer6= nn.Linear(fusion_dim, text_embed_dim)

    def forward(self, images, input_ids, attention_mask):
      
        image_features = self.image_encoder(images)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        text_features = text_outputs.hidden_states[-1][:, 0, :]
    
        combined_features = torch.cat([image_features, text_features], dim=-1)
        fused_features = torch.relu(self.fusion_layer1(combined_features))
        fused_features = torch.relu(self.fusion_layer2(fused_features))
        fused_features = torch.relu(self.fusion_layer3(fused_features))
        fused_features = torch.relu(self.fusion_layer4(fused_features))
        fused_features = torch.relu(self.fusion_layer5(fused_features))
        fused_to_text = self.fusion_layer6(fused_features)
        
        outputs = self.text_model.generate(inputs_embeds=fused_to_text.unsqueeze(1))
        
        return outputs