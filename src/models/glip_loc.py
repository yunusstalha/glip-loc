# src/models/model_builder.py

import torch
import torch.nn as nn
import numpy as np

from transformers import (
    CLIPVisionModelWithProjection, 
    CLIPTextModelWithProjection, 
    CLIPTokenizer
)
import timm

class GLIPLocModel(nn.Module):
    def __init__(self, model_name, pretrained=True, use_text=False):
        """
        Args:
            model_name (str): Model identifier. Examples:
                - "openai/clip-vit-base-patch32" for CLIP
                - "convnext_base" or "convnext_small" etc. for ConvNeXt from timm
            pretrained (bool): Whether to load pretrained weights.
            use_text (bool): Whether to use text embeddings.
        """
        super().__init__()
        self.model_name = model_name.lower()
        self.use_text = use_text

        is_clip_model = 'clip' in self.model_name or 'vit' in self.model_name
        # TEXT MODEL LOADING (if enabled)
        if use_text:
            # Load the text model with projection
            self.text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

            # Get the CLIP projection dimension from text model
            # text_model.text_projection: nn.Linear(hidden_size, projection_dim)
            # The output dimension of that layer gives the final embedding size:
            self.clip_dim = self.text_model.text_projection.out_features
        else:
            self.text_model = None
            self.text_tokenizer = None
            self.clip_dim = None


        # VISION MODEL LOADING
        if is_clip_model:
            # For CLIP vision
            self.vision_model = CLIPVisionModelWithProjection.from_pretrained(model_name)
            self.vision_projection = None
        else:
            # ConvNeXt scenario
            self.vision_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

            # Determine the feature dimension of ConvNeXt
            if 'tiny' in self.model_name or 'small' in self.model_name:
                hidden_size = 768
            elif 'base' in self.model_name:
                hidden_size = 1024
            elif 'large' in self.model_name:
                hidden_size = 1536
            else:
                raise ValueError(f"Unknown ConvNeXt variant in {self.model_name}.")

            if use_text:
                # We must align convnext embeddings to CLIP dimension
                self.vision_projection = nn.Linear(hidden_size, self.clip_dim)
            else:
                # If no text is used, we can just return raw convnext embeddings or keep them as is.
                # Since no alignment is necessary, we won't project.
                self.vision_projection = None
        # Optional logit scale for contrastive training or similar tasks
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, ground_image=None, 
                satellite_image=None, 
                ground_captions=None, 
                satellite_captions=None):
        """
        Forward pass returning a tuple of 4 embeddings:
        (ground_image_embedding, satellite_image_embedding, ground_text_embedding, satellite_text_embedding)
        
        Each will be None if the corresponding input is not provided or text is disabled.

        Args:
            ground_image (torch.Tensor): [B, C, H, W] ground images
            satellite_image (torch.Tensor): [B, C, H, W] satellite images
            ground_texts (list of str): captions corresponding to ground images
            satellite_texts (list of str): captions corresponding to satellite images
        """
        ground_embedding = None
        satellite_embedding = None
        ground_text_embedding = None
        satellite_text_embedding = None

        if ground_image is not None:
            ground_embedding = self._forward_image(ground_image)
        
        if satellite_image is not None:
            satellite_embedding = self._forward_image(satellite_image)

        if self.use_text and ground_captions is not None:
            ground_text_embedding = self._forward_text(ground_captions)

        if self.use_text and satellite_captions is not None:
            satellite_text_embedding = self._forward_text(satellite_captions)

        if ground_image is not None and satellite_image is not None and self.use_text:
            return ground_embedding, satellite_embedding, ground_text_embedding, satellite_text_embedding
        elif ground_image is not None and satellite_image is not None:
            return ground_embedding, satellite_embedding
        elif ground_image is not None:
            return ground_embedding
        elif satellite_image is not None:
            return satellite_embedding
        else:
            return None

    def _forward_image(self, image):
        '''
        Forward pass for only one type of image.
        '''
        if 'clip' in self.model_name or 'vit' in self.model_name:
            outputs = self.vision_model(image, interpolate_pos_encoding=True)
            emb = outputs.image_embeds  # [B, hidden_size]
        else:  # convnext
            emb = self.vision_model(image)  # [B, hidden_size]
            if self.vision_projection is not None:
                emb = self.vision_projection(emb)
        return emb

    def _forward_text(self, texts):
        '''
        Forward pass for a list of texts.
        '''
        # Tokenize
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(next(self.parameters()).device)
        outputs = self.text_model(**inputs)
        text_emb = outputs.text_embeds

        return text_emb