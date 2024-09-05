import torch.nn as nn
import torch.nn.functional as F

from SAM_Model.adapter import LoRA
from SAM_Model.build_sam import sam_model_registry
from SAM_Model.modeling.image_encoder import window_partition, window_unpartition, add_decomposed_rel_pos


class Model(nn.Module):

    def __init__(
            self,
            cfg,
            embed_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = sam_model_registry[cfg.model_type](checkpoint=cfg.checkpoint)
        self.mask_decoder2 = sam_model_registry[cfg.model_type](checkpoint=cfg.checkpoint).mask_decoder

        self.depth = 12
        self.lora_rgb_s = nn.ModuleList()
        self.lora_depth_s = nn.ModuleList()
        for i in range(self.depth):
            lora_r = LoRA(input_dim=embed_dim,
                          output_dim=embed_dim * 3,
                          alpha=32,
                          rank=16,
                          drop_rate=0.05
                          )
            lora_d = LoRA(input_dim=embed_dim,
                          output_dim=embed_dim * 3,
                          alpha=32,
                          rank=16,
                          drop_rate=0.05
                          )
            self.lora_rgb_s.append(lora_r)
            self.lora_depth_s.append(lora_d)

    def setup(self):
        if self.cfg.freeze_image_encoder:
            print("冻结编码器")
            for param in self.model.image_encoder.parameters():
                param.requires_grad_(False)

        if self.cfg.freeze_prompt_encoder:
            print("冻结提示编码器")
            for name, param in self.model.prompt_encoder.named_parameters():
                param.requires_grad_(False)
        if self.cfg.freeze_mask_decoder:
            print("冻结解码器")
            for name, param in self.model.mask_decoder.named_parameters():
                param.requires_grad_(False)

    def forward(self, images, depths, point_prompts):
        rgb_embeddings = self.model.image_encoder.patch_embed(images)
        depth_embeddings = self.model.image_encoder.patch_embed(depths)

        if self.model.image_encoder.pos_embed is not None:
            rgb_embeddings = rgb_embeddings + self.model.image_encoder.pos_embed
            depth_embeddings = depth_embeddings + self.model.image_encoder.pos_embed

        t = 0
        for i, blk in enumerate(self.model.image_encoder.blocks):
            if i > 11:
                shortcut = rgb_embeddings
                rgb_embeddings = blk.norm1(rgb_embeddings)
                # Window partition
                if blk.window_size > 0:
                    h, w = rgb_embeddings.shape[1], rgb_embeddings.shape[2]
                    rgb_embeddings, pad_hw = window_partition(rgb_embeddings, blk.window_size)
                # x = blk.attn(x)
                B, H, W, _ = rgb_embeddings.shape
                # qkv with shape (3, B, nHead, H * W, C)
                qkv = (blk.attn.qkv(rgb_embeddings) + self.lora_rgb_s[t](rgb_embeddings)). \
                    reshape(B, H * W, 3, blk.attn.num_heads, -1).permute(2, 0, 3, 1, 4)
                # q, k, v with shape (B * nHead, H * W, C)
                q, k, v = qkv.reshape(3, B * blk.attn.num_heads, H * W, -1).unbind(0)
                attn = (q * blk.attn.scale) @ k.transpose(-2, -1)
                if blk.attn.use_rel_pos:
                    attn = add_decomposed_rel_pos(attn, q, blk.attn.rel_pos_h, blk.attn.rel_pos_w, (H, W), (H, W))
                attn = attn.softmax(dim=-1)
                rgb_embeddings = (attn @ v).view(B, blk.attn.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
                rgb_embeddings = blk.attn.proj(rgb_embeddings)

                # Reverse window partition
                if blk.window_size > 0:
                    rgb_embeddings = window_unpartition(rgb_embeddings, blk.window_size, pad_hw, (h, w))

                rgb_embeddings = shortcut + rgb_embeddings
                rgb_embeddings = rgb_embeddings + blk.mlp(blk.norm2(rgb_embeddings))

                depthcut = depth_embeddings
                depth_embeddings = blk.norm1(depth_embeddings)
                # Window partition
                if blk.window_size > 0:
                    h, w = depth_embeddings.shape[1], depth_embeddings.shape[2]
                    depth_embeddings, pad_hw = window_partition(depth_embeddings, blk.window_size)
                # x = blk.attn(x)
                B, H, W, _ = depth_embeddings.shape
                # qkv with shape (3, B, nHead, H * W, C)
                qkv = (blk.attn.qkv(depth_embeddings) + self.lora_depth_s[t](depth_embeddings)). \
                    reshape(B, H * W, 3, blk.attn.num_heads, -1).permute(2, 0, 3, 1, 4)
                # q, k, v with shape (B * nHead, H * W, C)
                q, k, v = qkv.reshape(3, B * blk.attn.num_heads, H * W, -1).unbind(0)
                attn = (q * blk.attn.scale) @ k.transpose(-2, -1)
                if blk.attn.use_rel_pos:
                    attn = add_decomposed_rel_pos(attn, q, blk.attn.rel_pos_h, blk.attn.rel_pos_w, (H, W), (H, W))
                attn = attn.softmax(dim=-1)
                depth_embeddings = (attn @ v).view(B, blk.attn.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
                depth_embeddings = blk.attn.proj(depth_embeddings)

                # Reverse window partition
                if blk.window_size > 0:
                    depth_embeddings = window_unpartition(depth_embeddings, blk.window_size, pad_hw, (h, w))

                depth_embeddings = depthcut + depth_embeddings
                depth_embeddings = depth_embeddings + blk.mlp(blk.norm2(depth_embeddings))

                t = t + 1
            else:
                rgb_embeddings = blk(rgb_embeddings)
                depth_embeddings = blk(depth_embeddings)

        rgb_embeddings = self.model.image_encoder.neck(rgb_embeddings.permute(0, 3, 1, 2))
        depth_embeddings = self.model.image_encoder.neck(depth_embeddings.permute(0, 3, 1, 2))
        fuse_embeddings = rgb_embeddings + depth_embeddings

        dense_pe = self.model.prompt_encoder.get_dense_pe()
        dense_pe2 = self.model.prompt_encoder.get_dense_pe()

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )

        low_res_masks = self.model.mask_decoder(
            image_embeddings=fuse_embeddings,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        masks1 = F.interpolate(low_res_masks, (1024, 1024), mode="bilinear", align_corners=False)

        sparse_embeddings2, dense_embeddings2 = self.model.prompt_encoder(
            points=point_prompts,
            # points=None,
            boxes=None,
            masks=None,
        )

        low_res_masks2 = self.mask_decoder2(
            image_embeddings=fuse_embeddings,
            image_pe=dense_pe2,
            sparse_prompt_embeddings=sparse_embeddings2,
            dense_prompt_embeddings=dense_embeddings2,
            multimask_output=False,
        )

        masks2 = F.interpolate(low_res_masks2, (1024, 1024), mode="bilinear", align_corners=False)

        return masks1, masks2

