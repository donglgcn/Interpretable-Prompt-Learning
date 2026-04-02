import json
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from datasets.textDistribution import Context_sentences, Context_sentences_caption

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP_backup(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


class PromptLearner_descriptor(nn.Module):
    def __init__(self, cfg, class_descriptors, clip_model):
        super().__init__()
        n_cls = sum([len(l) for l in class_descriptors.values()])

        n_ctx_label = cfg.TRAINER.COOP.N_CTX_LABEL
        n_ctx_descriptor = cfg.TRAINER.COOP.N_CTX_DESCRIPTOR
        ctx_init_label = cfg.TRAINER.COOP.CTX_INIT_LABEL
        ctx_init_descriptor = cfg.TRAINER.COOP.CTX_INIT_DESCRIPTOR
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        weights = torch.ones(n_cls, dtype=dtype)
        self.weights = nn.Parameter(weights)
        if ctx_init:  # e.g. an image of a {label}, which is {descriptor}
            # use given words to initialize context vectors
            ctx_init_label = ctx_init_label.replace("_", " ")
            ctx_init_descriptor = ctx_init_descriptor.replace("_", " ")
            n_ctx_label = len(ctx_init_label.split(" "))
            n_ctx_descriptor = len(ctx_init_descriptor.split(" "))
            prompt_label = clip.tokenize(ctx_init_label)
            prompt_descriptor = clip.tokenize(ctx_init_descriptor)
            with torch.no_grad():
                embedding_label = clip_model.token_embedding(prompt_label).type(dtype)
                embedding_descriptor = clip_model.token_embedding(prompt_descriptor).type(dtype)
            ctx_vectors_label = embedding_label[0, 1: 1 + n_ctx_label, :]
            ctx_vectors_descriptor = embedding_descriptor[0, 1: 1 + n_ctx_descriptor, :]
            prompt_prefix_label = ctx_init_label
            prompt_prefix_descriptor = ctx_init_descriptor

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors_label = torch.empty(n_cls, n_ctx_label, ctx_dim, dtype=dtype)
                ctx_vectors_descriptor = torch.empty(n_cls, n_ctx_descriptor, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_label = torch.empty(n_ctx_label, ctx_dim, dtype=dtype)
                ctx_vectors_descriptor = torch.empty(n_ctx_descriptor, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_label, std=0.02)
            nn.init.normal_(ctx_vectors_descriptor, std=0.02)
            prompt_prefix_label = " ".join(["X"] * n_ctx_label)
            prompt_prefix_descriptor = " ".join(["X"] * n_ctx_descriptor)
            # prompt_prefix = " ".join(["X"] * n_ctx)

        print(
            f'Initial label context: "{prompt_prefix_label}"; Initial descriptor context: "{prompt_prefix_descriptor}"')
        print(f"Number of context words (tokens): {n_ctx_label + n_ctx_descriptor}")

        self.ctx_label = nn.Parameter(ctx_vectors_label)  # to be optimized
        self.ctx_descriptor = nn.Parameter(ctx_vectors_descriptor)  # to be optimized

        propmts = []
        # label_prompts = []
        # descriptor_prompts = []
        name_lens = []
        descriptor_lens = []
        for (key, values) in class_descriptors.items():
            name = key.replace("_", " ")
            name_len = len(_tokenizer.encode(name))
            for v in values:
                descriptor = v.replace("_", " ")
                descriptor_len = len(_tokenizer.encode(descriptor))
                propmts.append(
                    prompt_prefix_label + " " + name + " " + prompt_prefix_descriptor + " " + descriptor + ".")
                # label_prompts.append(prompt_prefix_label + " " + name)
                # descriptor_prompts.append(prompt_prefix_descriptor + " " + descriptor + ".")
                name_lens.append(name_len)
                descriptor_lens.append(descriptor_len)

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in propmts])
        # tokenized_label_prompts = torch.cat([clip.tokenize(p) for p in label_prompts])
        # tokenized_descriptor_prompts = torch.cat([clip.tokenize(p) for p in descriptor_prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            # label_embedding = clip_model.token_embedding(tokenized_label_prompts).type(dtype)
            # descriptor_embedding = clip_model.token_embedding(tokenized_descriptor_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.token_label = [embedding[i:i + 1, 1 + n_ctx_label:1 + n_ctx_label + name_lens[i]] for i in
                            range(len(name_lens))]
        self.token_suffix = [embedding[i:i + 1, 1 + n_ctx_label + name_lens[i] + n_ctx_descriptor:] for i in
                             range(len(descriptor_lens))]
        # self.register_buffer("token_label", torch.cat([embedding[i:i+1, 1 + n_ctx_label:1 + n_ctx_label+name_lens[i]] for i in range(len(name_lens))]))  # CLS
        # self.register_buffer("token_suffix", torch.cat([embedding[i:i+1, 1 + n_ctx_descriptor:1 + n_ctx_descriptor+descriptor_lens[i]] for i in range(len(descriptor_lens))]))  # DES, EOS

        # self.register_buffer("token_prefix", label_embedding[:, :1, :])  # SOS
        # self.register_buffer("token_label", label_embedding[:, 1 + n_ctx_label:-1, :])  # CLS
        # self.register_buffer("token_suffix", descriptor_embedding[:, 1 + n_ctx_descriptor:, :])  # DES, EOS
        # self.register_buffer("token_suffix", embedding[:, 1 + n_ctx_label:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx_label = n_ctx_label
        self.n_ctx_descriptor = n_ctx_descriptor
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.descriptor_lens = descriptor_lens
        # self.class_token_position = cfg.CLASS_TOKEN_POSITION

    def forward(self):
        ctx_label = self.ctx_label
        ctx_descriptor = self.ctx_descriptor
        if ctx_label.dim() == 2:
            ctx_label = ctx_label.unsqueeze(0).expand(self.n_cls, -1, -1)
        if ctx_descriptor.dim() == 2:
            ctx_descriptor = ctx_descriptor.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        token_label = self.token_label
        suffix = self.token_suffix

        prompts = []
        for i in range(self.n_cls):
            prefix_i = prefix[i: i + 1, :, :]
            ctx_label_i = ctx_label[i: i + 1, :, :]
            ctx_descriptor_i = ctx_descriptor[i: i + 1, :, :]
            token_label_i = token_label[i].to(prefix_i.device)
            suffix_i = suffix[i].to(prefix_i.device)
            prompt = torch.cat(
                [
                    prefix_i,  # (1, 1, dim)
                    ctx_label_i,  # (1, n_ctx_label, dim)
                    token_label_i,  # (1, name_len*, dim)
                    ctx_descriptor_i,  # (1, n_ctx_descriptor, dim)
                    suffix_i,  # (1, descriptor_len+1*, dim)
                ],
                dim=1,
            )
            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)
        return prompts


class TextClassifier(nn.Module):
    """
    for each label create a local threshold to determine whether go deep or not.
    """

    def __init__(self, input_dim):
        super(TextClassifier, self).__init__()
        hidden_dim = 2 * input_dim
        self.classifier = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, feature):
        """
        :param feature: text features mat, dim N*d
        :return:
        """
        return self.classifier(feature)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, class_descriptor, clip_model):
        super().__init__()
        if cfg.MODEL.BACKBONE.NAME == "ViT-L/14":
            input_dim = 768
            descriptor_model_path = "./textDiscrimiator/VITL14/textClassifier.pt"
        elif cfg.MODEL.BACKBONE.NAME == "RN50":
            input_dim = 1024
            descriptor_model_path = "./textDiscrimiator/RN50/textClassifier.pt"
        elif cfg.MODEL.BACKBONE.NAME == "ViT-B/16":
            input_dim = 512
            descriptor_model_path = "./textDiscrimiator/VITB16/textClassifier.pt"
        elif cfg.MODEL.BACKBONE.NAME == "ViT-B/32":
            input_dim = 512
            descriptor_model_path = "./textDiscrimiator/VITB32/textClassifier.pt"
        textClassifier = TextClassifier(input_dim=input_dim)
        textClassifier.load_state_dict(torch.load(descriptor_model_path))
        self.discriminator_model = textClassifier
        for param in self.discriminator_model.parameters():
            param.requires_grad_(False)
        self.class_descriptor = class_descriptor
        self.prompt_learner = PromptLearner_descriptor(cfg, class_descriptor, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.clip_text_encoder = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.test = cfg.TRAINER.COOP.EVAL_ONLY
        self.dataset = cfg.DATASET.NAME

    def forward(self, image, train=False):
        if not train:
            image_features = self.image_encoder(image.type(self.dtype))
        else:
            image_features = image.to(self.dtype)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        len_list = [len(v) for v in self.class_descriptor.values()]
        weights = self.prompt_learner.weights
        # random sample 2 text descriptors to train in order to deal with the OOM issue.
        if not self.test and self.dataset == "ImageNet":
            prompts = torch.split(prompts, len_list, dim=0)
            weights = torch.split(weights, len_list)
            tokenized_prompts = torch.split(tokenized_prompts, len_list, dim=0)
            indices = [torch.randperm(len(prompt))[:1] for prompt in prompts]
            prompts_new = [prompt[indices[i]] for i, prompt in enumerate(prompts)]
            weights_new = [weight[indices[i]] for i, weight in enumerate(weights)]
            tokenized_prompts_new = [tokenized_prompt[indices[i]] for i, tokenized_prompt in
                                     enumerate(tokenized_prompts)]
            prompts = torch.cat(prompts_new, dim=0)
            weights = torch.cat(weights_new)
            tokenized_prompts = torch.cat(tokenized_prompts_new, dim=0)
            len_list = [1 for v in self.class_descriptor.values()]

        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        text_features = torch.split(text_features, len_list, dim=0)
        weights = torch.split(weights, len_list)
        # if you don't want to have weighted sum, use this line.
        # text_features_mean = [torch.mean(feature, dim=0, keepdim=True) for feature in text_features]
        # ###########################
        text_features_mean = [torch.sum(torch.mul(feature, weight[:, None]), dim=0) for feature, weight in
                              zip(text_features, weights)]
        text_features = [feature / feature.norm() for feature in text_features_mean]
        text_features = torch.stack(text_features, dim=0)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        self.discrimator_logits = self.discriminator_model(text_features.to(torch.float32))

        return logits


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        class_descriptor = json.load(open(cfg.DATASET.class_descriptor_path, 'r'))
        class_captions = json.load(open(os.path.join(cfg.DATASET.class_captions_path, f"seed{cfg.SEED}.json"), 'r'))

        context_sentences = Context_sentences_caption(cfg, class_descriptor, class_captions, clip_model)
        self.train_loader_x = torch.utils.data.DataLoader(
            context_sentences,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
            shuffle=True
        )

        self.model = CustomCLIP(cfg, class_descriptor, clip_model)
        self.alpha = cfg.TRAINER.COOP.alpha
        self.beta = cfg.TRAINER.COOP.beta

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        self.save_model(-1, os.path.join(self.output_dir, 'init'))

    def forward_backward(self, batch):
        image, label = batch
        image = image.to(self.device)
        label = label.to(self.device)

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image, train=True)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image, train=True)
            loss_ce = F.cross_entropy(output, label)
            labels = torch.ones_like(self.model.discrimator_logits, dtype=self.model.discrimator_logits.dtype)
            loss_bce = F.binary_cross_entropy(self.model.discrimator_logits, labels)
            loss = self.alpha * loss_bce + self.beta * loss_ce
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss_bce": loss_bce.item(),
            "loss_ce": loss_ce.item(),
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)
            self.test()

