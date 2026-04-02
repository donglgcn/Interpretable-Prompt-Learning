import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from clip import clip

# randomly combine class with some attributes. They will represent image distribution hopefully.
class Context_sentences(Dataset):
    @torch.no_grad()
    def __init__(self, cfg, class_descriptor, class_attributes, class_contexts, clip_model):
        classnames = list(class_descriptor.keys())
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.sentences = []
        self.labels = []
        self.embeddings = torch.tensor([])
        self.dtype = clip_model.dtype
        clip_model = clip_model.to("cuda")

        self.get_embedding_path(cfg)

        if os.path.exists(self.embedding_path) and os.path.exists(self.label_path):
            print("loaded form ", self.embeddings)
            self.embeddings = torch.load(self.embedding_path)
            self.labels = pickle.load(open(self.label_path, 'rb'))
        else:
            for label, name in enumerate(classnames):
                classname, be = self.getclassname(cfg, name)
                attributes = class_attributes[name]
                contexts = class_contexts[name]
                for attribute in attributes:
                    attribute = attribute.replace("_", " ")
                    sentences = []
                    for context in contexts:
                        context = context.replace("_", " ")
                        sentence = "The " + attribute + " " + classname + be + context + "."
                        sentences.append(sentence)
                        self.labels.append(label)
                    self.sentences.extend(sentences)
                    sentences = clip.tokenize(sentences).to("cuda")
                    embeddings = clip_model.encode_text(sentences).cpu()
                    self.embeddings = torch.cat([self.embeddings, embeddings])
            assert len(self.embeddings) == len(self.labels)
            torch.save(self.embeddings, self.embedding_path)
            pickle.dump(self.labels, open(self.label_path, 'wb'))
            print("saved at ", self.embedding_path)
        assert len(self.embeddings) == len(self.labels)
        clip_model = clip_model.to("cpu")

    def get_embedding_path(self, cfg):
        path = cfg.DATASET.class_embeddings_path
        if cfg.MODEL.BACKBONE.NAME == "ViT-L/14":
            model = "VITL14"
        elif cfg.MODEL.BACKBONE.NAME == "RN50":
            model = "RN50"
        elif cfg.MODEL.BACKBONE.NAME == "ViT-B/16":
            model = "VITB16"
        elif cfg.MODEL.BACKBONE.NAME == "ViT-B/32":
            model = "VITB32"
        else:
            exit("no such clip model")
        path = os.path.join(path, model)
        os.makedirs(path, exist_ok=True)
        if not cfg.DATASET.general_class:
            self.embedding_path = os.path.join(path, "embeddings.pt")
            self.label_path = os.path.join(path, "label.pkl")
        else:
            self.embedding_path = os.path.join(path, "embeddings_general.pt")
            self.label_path = os.path.join(path, "label_general.pkl")

    def getclassname(self, cfg, name):
        classname = name.replace("_", " ")
        if cfg.DATASET.NAME == "Food101":
            # classname = "food"
            be = " is "
        elif cfg.DATASET.NAME == "DescribableTextures":
            # classname = "textual"
            be = " textual is "
        elif cfg.DATASET.NAME == "OxfordPets":
            # classname = "pet"
            be = " is "
        elif cfg.DATASET.NAME == "EuroSAT":
            # classname = "landscape"
            be = " has "
        else:
            # classname = name.replace("_", " ")
            be = " is "
        if not cfg.DATASET.general_class:
            be = " is "
        return classname, be

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, item):
        return self.embeddings[item], self.labels[item]


class Context_sentences_caption(Dataset):
    @torch.no_grad()
    def __init__(self, cfg, class_descriptor, class_captions, clip_model):
        classnames = list(class_descriptor.keys())
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.sentences = []
        self.labels = []
        self.embeddings = torch.tensor([])
        self.dtype = clip_model.dtype
        clip_model = clip_model.to("cuda")

        self.get_embedding_path(cfg)

        if False and os.path.exists(self.embedding_path) and os.path.exists(self.label_path):
            print("loaded from ", self.embeddings)
            self.embeddings = torch.load(self.embedding_path)
            self.labels = pickle.load(open(self.label_path, 'rb'))
        else:
            for label, name in enumerate(classnames):
                classname = name
                sentences = class_captions.get(name, None)
                if sentences is None:
                    name = name.lower().replace(" ", "_")
                    sentences = class_captions[name]
                # sentences = ["The " + classname + " is like " + sentence + "." for sentence in sentences]
                sentences = clip.tokenize(sentences[:], truncate=True).to("cuda")
                embeddings = clip_model.encode_text(sentences).cpu()
                self.embeddings = torch.cat([self.embeddings, embeddings])
                self.labels.extend([label]*len(sentences))
            assert len(self.embeddings) == len(self.labels)
            torch.save(self.embeddings, self.embedding_path)
            pickle.dump(self.labels, open(self.label_path, 'wb'))
            print("saved at ", self.embedding_path)
        assert len(self.embeddings) == len(self.labels)
        clip_model = clip_model.to("cpu")

    def get_embedding_path(self, cfg):
        path = cfg.DATASET.class_embeddings_path
        if cfg.MODEL.BACKBONE.NAME == "ViT-L/14":
            model = "VITL14"
        elif cfg.MODEL.BACKBONE.NAME == "RN50":
            model = "RN50"
        elif cfg.MODEL.BACKBONE.NAME == "ViT-B/16":
            model = "VITB16"
        elif cfg.MODEL.BACKBONE.NAME == "ViT-B/32":
            model = "VITB32"
        else:
            exit("no such clip model")
        path = os.path.join(path, model)
        os.makedirs(path, exist_ok=True)
        self.embedding_path = os.path.join(path, "embeddings_captions.pt")
        self.label_path = os.path.join(path, "label_captions.pkl")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, item):
        return self.embeddings[item], self.labels[item]
