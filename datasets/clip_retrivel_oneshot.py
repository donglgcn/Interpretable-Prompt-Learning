import json
import os.path
import pickle

from clip_retrieval.clip_client import ClipClient, Modality

client_image = ClipClient(
    url="https://knn.laion.ai/knn-service",
    indice_name="laion_400m",
    aesthetic_score=9,
    aesthetic_weight=0.0,
    modality=Modality.IMAGE,
    num_images=40,
    use_safety_model = False,
    use_violence_detector = False,
)
client_text = ClipClient(
    url="https://knn.laion.ai/knn-service",
    indice_name="laion_400m",
    aesthetic_score=9,
    aesthetic_weight=0.0,
    modality=Modality.TEXT,
    num_images=100,
    use_safety_model = False,
    use_violence_detector = False,
)

# query by image
# img_path = "/media/dongliang/10TB Disk/datasets/eurosat/2750/AnnualCrop/AnnualCrop_1.jpg"
def query_by_image(img_path):
    query_results = []
    # image_query_results = client_image.query(image=img_path)
    text_query_results = client_text.query(image=img_path)
    # query_results.extend(image_query_results)
    query_results.extend(text_query_results)
    print(len(query_results))
    captions = [result["caption"] for result in query_results]
    return captions
# log_result(beach_results[0])

if __name__ == '__main__':
    dirs = [
         "/data/dongliang/datasets/eurosat/split_fewshot/",
             "/data/dongliang/datasets/dtd/dtd/split_fewshot/",
             "/data/dongliang/datasets/oxford_pets/split_fewshot/",
            "/data/dongliang/datasets/food-101/split_fewshot/"]
    jsons = [
        "../imagedistribution_attributes/eurosat/",
        "../imagedistribution_attributes/dtd/",
        "../imagedistribution_attributes/pets/",
        "../imagedistribution_attributes/food/",
             ]
    for dir, json_path in zip(dirs, jsons):
        for i in [1,2,3]:
            path = os.path.join(dir, f"shot_1-seed_{i}.pkl")
            data = pickle.load(open(path, "rb"))['train']

            caption = {}
            jsonname = f"{json_path}text100_seed{i}.json"
            for sample in data:
                classname = sample.classname
                impath = sample.impath
                captions = query_by_image(impath)
                caption[classname] = captions
            json.dump(caption,open(jsonname,'w'))
            print(jsonname+" done.")
