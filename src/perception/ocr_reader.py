class OCRReader:
    def __init__(self, langs=None, gpu=False):
        self.langs = langs or ["vi", "en"]
        self.gpu = gpu

    def read_text(self, image):
        return {
            "texts": [],
            "joined_text": "",
            "mean_conf": 0.0,
            "text_len": 0,
            "num_tokens": 0,
        }