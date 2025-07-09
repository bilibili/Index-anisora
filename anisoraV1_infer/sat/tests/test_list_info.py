from sat.helpers import list_avail_models, list_avail_pretrained

def test_list_avail_models():
    models = list_avail_models()

def test_list_avail_pretrained():
    models = list_avail_pretrained()

if __name__ == '__main__':
    test_list_avail_models()
    test_list_avail_pretrained()