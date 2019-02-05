from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
import uuid

from fastai import *
from fastai.vision import *
from PIL import Image
import torch
defaults.device = torch.device('cpu')
export_file_url = 'https://www.dropbox.com/s/qn2f12rjwicu8p4/more_iters.pkl?raw=1'
export_file_name = 'export.pkl'

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

def load_callback(class_func, state, learn:Learner):
    init_kwargs, others = split_kwargs_by_func(state, class_func.__init__)
    res = class_func(learn, **init_kwargs) if issubclass(class_func, LearnerCallback) else class_func(**init_kwargs)
    for k,v in others.items(): setattr(res, k, v)
    return res

def load_learner_cpu(path:PathOrStr, fname:PathOrStr='export.pkl', test:ItemList=None):
    "Load a `Learner` object saved with `export_state` in `path/fn` with empty data, optionally add `test` and load on `cpu`."
    state = torch.load(open(Path(path)/fname, 'rb'), map_location=torch.device('cpu'))
    model = state.pop('model')
    src = LabelLists.load_state(path, state.pop('data'))
    if test is not None: src.add_test(test)
    data = src.databunch()
    cb_state = state.pop('cb_state')
    clas_func = state.pop('cls')
    res = clas_func(data, model, **state)
    res.callback_fns = state['callback_fns'] #to avoid duplicates
    res.callbacks = [load_callback(c,s, res) for c,s in cb_state.items()]
    return res
all_classes = labels = [
     'unlabeled'            ,
     'ego vehicle'          ,
     'rectification border' ,
     'out of roi'           ,
     'static'               ,
     'dynamic'              ,
     'ground'               ,
     'road'                 ,
     'sidewalk'             ,
     'parking'              ,
     'rail track'           ,
     'building'             ,
     'wall'                 ,
     'fence'                ,
     'guard rail'           ,
     'bridge'               ,
     'tunnel'               ,
     'pole'                 ,
     'polegroup'            ,
     'traffic light'        ,
     'traffic sign'         ,
     'vegetation'           ,
     'terrain'              ,
     'sky'                  ,
     'person'               ,
     'rider'                ,
     'car'                  ,
     'truck'                ,
     'bus'                  ,
     'caravan'              ,
     'trailer'              ,
     'train'                ,
     'motorcycle'           ,
     'bicycle'              ,
     'license plate'
]

void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

codes = np.array(all_classes)
name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['unlabeled']
def acc(input, target):
    '''
    Accuracy with all but the unlabelled
    '''
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
def filter_labels(y, labels):
    """Utility used to create a mask to filter values in a tensor.

    Args:
        y (list, torch.Tensor): tensor where each element is a numeric integer
            representing a label.
        labels (list, torch.Tensor): filter used to generate the mask. For each
            value in ``y`` its mask will be "1" if its value is in ``labels``,
            "0" otherwise".

    Shape:
        y: can have any shape. Usually will be :math:`(N, S)` or :math:`(S)`,
            containing `batch X samples` or just a list of `samples`.
        labels: a flatten list, or a 1D LongTensor.

    Returns:
        mask (torch.ByteTensor): a binary mask, with "1" with the respective value from ``y`` is
        in the ``labels`` filter.

    Example::

        >>> a = torch.LongTensor([[1,2,3],[1,1,2],[3,5,1]])
        >>> a
         1  2  3
         1  1  2
         3  5  1
        [torch.LongTensor of size 3x3]
        >>> classification.filter_labels(a, [1, 2, 5])
         1  1  0
         1  1  1
         0  1  1
        [torch.ByteTensor of size 3x3]
        >>> classification.filter_labels(a, torch.LongTensor([1]))
         1  0  0
         1  1  0
         0  0  1
        [torch.ByteTensor of size 3x3]
    """
    mapping = torch.zeros(y.size()).byte().cuda()

    for label in labels:
        mapping = mapping | (y == label).byte()

    return mapping

def acc_valid(input, target):
    '''
    Accuracy with Only the valid classes
    '''
    target = target.squeeze(1)
    mask = filter_labels(target, valid_classes)
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()


def iou1(input, target, n_class=len(codes)):
    target = target.squeeze(1)
    pred = input.argmax(dim=1)
    ret = 0
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds[target_inds]).float().sum()  # Cast to long to prevent overflows
        union = pred_inds.float().sum() + target_inds.float().sum() - intersection
        #         print(type(union))
        if union > 0:
            ret += (intersection / (union if union > 1 else 1.))
    return ret / n_class


def iou1_valid(input, target, n_class=len(codes)):
    target = target.squeeze(1)
    pred = input.argmax(dim=1)
    n_valid = len(valid_classes)

    ret = 0
    for cls in range(n_class):
        if cls in valid_classes:
            pred_inds = pred == cls
            target_inds = target == cls

            intersection = (pred_inds[target_inds]).float().sum()  # Cast to long to prevent overflows
            union = pred_inds.float().sum() + target_inds.float().sum() - intersection
            #         print(type(union))
            if union > 0:
                ret += (intersection / (union if union > 1 else 1.))
    return ret / n_valid
    
async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    out_mask = learn.predict(img)[0]
    # out_mask.save(path/'static'/f'mask.png', cmap=defaults.cmap)
    # return JSONResponse({'result': learn.predict(img)[0]})
    suffix = str(uuid.uuid1())

    plt.imsave(path/'static'/f'mask_{suffix}.png', image2np(out_mask.data), cmap='tab20', vmin=0, dpi=199)
    return JSONResponse({'result': f'mask_{suffix}.png'})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
